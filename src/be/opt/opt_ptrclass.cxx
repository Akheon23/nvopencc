/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//-*-c++-*-
#ifdef USE_PCH
#include "opt_pch.h"
#endif // USE_PCH
#pragma hdrstop

#include "defs.h"
#include "config_opt.h"
#include "opt_ptrclass.h"
#include "wn_dfsolve.h"
#include "be_symtab.h"        // for Find_Lda

// ====================================================================
//
// Description:
//
// This module implements pointer classification analysis to determine,
// using flow analysis, if a pointer points to global, local, constant,
// shared or parameter memory. It deals with the problem by relying on a
// forward data flow analysis formulated as follows:
//
// - all address taken operations are initialized to the kind of memory
//   that their symbols are stored in
// - all other pointers are initialized to top.
//
// Forward propagation them determines the kind of memory pointed to by
// each pointer, the meet operation moves down in the lattice which is
// as follows:
//
//           Top
//        / | | \\
//       g  l s c p
//       \  | | //
//         Bottom
//
// Note that if a pointer is uninitialized it may remain at Top and if
// a pointer may point to more than one memory space it may reach
// Bottom, the compiler should emit warnings if either condition is 
// encountered.
// 
// We have an extra issue for sm_20 because if the pointer class is not known
// then we can issue a cvta instruction after the lda and then use generic
// addressing.  But we don't want to do that unless we have to, because the
// performance is worse.  So we start by assuming that we will be able to
// resolve all the pointer uses.  If at the end (Apply time) we discover that
// we cannot resolve some pointer uses, then we set AllCanBeResolved to false
// and redo the analysis assuming that we will use generic addressing.
//
// Extend PTR_CLASS::Query to handle nodes other than ILOAD/ISTORE, so that
// cg can query the analysis on other nodes also.
// ====================================================================


static BOOL Tracing(void)
{
    return Get_Trace(TP_WOPT3, PTRCLASS_FLAG); 
}
 
//  Top         : T
//  Bottom      : B
//  Global      : G
//  Local       : L
//  Constant    : C
//  Shared      : S
//  Param       : P
//  Texture     : X
//  Generic     : R
char GetPtrClassesName(enum PtrClasses a)
{ 
    return "TBGLCSPXR"[(int)a]; 
}

// ====================================================================
//
// Description: Merge two pointer address spaces
//
// ====================================================================
enum PtrClasses MergePtrClasses(enum PtrClasses a, enum PtrClasses b) 
{
    if (a == Top || a == b)
        return b;
    if (b == Top)
        return a;
    return Bottom;
} // MergePtrClasses

// ====================================================================
//
// Description: Merge state with incoming state. Return TRUE if 
// state changed.
// ====================================================================
BOOL Pstate::Merge(Pstate *in)
{
    BOOL changed = FALSE;
    PtrClasses pclass;

    locmap_t *in_map = in->locmap;
    locmap_t::iterator itr, itr2;

    // For each location L from "in" Pstate:
    //      If L is absent in local state, insert into local state
    //      Else, merge L's ptrclass with local state's ptrclass.
    for (itr = in_map->begin(); itr != in_map->end(); itr++) {
        itr2 = locmap->find(itr->first);
        // not present in local state, insert
        if (itr2 == locmap->end()) {
            changed = TRUE;
            (*locmap)[itr->first] = itr->second;
        } else {
            // merge ptr classes. 
            pclass = MergePtrClasses(itr->second, itr2->second);
            if (pclass != itr2->second) {
                itr2->second = pclass;
                changed = TRUE;
            }
        }
    }
    // TRUE if state changed.
    return changed;
}// Pstate::Merge

// ====================================================================
//
// Description: Return TRUE if mtype can hold pointers.
// ====================================================================
static BOOL
Mtype_is_pointer_sized_integer(TYPE_ID mtype)
{
  // 32- or 64-bit sized integers
  if (mtype == MTYPE_U4 ||
      mtype == MTYPE_U8 ||
      mtype == MTYPE_I4 ||
      mtype == MTYPE_I8)
    return TRUE;
  
  return FALSE;
}

// ====================================================================
//
// Description: Return TRUE if ty is a pointer or can hold pointers.
// ====================================================================
static BOOL
TY_can_be_pointer_type(TY_IDX ty)
{
  if (TY_kind(ty) == KIND_POINTER) return TRUE;

  // Also check pointer-sized integers because they may contain pointers.
  return Mtype_is_pointer_sized_integer(TY_mtype(ty));
}


// ====================================================================
//
// Description: Walk down a composite or scalar type. Insert each pointer
// variable found into the location map with the specified PtrClass.
// ====================================================================
void Pstate::WalkType(ST *root_st, TY_IDX ty, WN_OFFSET off, PtrClasses state)
{
    loc_t loc;
    FLD_ITER fld_iter;

    // found scalar pointer. Insert it into the location map
    if (TY_can_be_pointer_type(ty))
    {
        loc.Set(root_st, off);
        (*locmap)[loc] = state;
         if (Tracing())
            fprintf(TFile,"initgen %p %d %c\n", loc.sym, (int)loc.off, GetPtrClassesName(state));
    } else if (TY_kind(ty) == KIND_STRUCT) {
        // found structure, iterate over fields.
        fld_iter = Make_fld_iter(TY_fld(ty)); 
        do {
            FLD_HANDLE fld(fld_iter);
            // if pointer, insert into location map
            if (TY_can_be_pointer_type(FLD_type(fld))) {
                loc.Set(root_st, off + FLD_ofst(fld));
                (*locmap)[loc] = state;
                if (Tracing())
                    fprintf(TFile,"initgen %p %d %c\n", loc.sym, (int)loc.off, GetPtrClassesName(state));
            } else if (TY_kind(FLD_type(fld)) == KIND_STRUCT && 
                       TY_fld(FLD_type(fld)) != FLD_HANDLE()) 
            {
                // walk children fields.
                WalkType(root_st, FLD_type(fld), off + FLD_ofst(fld), state);
            }
        } while (!FLD_last_field(fld_iter++));
    }
} // Pstate::WalkType


// ====================================================================
//
// Description: Given a function entry node, initialize the state of the 
// pointers in variables to ptrclasses, as specified by CUDA semantics.
// ====================================================================
void Pstate::Initialize(WN *pu)
{
    WN *wn;
    ST *st;
    int ii;
    FmtAssert(WN_operator(pu) == OPR_FUNC_ENTRY, ("expect function entry node"));
    
    // TODO: bug fix : handle unions conservatively
    // Walk over the parameters
    for(ii = 0; ii <= (WN_kid_count(pu) - 4); ii++) {
        wn = WN_kid(pu, ii);
        FmtAssert((WN_operator(wn) == OPR_IDNAME), ("Formal argument must be OPR_IDNAME")); 
        st = WN_st(wn);
#if 1
        // sm2x could be generic, but for now always treat as global
        // as that helps ocg optimize better.
        WalkType(st, ST_type(st), 0, Global);
#else
        // sm1x has entry params point to global space, but sm2x is generic
        if (Target_ISA < TARGET_ISA_compute_20 ||
            OPT_Enable_Global_Ptr_Params_For_Sm2x) {
          WalkType(st, ST_type(st), 0, Global);
        }
        else {
          WalkType(st, ST_type(st), 0, Generic);
        }
#endif
    }

    // Walk over global symtab
    // Look at pointer variables in constant and global space. They come
    // from the host code. The only space they can point to is the global
    // space. Just force them to point to global space in the locmap.
    FOREACH_SYMBOL(GLOBAL_SYMTAB, st, ii) {
        // pointers in constant memory must point to global memory
        if (ST_in_constant_mem(st) || ST_in_global_mem(st)) {
            WalkType(st, ST_type(st), 0, Global);
        }
    }
} // Pstate::Initialize

// ====================================================================
//
// Description: Reset the memory space for all pointers that are in
//              global and local symbol tables to bottom.
// ====================================================================
void Pstate::ResetToBottom(void)
{
    ST *st;
    int ii;

    FOREACH_SYMBOL(GLOBAL_SYMTAB, st, ii) {
      WalkType(st, ST_type(st), 0, Bottom);
    }
    FOREACH_SYMBOL(CURRENT_SYMTAB, st, ii) {
      WalkType(st, ST_type(st), 0, Bottom);
    }
} // Pstate::ResetToBottom

/////////////////////// start of PTR_CLASS methods
PTR_CLASS::PTR_CLASS(void)
{
    MEM_POOL_Initialize(&_ptr_pool, "ptr_class pool", FALSE);
    _expr_map = WN_MAP32_Create(&_ptr_pool);
    _applied = FALSE;
    AllCanBeResolved = TRUE;
}; // PTR_CLASS::PTR_CLASS

PTR_CLASS::~PTR_CLASS(void) 
{
    WN_MAP_Delete(_expr_map);
    MEM_POOL_Delete(&_ptr_pool);
}; // PTR_CLASS::~PTR_CLASS

// ====================================================================
//
// Description: Update the ptrclasses state attached to a WHIRL node.
//              Return TRUE if new state is different than old state
// ====================================================================
BOOL PTR_CLASS::UpdateExprMap(WN *wn, PtrClasses pstate)
{
    BOOL changed = FALSE;
    if (WN_MAP32_Get(_expr_map, wn) != pstate)
        changed = TRUE;
    WN_MAP32_Set(_expr_map, wn, (INT32)pstate);
    return changed;
}; // PTR_CLASS::UpdateExprMap
    
// ====================================================================
//
// Description: Apply the transfer function to "inp" and update "state"
// 
// ====================================================================
void PTR_CLASS::TransferFunction(WN *inp, Pstate *state)
{
    locmap_t::iterator itr;
    locmap_t *locmap;
    ST *st;
    loc_t loc;
    OPERATOR opr;
    INT32 t1, t2;
    enum PtrClasses pstate = Top;
    BOOL changed;

    changed = FALSE;

    locmap = state->locmap;

    // TODO: bug fix: handle calls conservatively.
    opr = WN_operator(inp);
    if (opr == OPR_LDA) {
        // TODO: bug fix: handle address taken of symbol that is
        // a pointer or can contain a pointer conservatively.
        st = WN_st(inp);
        if (ST_class(st) == CLASS_VAR) {
            switch (ST_memory_space(st)) {
            case MEMORY_GLOBAL:
                // if lda on sm20, will do cvta to put in generic space
                if (AllCanBeResolved || Target_ISA < TARGET_ISA_compute_20) {
                  pstate = Global;
                }
                break;
            case MEMORY_SHARED:
                // if lda on sm20, will do cvta to put in generic space
                if (AllCanBeResolved || Target_ISA < TARGET_ISA_compute_20) {
                  pstate = Shared;
                }
                break;
            case MEMORY_LOCAL:
                // if lda on sm20, will do cvta to put in generic space
                if (AllCanBeResolved || Target_ISA < TARGET_ISA_compute_20) {
                  pstate = Local;
                }
                break;
            case MEMORY_CONSTANT:
                pstate = Constant;
                break;
            case MEMORY_PARAM:
                pstate = Param;
                break;
            case MEMORY_TEXTURE:
                pstate = Texture;
                break;
            case MEMORY_UNKNOWN:
                FmtAssert(ST_sclass(st) == SCLASS_AUTO || 
                          // MEMORY_UNKNOWN for SCLASS_FORMAL indicates the 
                          // formal parameter is a parameter of a device routine
                          // which is the same as SCLASS_AUTO.
                          ST_sclass(st) == SCLASS_FORMAL, 
                          ("Must be a local variable"));
                // if lda on sm20, will do cvta to put in generic space
                if (AllCanBeResolved || Target_ISA < TARGET_ISA_compute_20) {
                  pstate = Local;
                }
                break;
            default:
                FmtAssert(0, ("Must have a known memory space"));
                pstate = Bottom;
                break;
            }
        } else if (ST_class(st) == CLASS_CONST) {
            if (Target_ISA < TARGET_ISA_compute_20) {
              pstate = Constant;
            } else {
              // Address of the const may be passed to a function call,
              // so cannot be in constant memory. Note that this
              // change must be consistent with the setting of
              // memory space for initialized local objects in 
              // exp_loadstore.cxx.
              pstate = Global;
            }
        } else if (ST_class(st) == CLASS_FUNC) {
            pstate = Bottom;
        } else {
            FmtAssert(0, ("Unknown ST_Class for LDA"));
            pstate = Global;
        }
        if (Tracing())
            fprintf(TFile, "LDA: gen %p %c\n", inp, GetPtrClassesName(pstate));
        changed |= UpdateExprMap(inp, pstate);
    } else if (opr == OPR_ADD || opr == OPR_SUB) {
            t1 = WN_MAP32_Get(_expr_map, WN_kid0(inp));
            t2 = WN_MAP32_Get(_expr_map, WN_kid1(inp));

            // If only one of the operands is a pointer, assume the memory
            // space of the expression to be that of the pointer. This is
            // necessary because we track memory space of pointer-sized
            // integers (due to the possibility they may contain pointers).
            // Without this extra check a mismatch in the state of the
            // pointer and the integer will lower it to Bottom.
            // But do not update the map, otherwise the state will again be
            // changed in the next pass.
            WN *lda0 = Find_Lda(WN_kid0(inp));
            WN *lda1 = Find_Lda(WN_kid1(inp));
            if (lda0 && !lda1)
              pstate = (PtrClasses)t1;
            else if (!lda0 && lda1)
              pstate = (PtrClasses)t2;
            else
              pstate = MergePtrClasses((PtrClasses)t1, (PtrClasses)t2);
            if (UpdateExprMap(inp, pstate)) {
                changed = TRUE;
                if (Tracing()) {
                    fprintf(TFile, "%s: gen %p %c\n", opr == OPR_ADD ? "ADD" : "SUB", inp,
                            GetPtrClassesName(pstate));
                }
            }
            
    } else if (opr == OPR_SELECT) {
            t1 = WN_MAP32_Get(_expr_map, WN_kid1(inp));
            t2 = WN_MAP32_Get(_expr_map, WN_kid2(inp));
            pstate = MergePtrClasses((PtrClasses)t1, (PtrClasses)t2);
            if (UpdateExprMap(inp, pstate)) {
                changed = TRUE;
                if (Tracing())
                    fprintf(TFile, "SELECT: gen %p %c\n", inp, GetPtrClassesName(pstate));
            }
            // If the select operation is on two pointers, then their
            // memory spaces need to match.
            if (_applied && pstate == Bottom)
              AllCanBeResolved = FALSE;
    } else if (opr == OPR_ASM_STMT) {
            pstate = Bottom;
            if (UpdateExprMap(inp, pstate)) {
                changed = TRUE;
                if (Tracing())
                    fprintf(TFile, "ASM_STMT: gen %p %c\n", inp, GetPtrClassesName(pstate));
            }
    } else if (opr == OPR_LDID) {
            // Do not track the sym for a preg, because the preg-id is
            // sufficient. Also uses of the same preg can use different
            // signedness, which will use different sym and break this search.
            // So search just based on preg-id.
            if (ST_class(WN_st(inp)) == CLASS_PREG)
              loc.Set(NULL, WN_offset(inp));
            else
              loc.Set(WN_st(inp), WN_offset(inp));
            itr = locmap->find(loc);
            if (itr != locmap->end()) {
                changed |= UpdateExprMap(inp, itr->second);
                 if (Tracing())
                     fprintf(TFile, "LDID: gen %p %c\n", inp, GetPtrClassesName(itr->second));
            }
            // If the memory space of a pointer cannot be resolved, then
            // we need to convert to generic addressing in cg.
            if (_applied && TY_kind(WN_ty(inp)) == KIND_POINTER &&
                (PtrClasses)WN_MAP32_Get(_expr_map, inp) == Bottom)
              AllCanBeResolved = FALSE;
    } else if (opr == OPR_STID) {
            // Do not track the sym for a preg, because the preg-id is
            // sufficient. Also uses of the same preg can use different
            // signedness, which will use different sym and break this search.
            // So search just based on preg-id.
            if (ST_class(WN_st(inp)) == CLASS_PREG)
              loc.Set(NULL, WN_offset(inp));
            else
              loc.Set(WN_st(inp), WN_offset(inp));
            itr = locmap->find(loc);
            pstate = (PtrClasses )WN_MAP32_Get(_expr_map, WN_kid0(inp));
            if (itr != locmap->end()) {
                changed |= (pstate != itr->second);
                if (Tracing()) {
                    fprintf(TFile,"STID: kill %p %d %c\n", WN_st(inp), (int)WN_offset(inp), 
                            GetPtrClassesName(itr->second));
                    fprintf(TFile,"STID: gen %p %d %c\n", WN_st(inp), (int)WN_offset(inp), 
                            GetPtrClassesName(pstate));
                }
                itr->second = pstate;
            } else if (pstate != Top) {// default state of whirl node is Top (0).
                // create on demand.
                (*locmap)[loc] = pstate;
                changed = TRUE;
                if (Tracing()) {
                    fprintf(TFile,"STID: gen %p %d %c\n", WN_st(inp), (int)WN_offset(inp),
                            GetPtrClassesName(pstate));
                }
            }
    } else if (opr == OPR_CALL || opr == OPR_ICALL) {
          // Conservatively assuming that a function call can assign to any pointer.
          // Therefore reset memory spaces of all pointers to bottom.
          state->ResetToBottom();
          if (_applied) {
            AllCanBeResolved = FALSE;
          }
    } else if ((opr == OPR_CVT &&
                Mtype_is_pointer_sized_integer(WN_rtype(inp))) ||
               (opr == OPR_CVTL && WN_cvtl_bits(inp) >= 32)) {
          // Propagate state through type conversions to pointer-sized integers.
          pstate = (PtrClasses )WN_MAP32_Get(_expr_map, WN_kid0(inp));
          if (UpdateExprMap(inp, pstate)) {
            changed = TRUE;
            if (Tracing())
              fprintf(TFile, "%s: gen %p %c\n",
                      opr == OPR_CVT ? "CVT" : "CVTL", inp,
                      GetPtrClassesName(pstate));
          }
    } else if (opr == OPR_ILOAD) {
      if (_applied) {
        pstate = (PtrClasses)WN_MAP32_Get(_expr_map, WN_kid0(inp));
        if (pstate == Top || pstate == Bottom)
          AllCanBeResolved = FALSE;
      }
    } else if (opr == OPR_ISTORE) {
      if (_applied) {
        pstate = (PtrClasses)WN_MAP32_Get(_expr_map, WN_kid1(inp));
        if (pstate == Top || pstate == Bottom)
          AllCanBeResolved = FALSE;
      }
    }
}; // PTR_CLASS::TransferFunction

void PTR_CLASS::Apply (WN *wn, Pstate *state)
{ 
  _applied = TRUE;
}

// ====================================================================
//
// Description: Do the ptr space analysis on the given function.
// 
// ====================================================================
void PTR_CLASS::Compute(WN *pu) 
{
    Pstate *initState = new Pstate();
    initState->Initialize(pu);
    DFSolver<PTR_CLASS, Pstate> solver(this);
    solver.Solve(TRUE, pu, initState);
    // The initial solver is done optimistically, marking the lda's with
    // ptrclass info so can resolve the uses to specific pstate.  
    // If found some iload/istore that cannot be resolved, 
    // redo analysis so that lda conservatively starts with generic, 
    // and then uses will be generic.
    if ( ! AllCanBeResolved && Target_ISA >= TARGET_ISA_compute_20) {
      DevWarn("redo ptrclass solver");
      WN_MAP_Delete(_expr_map); // recreate expr_map
      _expr_map = WN_MAP32_Create(&_ptr_pool);
      DFSolver<PTR_CLASS, Pstate> solver(this);
      solver.Solve(TRUE, pu, initState);
    }
    delete initState;
}; // PTR_CLASS::Compute

// ====================================================================
//
// Description: For a given WN, return its PtrClass
// 
// ====================================================================
enum PtrClasses PTR_CLASS::Query(WN *inp)
{
    WN *wn;
    PtrClasses pstate;
    OPERATOR opr = WN_operator(inp);
            
    if (opr == OPR_ILOAD)
      wn = WN_kid0(inp);
    else if (opr == OPR_ISTORE)
      wn = WN_kid1(inp);
    else
      wn = inp;
    pstate = (PtrClasses)WN_MAP32_Get(_expr_map, wn);
    if (pstate == Top || /* error condition? Top indicates uniniatialized memory */
        pstate == Generic /* incoming pointer param for >= sm_20 */ )
        pstate = Bottom;
    return pstate;
}; // PTR_CLASS::Query
