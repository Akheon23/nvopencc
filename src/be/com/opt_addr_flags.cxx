/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/*
 * Copyright 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
 */

/*

  Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.

  This program is free software; you can redistribute it and/or modify it
  under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it would be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

  Further, this software is distributed without any warranty that it is
  free of the rightful claim of any third person regarding infringement 
  or the like.  Any license provided herein, whether implied or 
  otherwise, applies only to this software file.  Patent licenses, if 
  any, provided herein do not apply to combinations of this program with 
  other software, or any other product whatsoever.  

  You should have received a copy of the GNU General Public License along
  with this program; if not, write the Free Software Foundation, Inc., 59
  Temple Place - Suite 330, Boston MA 02111-1307, USA.

  Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
  Mountain View, CA 94043, or:

  http://www.sgi.com

  For further information regarding this notice, see:

  http://oss.sgi.com/projects/GenInfo/NoticeExplan

*/


#include "opt_alias_interface.h"  
#include "opt_points_to.h"        
#include "config_opt.h"           
#include "symtab_access.h"
#include "wn.h"
#include "be_symtab.h"
#include "erbe.h"

// Adjust address flags

static BOOL suppress_all_warnings;
static BOOL trace = FALSE;
static SRCPOS current_srcpos;

#ifdef TARG_NVISA
ST_MEMORY
Find_Memory_Pointed_To (WN *wn)
{
    ST_MEMORY m = MEMORY_UNKNOWN;
    if (Find_Lda(wn)) {
        // mark what is pointing to
        WN* lda = Find_Lda (wn);
        ST *lda_st = WN_st(lda);
        if (WN_operator(lda) == OPR_LDID) {
          if (ST_sclass(lda_st) == SCLASS_FORMAL && 
              (ST_in_shared_mem(lda_st) || ST_in_param_mem(lda_st))) 
          {
            // not an lda, but is ldid of entry shared parameter pointer,
            // which must point to global.
            m = MEMORY_GLOBAL;
          }
          else if ((ST_sclass(lda_st) == SCLASS_AUTO
               || ST_sclass(lda_st) == SCLASS_PSTATIC)
            && TY_kind(WN_ty(lda)) == KIND_POINTER
            && BE_ST_memory_pointed_to(lda_st) != MEMORY_UNKNOWN)
          {
            // is local variable that has info about what it points to
            m = BE_ST_memory_pointed_to(lda_st);
          }
        }
        else {
          m = ST_memory_space(lda_st);
          if (m == MEMORY_UNKNOWN && ST_sclass(lda_st) == SCLASS_AUTO) {
            m = MEMORY_LOCAL; // stack variable
          }
        }
    }
    else if (Find_Pointer_Ldid(wn)) {
        WN* ldid = Find_Pointer_Ldid (wn);
        ST *ldid_st = WN_st(ldid);
        if (ST_sclass(ldid_st) == SCLASS_FORMAL && 
            (ST_in_shared_mem(ldid_st) || ST_in_param_mem(ldid_st))) {
            // not an lda, but is ldid of entry shared parameter pointer,
            // which must point to global.
            m = MEMORY_GLOBAL;
        }
        else {
            // copy memory_pointed_to
            m = BE_ST_memory_pointed_to(ldid_st);
        }
    }
    return m;
}
#endif

// wn is an actual parameter.  Search for LDAs under the expr that 
// are not consumed by an ILOAD, and set their addr_saved flag.
// warn is TRUE iff we should issue a DevWarn for each ST whose addr_saved
// flag we set.
void
Set_addr_saved_expr(WN *wn, BOOL warn, BOOL for_cg)
{
  OPCODE opc = WN_opcode(wn);
  Is_True(OPCODE_is_expression(opc),
	  ("Update_addr_saved: opcode must be expression"));

#ifdef TARG_NVISA
  // only ignore if LDA directly under ILOAD,
  // cause if other expression (like ILOAD(add(lda,mul))
  // then will not be simple ldid in cg.
  if (OPCODE_operator(opc) == OPR_ILOAD && WN_operator(WN_kid(wn,0)) == OPR_LDA)
    return;	// ignore
#else
  if (OPCODE_is_load(opc))
    return;
#endif

  if (OPCODE_operator(opc) == OPR_LDA) {
    ST *st = WN_st(wn);
    if (ST_class(st) == CLASS_VAR &&
	!ST_addr_saved(st)) {
      Set_ST_addr_saved(st);
      if (warn && !suppress_all_warnings)
	DevWarn("Set_addr_saved_expr: addr_saved flag of ST (%s) should be set.", 
		ST_name(st));
    }
  }
#ifdef TARG_NVISA
  // If accessing larger area than symbol,
  // must be a cast, and requires accessing memory location.
  // Also requires memory location if casting to a different struct
  // since offsets may not match (could be okay if all field offsets 
  // were same, but that case is rare so we'll be conservative).
  // Other targets can handle this because var symbols have memory location,
  // but we try to map remaining vars to registers later in cgexp.
  // So need to mark that this symbol cannot be put in a reg;
  // easiest to reuse addr_saved flag for this purpose, as that flag is
  // already checked, and can think of this cast as being an implicit use
  // of the addr.
  else if (OPCODE_operator(opc) == OPR_LDID && for_cg) {
    ST *st = WN_st(wn);
    if (ST_class(st) == CLASS_VAR && !ST_addr_saved(st)
      && !TY_has_union(ST_type(st)))
    {
      TY_IDX op_ty_idx = WN_ty(wn);
      TY_IDX ld_ty_idx = ST_type(st);

      // If we are reading field of structure, use field type
      if (Is_Structure_Type(op_ty_idx) && WN_field_id(wn) != 0) {
        op_ty_idx = Get_Field_Type(op_ty_idx, WN_field_id(wn));
      }

      if (Is_Structure_Type(ld_ty_idx)
          && Is_Structure_Type(op_ty_idx)
          && ! TY_are_equivalent(ld_ty_idx, op_ty_idx))
      {
        DevWarn("set addr_saved on ldid that accesses different struct");
        Set_ST_addr_saved(st);
      }
      else if (MTYPE_byte_size(OPCODE_desc(opc)) > TY_size(op_ty_idx))
      {
        DevWarn("set addr_saved on ldid that accesses larger area");
        Set_ST_addr_saved(st);
      }
      else if (Is_Structure_Type(ld_ty_idx)
        && !Is_Structure_Type(WN_ty(wn))
        && MTYPE_byte_size(OPCODE_desc(opc)) == TY_size(ld_ty_idx))
      {
        DevWarn("set addr_saved on ldid that accesses whole struct as scalar");
        Set_ST_addr_saved(st);
      }
    }
  }
  else if (OPCODE_operator(opc) == OPR_SELECT && for_cg) {
    if (OPT_Old_Mem_Ptr_Analysis) {
      // check that the memory on each side of the select matches
      ST_MEMORY m1 = Find_Memory_Pointed_To (WN_kid1(wn));
      ST_MEMORY m2 = Find_Memory_Pointed_To (WN_kid2(wn));
      if (m1 != m2 && m1 != MEMORY_UNKNOWN && m2 != MEMORY_UNKNOWN) {
        ErrMsgSrcpos (EC_Conflicting_Memory_Ptr, current_srcpos);
      }
    }
  }
#endif
  if (OPCODE_operator(opc) == OPR_COMMA) {
    	Set_addr_saved_stmt(WN_kid(wn,0), warn, for_cg);
    	Set_addr_saved_expr(WN_kid(wn,1), warn, for_cg);
	return;
  }
  if (OPCODE_operator(opc) == OPR_RCOMMA) {
    	Set_addr_saved_expr(WN_kid(wn,0), warn, for_cg);
    	Set_addr_saved_stmt(WN_kid(wn,1), warn, for_cg);
	return;
  }
#ifdef KEY // only LDAs from kid 0 of ARRAY and ARRSECTION are relevant
  if (OPCODE_operator(opc) == OPR_ARRAY || 
      OPCODE_operator(opc) == OPR_ARRSECTION)
    Set_addr_saved_expr(WN_kid0(wn), warn, for_cg);
  else
#endif
  for (INT i = 0; i < WN_kid_count(wn); i++) 
    Set_addr_saved_expr(WN_kid(wn,i), warn, for_cg);
}

void 
Set_addr_saved_stmt(WN *wn, BOOL use_passed_not_saved, BOOL for_cg)
{
  if (wn == NULL) return;	
  OPCODE opc = WN_opcode(wn);
  if (WN_Get_Linenum(wn)) // only reset if has value
    current_srcpos = WN_Get_Linenum(wn);

  if (OPCODE_is_call(opc)
#ifdef KEY
      || OPCODE_operator(opc) == OPR_PURE_CALL_OP
#endif
      ) {
    for (INT32 i = 0; i < WN_kid_count(wn); i++) {
      WN *actual = WN_actual(wn,i);
      // Question: What justification could there be for the
      // following line? Answer: It is a dangerous but cheap hack to
      // avoid processing the function address kid of ICALL as if it
      // were a parameter, which would otherwise happen because
      // WN_actual is naively implemented as WN_kid, with no check
      // for ICALL or other kinds of calls. We count on alias
      // classification or some other relatively conservative phase
      // to assert that the parameters are all PARM nodes.

      // Answer2:  WN_actual() does not guarantee returning a PARM node.
      // In this analysis, we don't care about the function address 
      // because it will not affect setting of addr saved.
      // Consider a direct call to FUNC and indiret call to FUNC should
      // be equivalent, although there is an extra function addr kid to
      // the indirect call.

      if (WN_operator(actual) != OPR_PARM) continue;
#ifdef TARG_NVISA
      // we transform MPARM(MLOAD(LDA)) to simple LDID when lowering 
      // to avoid the LDA penalty, so ignore the LDA under the MPARM.
      if (WN_opcode(actual) == OPC_MPARM) continue;
#endif
      if (!use_passed_not_saved ||
	  !WN_Parm_Passed_Not_Saved(actual))
	Set_addr_saved_expr(WN_kid0(actual), FALSE);
    }
    return;
  }

  switch (OPCODE_operator(opc)) {
  case OPR_FORWARD_BARRIER:
  case OPR_BACKWARD_BARRIER:
  case OPR_ALLOCA:
  case OPR_DEALLOCA:
    return;
  }

  if (OPCODE_is_black_box(opc)) 
    return;
  
#ifdef TARG_NVISA
  // See earlier comment about LDID
  // only do this if for_cg, least we get conflicts between passes
  if (OPCODE_operator(opc) == OPR_STID && for_cg) {
    ST *st = WN_st(wn);
    if (ST_class(st) == CLASS_VAR && !ST_addr_saved(st)
      && !TY_has_union(ST_type(st)))
    {
      TY_IDX op_ty_idx = WN_ty(wn);
      TY_IDX st_ty_idx = ST_type(st);

      // If we are writing field of structure, use field type
      if (Is_Structure_Type(op_ty_idx) && (WN_field_id(wn) != 0)) {
        op_ty_idx = Get_Field_Type(op_ty_idx, WN_field_id(wn));
      }

      if (Is_Structure_Type(st_ty_idx)
          && Is_Structure_Type(op_ty_idx)
          && ! TY_are_equivalent(st_ty_idx, op_ty_idx))
      {
        DevWarn("set addr_saved on stid that accesses different struct");
        Set_ST_addr_saved(st);
      }
      else if (MTYPE_byte_size(OPCODE_desc(opc)) > TY_size(op_ty_idx))
      {
        DevWarn("set addr_saved on stid that accesses larger area");
        Set_ST_addr_saved(st);
      }
      else if (Is_Structure_Type(st_ty_idx)
        && !Is_Structure_Type(WN_ty(wn))
        && MTYPE_byte_size(OPCODE_desc(opc)) == TY_size(st_ty_idx))
      {
        DevWarn("set addr_saved on stid that accesses whole struct as scalar");
        Set_ST_addr_saved(st);
      }
    }

    if (OPT_Old_Mem_Ptr_Analysis) {
      // want to track what memory is being pointed to
      // so -O0 compiles can find this info.
      // For non-simple types (structs), 
      // will mark whole object as pointing to same memory space 
      // (no guarantee of this, but probably safe);
      // Arrays are trickier though because they can be treated as pointers,
      // but their elements already have a declared memory space.
      // But can have array of pointers, so only track arrays when
      // element is pointer type.
      // then will get conflict error if not consistent.
      // If determine that multiple memory spaces are pointed to,
      // will emit warning.
      if (TY_kind(ST_type(st)) != KIND_ARRAY 
       || TY_kind(WN_ty(wn)) == KIND_POINTER)
      {
        ST_MEMORY m = Find_Memory_Pointed_To (WN_kid0(wn));
        if (m != MEMORY_UNKNOWN) {
          if (ST_class(st) != CLASS_PREG) {
            if (BE_ST_memory_pointed_to(st) == MEMORY_UNKNOWN) {
              Set_BE_ST_memory_pointed_to(st, m);
              if (trace)
                fprintf(TFile, "set memory space for %s to %d\n", ST_name(st), m);
            } else if (BE_ST_memory_pointed_to(st) != m) {
              // e.g. if same pointer used to point to multiple spaces.
              // also if different fields of same struct point to multiple spaces.
              ErrMsgSrcpos (EC_Conflicting_Memory_Ptr, current_srcpos);
              Set_BE_ST_memory_pointed_to(st, MEMORY_UNKNOWN);
            }
          }
          else { // preg
            PREG_NUM preg = WN_store_offset(wn);
            // Check if the preg is not dedicated. Don't have to  do anything
            // for dedicated pregs. Bug 493411.
            if(!Preg_Is_Dedicated(preg)) {
              if (Preg_memory_pointed_to(preg) == MEMORY_UNKNOWN) {
                Set_Preg_memory_pointed_to(preg, m);
                if (trace)
                  fprintf(TFile, "set memory space for preg %d to %d\n", preg, m);
              } else if (Preg_memory_pointed_to(preg) != m) {
                // e.g. if same pointer used to point to multiple spaces.
                // also if different fields of same struct point to multiple spaces.
                ErrMsgSrcpos (EC_Conflicting_Memory_Ptr, current_srcpos);
                Set_Preg_memory_pointed_to(preg, MEMORY_UNKNOWN);
              }
            }
          }
        }
      }
    }
  }
#endif

  if (opc == OPC_BLOCK) {
    for (WN *stmt = WN_first(wn); stmt != NULL; stmt = WN_next(stmt))  
      Set_addr_saved_stmt(stmt, use_passed_not_saved, for_cg);
  } else {
    for (INT i = 0; i < WN_kid_count(wn); i++) {
      Set_addr_saved_stmt(WN_kid(wn,i), use_passed_not_saved, for_cg);
    }
  }
}


// For debugging only!
void 
Recompute_addr_saved_stmt(WN *wn, BOOL for_cg)
{
  if (wn == NULL) return;	
  OPCODE opc = WN_opcode(wn);
  if (WN_Get_Linenum(wn)) // only reset if has value
    current_srcpos = WN_Get_Linenum(wn);

  if (OPCODE_is_store(opc)) {
    // the RHS expr of any store is kid0
    // Any idea on how to assert?
    Set_addr_saved_expr(WN_kid0(wn), TRUE, for_cg);
#ifdef TARG_NVISA
    // Other targets should probably do this too,
    // but just to be safe only do it for our target.
    if (OPCODE_operator(opc) == OPR_ISTORE) {
      // can be address on lhs too
      Set_addr_saved_expr(WN_kid1(wn), TRUE, for_cg);
    }
#endif
  }
#ifdef TARG_NVISA
  else if (OPCODE_operator(opc) == OPR_ASM_STMT) {
    // need to search input nodes
    for (INT i = 2; i < WN_kid_count(wn); ++i) {
      Set_addr_saved_expr(WN_kid(wn, i), TRUE, for_cg);
    }
  }
#endif

  if (OPCODE_is_black_box(opc)) 
    return;
  
  if (opc == OPC_BLOCK) {
    for (WN *stmt = WN_first(wn); stmt != NULL; stmt = WN_next(stmt))  
      Recompute_addr_saved_stmt(stmt, for_cg);
#ifdef TARG_NVISA
  } else if (OPCODE_is_expression(opc)) {
      // search all expr nodes
      Set_addr_saved_expr(wn, TRUE, for_cg);
#endif // NVISA
  } else {
    for (INT i = 0; i < WN_kid_count(wn); i++) {
      Recompute_addr_saved_stmt(WN_kid(wn,i), for_cg);
    }
  }
}


#ifdef Is_True_On

static void Verify_addr_flags_stmt(WN *wn);

static void
Verify_addr_saved_expr(WN *wn)
{
  OPCODE opc = WN_opcode(wn);
  Is_True(OPCODE_is_expression(opc),
	  ("Update_addr_saved: opcode must be expression"));

  if (OPCODE_is_load(opc))
    return;

  if (OPCODE_operator(opc) == OPR_LDA) {
    ST *st = WN_st(wn);
    if (ST_class(st) == CLASS_VAR &&
	!ST_addr_saved(st)) {
      FmtAssert(TRUE, ("PU_adjust_addr_flags:  ST %s should be addr_saved.\n",
		       ST_name(st)));
    }
  }
  if (OPCODE_operator(opc) == OPR_COMMA) {
    	Verify_addr_flags_stmt(WN_kid(wn,0));
    	Verify_addr_saved_expr(WN_kid(wn,1));
	return;
  }
  if (OPCODE_operator(opc) == OPR_RCOMMA) {
    	Verify_addr_saved_expr(WN_kid(wn,0));
    	Verify_addr_flags_stmt(WN_kid(wn,1));
	return;
  }
  for (INT i = 0; i < WN_kid_count(wn); i++) 
    Verify_addr_saved_expr(WN_kid(wn,i));
}

static void 
Verify_addr_flags_stmt(WN *wn)
{
  if (wn == NULL) return;	
  OPCODE opc = WN_opcode(wn);

  if (OPCODE_is_store(opc)) {
    // the RHS expr of any store is kid0
    // Any idea on how to assert?
    Verify_addr_saved_expr(WN_kid0(wn));
  }

  switch (OPCODE_operator(opc)) {
  case OPR_FORWARD_BARRIER:
  case OPR_BACKWARD_BARRIER:
  case OPR_ALLOCA:
  case OPR_DEALLOCA:
    return;
  }

  if (OPCODE_is_black_box(opc)) 
    return;
  
  if (opc == OPC_BLOCK) {
    for (WN *stmt = WN_first(wn); stmt != NULL; stmt = WN_next(stmt))  
      Verify_addr_flags_stmt(stmt);
  } else {
    for (INT i = 0; i < WN_kid_count(wn); i++) {
      Verify_addr_flags_stmt(WN_kid(wn,i));
    }
  }
}
#endif


void
PU_adjust_addr_flags(ST* pu_st, WN *wn, BOOL for_cg)
{
  suppress_all_warnings = FALSE;
  trace = Get_Trace (TP_DATALAYOUT, 0x10);
#if 1 // Fix 10-26-2002: Enhancement to reset addr_saved flag before Mainopt
  Set_Error_Phase("PU_adjust_addr_flags");
#endif
          // PV 682222: the MP lowerer may introduce LDA's on privatized
	  // ST's which require setting their addr_saved flag before WOPT.
	  // So the MP lowerer sets the PU_needs_addr_flag_adjust bit.
  BOOL has_privatization_LDAs = BE_ST_pu_needs_addr_flag_adjust(pu_st);

  if (OPT_recompute_addr_flags || has_privatization_LDAs) {
    if (!OPT_recompute_addr_flags)
      suppress_all_warnings = TRUE; // LDAs from privatization are OK

#if 1 // Fix 10-26-2002: Enhancement to reset addr_saved flag before Mainopt 
    Clear_local_symtab_addr_flags(Scope_tab[CURRENT_SYMTAB]);
#endif
    Recompute_addr_saved_stmt(wn, for_cg);
  }

  if (BE_ST_pu_needs_addr_flag_adjust(pu_st))
    Clear_BE_ST_pu_needs_addr_flag_adjust(pu_st);

#ifdef Is_True_On
  if (!PU_smart_addr_analysis(Pu_Table[ST_pu(pu_st)]))
    Verify_addr_flags_stmt(wn);
#endif

  // Adjust addr_saved from actual parameters for non-Fortran programs.
  if (!Is_FORTRAN()) {
    PU& pu = Pu_Table[ST_pu(pu_st)];
    Set_addr_saved_stmt(wn,
			CXX_Alias_Const || 
			(OPT_IPA_addr_analysis && PU_ipa_addr_analysis(pu)),
			for_cg);
  }
}
