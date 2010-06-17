/*
 *  Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */
/*
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of version 2 of the GNU General Public License as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it would be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 *  Further, this software is distributed without any warranty that it is
 *  free of the rightful claim of any third person regarding infringement
 *  or the like.  Any license provided herein, whether implied or
 *  otherwise, applies only to this software file.  Patent licenses, if
 *  any, provided herein do not apply to combinations of this program with
 *  other software, or any other product whatsoever.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write the Free Software Foundation, Inc., 59
 *  Temple Place - Suite 330, Boston MA 02111-1307, USA.
 */

// Implement new .param syntax for calling convention.
//
// The model for most targets is to pass parameters in registers or stack 
// locations, so in whirl we lower formals and actuals into ldid/stid of
// dedicated pregs for the parameter registers.  These typically become copies
// to/from general registers in OPs.  This is what was used for earlier ptx isa.
//
// For compute_20 and beyond, we have a new syntax that refers to parameters
// by name, rather than by passing explicit registers.  This poses some 
// problems.  For one, we don't have access to the actual parameter names that
// the user gave, as they are only visible inside the function, not at the
// the caller.  We could add the names to the symtab in the front end, 
// but some prototypes can have unnamed parameters, return values don't have 
// names, doing so may have side effects, and we still need a way of 
// associating each name with its place in the param list.  So instead, 
// we just create dummy numbered names at cg time, which will be easier 
// to associate.
//
// Another issue is that we have to map the parameter registers back to these
// dummy names, and to do that we need to know what function the parameter 
// registers are associated with.  That is easy for formals, but when storing
// actuals we need to look ahead to the call.  Therefore we wait and do this
// after the whole cfg is created, so that we can use the BB_call information,
// rather than trying to do it on the fly in cgexp.  An alternative would have
// been to change the whirl to not refer to dedicated registers, but that would
// have affected target-independent code.

#include <string.h>
#include <map>
#include "defs.h"
#include "tracing.h"
#include "errors.h"
#include "wn.h"
#include "bb.h"
#include "op.h"
#include "tn.h"
#include "targ_sim.h"
#include "variants.h"
#include "cgexp_internals.h"
#include "data_layout.h"

static BOOL tracing = FALSE;

// Assume that the string will be used before this is called again,
// so okay to use temporary static buffer.
// Use a different name for formal vs actual,
// so that recursive calls will see a copy vs reuse of name.
extern char *
Get_Retval_Name (const char *fname, BOOL is_formal)
{
  static char buf[2048];
  sprintf(buf, (is_formal ? "__cudaretf_%s" : "__cudareta_%s"), fname);
  return buf;
}
extern char *
Get_Param_Name (const char *fname, INT index, BOOL is_formal)
{
  static char buf[2048];
  sprintf(buf, (is_formal ? "__cudaparmf%d_%s" : "__cudaparma%d_%s"), 
    index, fname);
  return buf;
}

// Return a mangled name for TY.
static const char *
Mangle_Type (TY_IDX ty)
{
  TYPE_ID mtype = TY_mtype(ty);
  static char buf[16];
  char *ptr = buf;

  if (MTYPE_is_unsigned(mtype)) *ptr++ = 'U';
  switch (mtype) {
    case MTYPE_B:
      *ptr++ = 'b';
      break;
    case MTYPE_I1: // widen to 32-bits
    case MTYPE_I2:
    case MTYPE_I4:
    case MTYPE_U1:
    case MTYPE_U2:
    case MTYPE_U4:
      *ptr++ = 'i';
      break;
    case MTYPE_I8:
    case MTYPE_U8:
      *ptr++ = 'l';
      break;
    case MTYPE_F4:
      *ptr++ = 'f';
      break;
    case MTYPE_F8:
      *ptr++ = 'd';
      break;
    case MTYPE_V:
      *ptr++ = 'v';
      break;
    case MTYPE_M: // composite types
      sprintf(buf, "c%d_%d", TY_align(ty), (INT)TY_size(ty));
      break;
  }

  if (mtype != MTYPE_M) *ptr = '\0';

  return buf;
}

// WN must be an indirect call. Return a mangled name based on the
// prototype of the callee.
static STR_IDX
Create_Mangled_Name (const WN *wn)
{
  Is_True(WN_operator(wn) == OPR_ICALL, ("Indirect call expected"));

  TY_IDX func_ty = WN_ty(wn);
  TY_IDX ty = TY_ret_type(func_ty);
  // We could make a separate pass to calculate the size of the buffer
  // required to hold the mangled name, but this static array is probably
  // sufficient, without requiring a separate pass.
  static char buf[2048];

  // emit mangled return type
  sprintf(buf, "fproto_%s", Mangle_Type(ty));

  // now process the params
  TYLIST_IDX tl = TY_parms(func_ty);
  for (; TYLIST_ty(tl); tl = TYLIST_next(tl)) {
      ty = TYLIST_ty(tl);
      sprintf(buf+strlen(buf), "_%s", Mangle_Type(ty));
  }

  return Save_Str(buf);
}

// This function returns a STR_IDX instead of the string from the
// string table because the pointer is not guaranteed to be valid at the
// point of use.
extern STR_IDX
Get_Called_Func_Name (const WN *wn)
{
  STR_IDX result;
  TY_IDX func_ty = WN_ty(wn);
  // maintain a map for quick lookup of names for a function prototype.
  static std::map<TY_IDX, STR_IDX> prototype_to_name_map;
  std::map<TY_IDX, STR_IDX>::iterator iter = 
                                prototype_to_name_map.find(func_ty);
  if (iter == prototype_to_name_map.end())
    result = prototype_to_name_map[func_ty] = Create_Mangled_Name(wn);
  else
    result = iter->second;

  return result;
}

// create st for formal param;
// will not get emitted, as defined in prototype.
static ST*
Create_Formal_Param_ST (const char *name, TY_IDX ty)
{
  // want to reuse if already defined
  ST *st;
  INT i;
  FOREACH_SYMBOL (CURRENT_SYMTAB, st, i) {
    if (strcmp(ST_name(st), name) == 0) {
      return st; // found match
    }
  }
  st = New_ST(CURRENT_SYMTAB);
  ST_Init(st, Save_Str(name), CLASS_VAR, SCLASS_FORMAL_REF, EXPORT_LOCAL, ty);
  return st;
}

// create st for actual param;
// should get emitted in caller.
static ST*
Create_Actual_Param_ST (const char *name, TY_IDX ty)
{
  // want to reuse if already defined
  ST *st;
  INT i;
  FOREACH_SYMBOL (CURRENT_SYMTAB, st, i) {
    if (strcmp(ST_name(st), name) == 0) {
      return st; // found match
    }
  }
  st = New_ST(CURRENT_SYMTAB);
  ST_Init(st, Save_Str(name), CLASS_VAR, SCLASS_PSTATIC, EXPORT_LOCAL, ty);
  Set_ST_in_param_mem(st);
  // allocate object so know to emit it
  // (note that this leaves stack allocated,
  // but that's okay since we ignore stack).
  Allocate_Object(st);
  return st;
}

// return preg associated with return register
static PREG_NUM
Retval_Reg_To_Preg (ISA_REGISTER_CLASS rc, REGISTER reg)
{
  PREG_NUM pnum;
  INT rnum = REGISTER_machine_id(rc,reg);
  switch (rc) {
  case ISA_REGISTER_CLASS_integer:
    pnum = First_Int32_Preg_Return_Offset
      + (rnum - ABI_PROPERTY_integer_func_val_First_Register);
    break;
  case ISA_REGISTER_CLASS_integer64:
    pnum = First_Int64_Preg_Return_Offset
      + (rnum - ABI_PROPERTY_integer64_func_val_First_Register);
    break;
  case ISA_REGISTER_CLASS_float:
    pnum = First_Float32_Preg_Return_Offset
      + (rnum - ABI_PROPERTY_float_func_val_First_Register);
    break;
  case ISA_REGISTER_CLASS_float64:
    pnum = First_Float64_Preg_Return_Offset
      + (rnum - ABI_PROPERTY_float64_func_val_First_Register);
    break;
  default:
    FmtAssert(FALSE, ("unexpected class"));
  }
  return pnum;
}

// return preg associated with param register
static PREG_NUM
Param_Reg_To_Preg (ISA_REGISTER_CLASS rc, REGISTER reg)
{
  PREG_NUM pnum;
  INT rnum = REGISTER_machine_id(rc,reg);
  switch (rc) {
  case ISA_REGISTER_CLASS_integer:
    pnum = First_Int32_Preg_Param_Offset
      + (rnum - ABI_PROPERTY_integer_func_arg_First_Register);
    break;
  case ISA_REGISTER_CLASS_integer64:
    pnum = First_Int64_Preg_Param_Offset
      + (rnum - ABI_PROPERTY_integer64_func_arg_First_Register);
    break;
  case ISA_REGISTER_CLASS_float:
    pnum = First_Float32_Preg_Param_Offset
      + (rnum - ABI_PROPERTY_float_func_arg_First_Register);
    break;
  case ISA_REGISTER_CLASS_float64:
    pnum = First_Float64_Preg_Param_Offset
      + (rnum - ABI_PROPERTY_float64_func_arg_First_Register);
    break;
  default:
    FmtAssert(FALSE, ("unexpected class"));
  }
  return pnum;
}

// widen mtype to param reg size
static TYPE_ID
Widen_Mtype (TYPE_ID m)
{
  switch (m) {
  case MTYPE_I1:
  case MTYPE_I2:
    return MTYPE_I4;
  case MTYPE_U1:
  case MTYPE_U2:
    return MTYPE_U4;
  default:
    return m;
  }
}

static void
Retval_Reg_To_Symbol_And_Offset (
  ISA_REGISTER_CLASS rc, 
  REGISTER reg,
  ST *func_st,
  WN *call_wn,
  BOOL is_formal,
  ST **param_st,
  INT *param_offset,
  TYPE_ID *param_mtype)
{
  PREG_NUM pnum = Retval_Reg_To_Preg (rc, reg);
  Is_True (func_st || call_wn, ("Either of func_st or call_wn must be valid"));
  TY_IDX func_ty = func_st ? ST_pu_type(func_st) : WN_ty(call_wn);
  STR_IDX idx = func_st ? STR_IDX_ZERO : Get_Called_Func_Name(call_wn);
  TY_IDX ty = TY_ret_type(func_ty);
  RETURN_INFO return_info = Get_Return_Info (ty, No_Simulated);
  FmtAssert(RETURN_INFO_count(return_info) > 0, ("no return info?"));
  if (is_formal) {
    // create formal param symbol for this compiler-generated name
    if (func_st)
      *param_st = Create_Formal_Param_ST (
        Get_Retval_Name(ST_name(func_st), is_formal), ty);
    else
      *param_st = Create_Formal_Param_ST (
        Get_Retval_Name(&Str_Table[idx], is_formal), ty);
  }
  else {
    // create or re-use (if multiple calls to function) the actual param symbol
    if (func_st)
      *param_st = Create_Actual_Param_ST (
        Get_Retval_Name(ST_name(func_st), is_formal), ty);
    else
      *param_st = Create_Actual_Param_ST (
        Get_Retval_Name(&Str_Table[idx], is_formal), ty);
  }
  for (INT i = 0; i < RETURN_INFO_count(return_info); i++) {
    *param_offset = RETURN_INFO_offset(return_info,i);
    *param_mtype = RETURN_INFO_mtype(return_info,i);
    if (Is_Simple_Type(ST_type(*param_st))) {
      // should match param register size
      *param_mtype = Widen_Mtype(*param_mtype);
    }
    if (RETURN_INFO_preg(return_info,i) == pnum) {
      return;
    }
  }
  FmtAssert(FALSE, ("no matching retval"));
}

static TYPE_ID
Mtype_Of_PLOC (PLOC pl)
{
  if (Preg_Offset_Is_Int32(PLOC_reg(pl))) {
    if (PLOC_size(pl) == 1) return MTYPE_U1;
    else if (PLOC_size(pl) == 2) return MTYPE_U2;
    else return MTYPE_U4;
  }
  else if (Preg_Offset_Is_Int64(PLOC_reg(pl))) {
    return MTYPE_U8;
  }
  else if (Preg_Offset_Is_Float32(PLOC_reg(pl))) {
    return MTYPE_F4;
  }
  else if (Preg_Offset_Is_Float64(PLOC_reg(pl))) {
    return MTYPE_F8;
  }
  FmtAssert(FALSE, ("unexpected ploc"));
}

static void
Param_Reg_To_Symbol_And_Offset (
  ISA_REGISTER_CLASS rc, 
  REGISTER reg,
  ST *func_st,
  WN *call_wn,
  BOOL is_formal,
  ST **param_st,
  INT *param_offset,
  TYPE_ID *param_mtype)
{
  PREG_NUM pnum = Param_Reg_To_Preg (rc, reg);
  Is_True (func_st || call_wn, ("Either of func_st or call_wn must be valid"));
  TY_IDX func_ty = func_st ? ST_pu_type(func_st) : WN_ty(call_wn);
  STR_IDX idx = func_st ? STR_IDX_ZERO : Get_Called_Func_Name(call_wn);
  // input and output parameter locations are same for us, 
  // so don't need to distinguish.
  PLOC ploc = Setup_Input_Parameter_Locations (func_ty);
  PLOC start_ploc;
  TYLIST_IDX tl = TY_parms(func_ty);
  INT count = 0;
  FmtAssert(tl != (TYLIST_IDX) NULL, ("no parameters?"));
  for (; TYLIST_ty(tl); tl = TYLIST_next(tl)) {
    TY_IDX ty = TYLIST_ty(tl);
    ploc = Get_Input_Parameter_Location (ty);
    ++count;
    if (is_formal) {
      // create formal param symbol for this compiler-generated name
      if (func_st)
        *param_st = Create_Formal_Param_ST (
          Get_Param_Name(ST_name(func_st), count, is_formal), ty);
      else
        *param_st = Create_Formal_Param_ST (
          Get_Param_Name(&Str_Table[idx], count, is_formal), ty);
    }
    else {
      // create (or reuse if multiple calls to function) st for actual parm
      if (func_st)
        *param_st = Create_Actual_Param_ST (
          Get_Param_Name(ST_name(func_st), count, is_formal), ty); 
      else
        *param_st = Create_Actual_Param_ST (
          Get_Param_Name(&Str_Table[idx], count, is_formal), ty); 
    }
    *param_offset = 0;
    start_ploc = ploc;
    ploc = First_Input_PLOC_Reg (start_ploc, ty);
    while (PLOC_is_nonempty(ploc)) {
      if (PLOC_reg(ploc) == pnum) {
        *param_offset = PLOC_offset(ploc) - PLOC_offset(start_ploc);
        *param_mtype = Mtype_Of_PLOC(ploc);
        if (Is_Simple_Type(ST_type(*param_st))) {
          // should match param register size
          *param_mtype = Widen_Mtype(*param_mtype);
        }
        return;
      }
      ploc = Next_Input_PLOC_Reg (ploc);
    }
  }
  FmtAssert(FALSE, ("no matching param"));
}

// change references to paramter registers to ld/st.param
// mov <paramreg>, reg => st.param [], reg
// mov reg, <paramreg> => ld.param reg, []
// op <paramreg>, ...  => op reg, ...; st.param [], reg
// op ..., <paramreg>  => ld.param reg, []; op ..., reg
void
Modify_Parameter_Accesses (void)
{
  BB *bb;
  OP *op;
  TN *tn;
  TN *tmp_tn;
  ISA_REGISTER_CLASS rc;
  REGISTER reg;
  INT i;
  ST *func_st;
  ST *param_st;
  WN *call_wn;
  INT param_ofst;
  TYPE_ID param_mtype;
  OPS newops = OPS_EMPTY;

  tracing = Get_Trace(TP_CGEXP, 0x2000);
  for (bb = REGION_First_BB; bb != NULL; bb = BB_next(bb)) {
    FOR_ALL_BB_OPs (bb, op) {
      for (i = 0; i < OP_results(op); ++i) {
        tn = OP_result(op,i);
        if (TN_is_register(tn) && TN_register(tn) != REGISTER_UNDEFINED) {
          rc = TN_register_class(tn);
          reg = TN_register(tn);
          param_st = NULL;
          if (REGISTER_SET_MemberP(REGISTER_CLASS_function_argument(rc),reg))
          {
            // def of param should be in call block
            FmtAssert(BB_call(bb), ("def of param not in call block"));
            func_st = CALLINFO_call_st(ANNOT_callinfo(
              ANNOT_Get(BB_annotations(bb), ANNOT_CALLINFO) ));
            call_wn = CALLINFO_call_wn(ANNOT_callinfo(
              ANNOT_Get(BB_annotations(bb), ANNOT_CALLINFO) ));
            if (tracing) {
              if (func_st)
                fprintf(TFile, "replace def of %s param:\n", ST_name(func_st));
              else
                fprintf(TFile, "replace def of %s param:\n",
                        &Str_Table[Get_Called_Func_Name(call_wn)]);
              Print_OP_No_SrcLine(op);
            }
            Param_Reg_To_Symbol_And_Offset (rc, reg, func_st, call_wn, FALSE,
              &param_st, &param_ofst, &param_mtype);
            if (tracing) {
              fprintf(TFile, "%s maps to %s+%d\n", 
                ABI_PROPERTY_Reg_Name(rc,REGISTER_machine_id(rc,reg)), 
                ST_name(param_st), param_ofst);
            }
          }
          else if (REGISTER_SET_MemberP(REGISTER_CLASS_function_value(rc),reg))
          {
            // def of retval should be in exit block
            FmtAssert(BB_exit(bb), ("def of retval not in exit block"));
            func_st = Get_Current_PU_ST();
            if (tracing) {
              fprintf(TFile, "replace def of retv in %s:\n", ST_name(func_st));
              Print_OP_No_SrcLine(op);
            }
            Retval_Reg_To_Symbol_And_Offset (rc, reg, func_st, NULL, TRUE,
              &param_st, &param_ofst, &param_mtype);
            if (tracing) {
              fprintf(TFile, "%s maps to %s+%d\n", 
                ABI_PROPERTY_Reg_Name(rc,REGISTER_machine_id(rc,reg)), 
                ST_name(param_st), param_ofst);
            }
          }
          if (param_st) { // replace op
            // op parm, r -> op tmp, r; st.param [], tmp
            tmp_tn = Build_TN_Like(tn);
            Set_OP_result(op, i, tmp_tn);
            Expand_Store (param_mtype, tmp_tn,
                Gen_Symbol_TN(param_st,0,0), Gen_Literal_TN(param_ofst,4), 
                V_PARAM_MEM, &newops);
            BB_Insert_Ops_After(bb, op, &newops);
            op = OPS_last(&newops);
            if (tracing) {
              Print_OPS_No_SrcLines(&newops);
            }
            OPS_Init(&newops); // clear newops for future use
          }
        }
      }
      for (i = 0; i < OP_opnds(op); ++i) {
        tn = OP_opnd(op,i);
        if (TN_is_register(tn) && TN_register(tn) != REGISTER_UNDEFINED) {
          rc = TN_register_class(tn);
          reg = TN_register(tn);
          param_st = NULL;
          if (REGISTER_SET_MemberP(REGISTER_CLASS_function_argument(rc),reg)) 
          {
            // use of param should be in entry block
            FmtAssert(BB_entry(bb), ("use of param not in entry block"));
            func_st = Get_Current_PU_ST();
            if (tracing) {
              fprintf(TFile, "replace use of param in %s:\n", ST_name(func_st));
              Print_OP_No_SrcLine(op);
            }
            Param_Reg_To_Symbol_And_Offset (rc, reg, func_st, NULL, TRUE,
              &param_st, &param_ofst, &param_mtype);
            if (tracing) {
              fprintf(TFile, "%s maps to %s+%d\n", 
                ABI_PROPERTY_Reg_Name(rc,REGISTER_machine_id(rc,reg)), 
                ST_name(param_st), param_ofst);
            }
          }
          else if (REGISTER_SET_MemberP(REGISTER_CLASS_function_value(rc),reg)) 
          {
            // use of retval should be in post-call block
            FmtAssert(BB_call(BB_prev(bb)), ("use of retval not in post-call block"));
            func_st = CALLINFO_call_st(ANNOT_callinfo(
              ANNOT_Get(BB_annotations(BB_prev(bb)), ANNOT_CALLINFO) ));
            call_wn = CALLINFO_call_wn(ANNOT_callinfo(
              ANNOT_Get(BB_annotations(BB_prev(bb)), ANNOT_CALLINFO) ));
            if (tracing) {
              fprintf(TFile, "replace use of %s retv:\n", ST_name(func_st));
              Print_OP_No_SrcLine(op);
            }
            Retval_Reg_To_Symbol_And_Offset (rc, reg, func_st, call_wn, FALSE,
              &param_st, &param_ofst, &param_mtype);
            if (tracing) {
              fprintf(TFile, "%s maps to %s+%d\n", 
                ABI_PROPERTY_Reg_Name(rc,REGISTER_machine_id(rc,reg)), 
                ST_name(param_st), param_ofst);
            }
          }
          if (param_st) { // replace op
            // op r, parm -> ld.param tmp, []; op r, tmp
            tmp_tn = Build_TN_Like(tn);
            Set_OP_opnd(op,i, tmp_tn);
            Expand_Load (
                OPCODE_make_op (OPR_LDID,
                  Mtype_TransferSign(param_mtype,Mtype_Of_TN(tn)),param_mtype),
                tmp_tn,
                Gen_Symbol_TN(param_st,0,0), Gen_Literal_TN(param_ofst,4), 
                V_PARAM_MEM, &newops);
            BB_Insert_Ops_Before(bb, op, &newops);
            if (tracing) {
              Print_OPS_No_SrcLines(&newops);
            }
            OPS_Init(&newops); // clear newops for future use
          }
        }
      }
    }
  }
}
