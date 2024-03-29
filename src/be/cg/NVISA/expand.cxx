/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/*

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

*/


/* ====================================================================
 * ====================================================================
 *
 * Module: expand.c
 * $Revision: 1.351 $
 * $Date: 05/07/08 10:47:17-07:00 $
 * $Author: tkong@hyalite.keyresearch $
 * $Source: /scratch/mee/Patch0002-taketwo/kpro64-pending/be/cg/x8664/SCCS/s.expand.cxx $
 *
 * Description:
 *
 * This file contains the internals of code expansion. Its interface
 * is 'Exp_OP', which takes an OP, expands it into a list of OPs which
 * are appended to the oplist passed in.
 *
 * It handles all the macro expansions, special handling during 
 * expansion and all the nitty gritty stuff that goes along with it.
 *
 * ====================================================================
 * ====================================================================
 */

#include <stdint.h>
#include "defs.h"
#include "config.h"
#include "erglob.h"
#include "ercg.h"
#include "erbe.h"
#include "glob.h"
#include "tracing.h"
#include "util.h"

#include "tn.h"
#include "cg_flags.h"
#include "bb.h"
#include "symtab.h"
#include "opcode.h"
#include "const.h"	/* needed to manipulate target/host consts */
#include "targ_const.h"	/* needed to manipulate target/host consts */
#include "op.h"
#include "data_layout.h"
#include "stblock.h"
#include "cgexp.h"
#include "cgexp_internals.h"
#include "w2op.h"
#include "label_util.h"
#include "cgtarget.h"
#include "whirl2ops.h"
#include "targ_sim.h"   /* To generate stores of param registers in builtin_apply_args */
#include "targ_const_private.h"
#include "config_opt.h" /* For Force_IEEE_Comparisons */
#ifdef KEY
#include "ebo.h"
#endif
#include "be_symtab.h"

BOOL Reuse_Temp_TNs = FALSE;

BOOL Trace_Exp2 = FALSE;      /* extra cgexp trace*/

/* Disable conversion of constant integer multiplies into shift/adds:*/
static BOOL Disable_Const_Mult_Opt = FALSE;

// rely on order of topcodes always being same for types within an opcode
INT Mtype_Index (TYPE_ID mtype)
{
  switch (mtype) {
  case MTYPE_I1: return 0;
  case MTYPE_I2: return 1;
  case MTYPE_I4: return 2;
  case MTYPE_I8: return 3;
  case MTYPE_U1: return 4;
  case MTYPE_U2: return 5;
  case MTYPE_U4: return 6;
  case MTYPE_U8: return 7;
  case MTYPE_F4: return 8;
  case MTYPE_F8: return 9;
  default: FmtAssert(FALSE, ("unexpected mtype")); return 0;
  }
}

// rely on order of topcodes always being same for types within an opcode
// Similar to above, but only generic b8,b16,etc types
INT Mtype_Byte_Index (TYPE_ID mtype)
{
  switch (mtype) {
  case MTYPE_I1: return 0;
  case MTYPE_U1: return 0;
  case MTYPE_I2: return 1;
  case MTYPE_U2: return 1;
  case MTYPE_I4: return 2;
  case MTYPE_U4: return 2;
  case MTYPE_I8: return 3;
  case MTYPE_U8: return 3;
  default: FmtAssert(FALSE, ("unexpected mtype")); return 0;
  }
}

void Expand_Start()
{
    return;

}



void Expand_Finish()
{
    return;

}

static TN* Gen_Const_Symbol_TN( INT64 int_val,
				double float_val,
				TYPE_ID mtype,
				TN_RELOCS relocs = TN_RELOC_NONE )
{
  FmtAssert( !MTYPE_is_quad(mtype), ("Quad const is not supported") );
  FmtAssert( !MTYPE_is_vector(mtype), ("Vector const is not supported") );

  const TCON tcon = MTYPE_is_integral(mtype)
    ? Host_To_Targ( mtype, int_val ) : Host_To_Targ_Float( mtype, float_val );

  ST* sym = New_Const_Sym( Enter_tcon(tcon),  Be_Type_Tbl( TCON_ty(tcon) ) );

  Allocate_Object(sym);

  return Gen_Symbol_TN( sym, 0, relocs );
}

TN *
Gen_Default_Enum_TN_ftz (void)
{
  return Gen_Enum_TN (Target_ISA >= TARGET_ISA_compute_20 && CG_use_ftz
                      ? ECV_ftz_ftz
                      : ECV_ftz_none);
}

TN *
Gen_Default_Enum_TN_prec_div (void)
{
  return Gen_Enum_TN (Target_ISA >= TARGET_ISA_compute_20 && CG_use_prec_div
                      ? ECV_fround_rn
                      : ECV_fround_full);
}

TN *
Gen_Default_Enum_TN_prec_rcp (void)
{
  return Gen_Enum_TN (Target_ISA >= TARGET_ISA_compute_20 && CG_use_prec_div
                      ? ECV_fround_rn
                      : ECV_fround_approx);
}

TN *
Gen_Default_Enum_TN_prec_sqrt (void)
{
  return Gen_Enum_TN (Target_ISA >= TARGET_ISA_compute_20 && CG_use_prec_sqrt
                      ? ECV_fround_rn
                      : ECV_fround_approx);
}

void
Expand_Copy (TN *result, TN *src, TYPE_ID mtype, OPS *ops)
{
  if (mtype == MTYPE_I1 || mtype == MTYPE_U1) {
    // there are no byte registers, and normally should not see a request
    // for a byte copy, but if do ask for byte copy instead do short copy.
    mtype = Mtype_TransferSize (MTYPE_I2, mtype);
  }
  TOP opc = (TOP) (TOP_mov_s8 + Mtype_Index(mtype));
  Build_OP (opc, result, src, ops);
  Set_OP_copy (OPS_last(ops));
  if (TN_has_memory_space(src))
	Set_TN_memory_space (result, TN_memory_space(src));
  if ((TN_in_texture_mem(src) || TN_in_surface_mem(src)) 
    && TN_home(src) != NULL) 
  {
	if (Trace_Exp2) fprintf(TFile,"replace texture tn home copy\n");
        // set the home of the result
        // which will later be used by the texture asm.
        Set_TN_home (result, TN_home(src));
  }
}

void
Expand_Int_To_Int (TN *result, TYPE_ID rtype, TN *src, TYPE_ID stype, OPS *ops)
{
  // cvt are ordered by rtype and they by stype,
  // but skips when rtype == stype.
  TOP top;
  FmtAssert(MTYPE_byte_size(rtype) <= TN_size(result),
	("cvt rtype won't fit in register"));
  if (MTYPE_size_reg(stype) < MTYPE_size_reg(Mtype_Of_TN(src))
    && MTYPE_size_reg(rtype) == MTYPE_size_reg(Mtype_Of_TN(result))) 
  {
        // e.g. cvt.u64.u16 where src is in 32bit reg;
        // need to first convert source to smaller reg
        DevWarn("stype < tn size");
        TN *tmp = Build_TN_Of_Mtype (stype);
        Expand_Int_To_Int (tmp, stype, src, Mtype_Of_TN(src), ops);
        src = tmp;
  }
  if ((MTYPE_size_reg(Mtype_Of_TN(result)) > MTYPE_size_reg(Mtype_Of_TN(src)))
    && (MTYPE_size_reg(rtype) < MTYPE_size_reg(Mtype_Of_TN(result))))
  {
	// e.g. s8.s16, result is 32bit, src is 16bit
	// should become s8.s16 result in 16bit, then s32.s8 result.
	// This is messy but rare; first need to convert in same reg size,
	// then do cvt to larger reg size 
	// (should later get optimized into single convert).
	DevWarn("convert of mismatched reg sizes requires extra cvt");
	TN *tmp = Build_TN_Like(src);
	Expand_Int_To_Int (tmp, rtype, src, stype, ops);
	src = tmp;
	rtype = Mtype_Of_TN(result);
  }
  if (MTYPE_size_reg(Mtype_Of_TN(result)) == 32
	&& MTYPE_size_reg(rtype) < 32)
  {
	// may have cvt to smaller size but done in larger reg,
	// in which case we use different cvt (same eventual inst,
	// but takes different reg sizes).
	// just have s8/u8/s16/u16 rtypes for this, so not mtype_index.
	INT ridx;
	switch (rtype) {
	case MTYPE_I1: ridx = 0; break;
	case MTYPE_I2: ridx = 1; break;
	case MTYPE_U1: ridx = 2; break;
	case MTYPE_U2: ridx = 3; break;
	default: FmtAssert(FALSE, ("unexpected mtype"));
	}
	top = (TOP) (TOP_cvt_sat_s8_s16_b32 + 5*ridx + Mtype_Index(stype));
	if (MTYPE_is_unsigned(stype)) {
	  // doesn't make sense to have s64 in b32, 
	  // but that means mtype_index is non-standard.
	  top = (TOP) (top - 1); // skipped s64
	}
  }
  else if (MTYPE_size_reg(Mtype_Of_TN(result)) == 64
	&& MTYPE_size_reg(rtype) < 64)
  {
	// same as 32bit case, doing cvt to smaller size but in larger reg.
	INT ridx;
	switch (rtype) {
	case MTYPE_I1: ridx = 0; break;
	case MTYPE_I2: ridx = 1; break;
	case MTYPE_I4: ridx = 2; break;
	case MTYPE_U1: ridx = 3; break;
	case MTYPE_U2: ridx = 4; break;
	case MTYPE_U4: ridx = 5; break;
	default: FmtAssert(FALSE, ("unexpected mtype"));
	}
	top = (TOP) (TOP_cvt_sat_s8_s16_b64 + 7*ridx + Mtype_Index(stype));
  }
  else {
  	top = (TOP) (TOP_cvt_sat_s8_s16 + 9*Mtype_Index(rtype) + Mtype_Index(stype));
  }
  if (Mtype_Index(stype) > Mtype_Index(rtype)) {
	// skipped identity convert
	top = (TOP) (top - 1);
  }
  // Account for <sat> field as needed
  if (MTYPE_size_reg(rtype) <= MTYPE_size_reg(stype)
      || (MTYPE_is_unsigned(rtype) && MTYPE_is_signed(stype))) {
    Build_OP (top, result, Gen_Enum_TN(ECV_sat_none), src, ops);
  } else {
    Build_OP (top, result, src, ops);
  }  
  if (TN_is_boolean(src))
    Set_TN_is_boolean(result);
}

void
Expand_Convert (TN *result, TYPE_ID rtype, TN *src, TYPE_ID stype, OPS *ops)
{
  TOP top;
  if (rtype == stype) {
	Expand_Copy (result, src, rtype, ops);
  }
  else if (MTYPE_is_integral(stype) && MTYPE_is_integral(rtype)) {
	Expand_Int_To_Int (result, rtype, src, stype, ops);
  }
  else if (MTYPE_is_integral(stype) && MTYPE_is_float(rtype)) {
	Expand_Int_To_Float (result, src, stype, rtype, ops);
  } 
  else if (MTYPE_is_float(stype) && MTYPE_is_integral(rtype)) {
	Expand_Float_To_Int_Cvt (result, src, rtype, stype, ops);
  }
  else if (MTYPE_is_float(stype) && MTYPE_is_float(rtype)) {
	Expand_Float_To_Float (result, src, rtype, stype, ops);
  }
}

/* ====================================================================
 *
 * Expand_Convert_Length
 *
 * ====================================================================
 */
void Expand_Convert_Length ( TN *dest, TN *src, TN *length_tn, TYPE_ID mtype,
			     BOOL signed_extension, OPS *ops )
{
  FmtAssert (! MTYPE_float(mtype),
	     ("Expand_Convert_Length: illegal data type\n"));
  FmtAssert (TN_has_value(length_tn),
	     ("Expand_Convert_Length: non-constant length\n"));

  TYPE_ID length_mtype;
  UINT64 length = TN_value (length_tn);
  switch (length) {
  case 8:
	length_mtype = (signed_extension ? MTYPE_I1 : MTYPE_U1);
	break;
  case 16:
	length_mtype = (signed_extension ? MTYPE_I2 : MTYPE_U2);
	break;
  case 32:
	length_mtype = (signed_extension ? MTYPE_I4 : MTYPE_U4);
	break;
  case 64:
	length_mtype = (signed_extension ? MTYPE_I8 : MTYPE_U8);
	break;
  default:
	// bit conversion
        Exp_Extract_Bits (mtype, mtype, 0, length, dest, src, ops);
        return;
  }
  Expand_Convert (dest, length_mtype, src, mtype, ops);
}

void
Exp_Immediate (TN *dest, TN *src, BOOL is_signed, OPS *ops)
{
  Expand_Mtype_Immediate(dest, src, Mtype_Of_TN(dest), ops);
}

/* 
 * Expand Immediate value.
 */
void
Expand_Immediate (TN *dest, TN *src, BOOL is_signed, OPS *ops)
{
  FmtAssert((TN_is_constant(src)),
	    ("unexpected non-constant in Expand_Immediate"));
  FmtAssert((TN_has_value(src) || TN_is_symbol(src)), 
	    ("expected value or const in Expand_Immediate"));
  FmtAssert(FALSE,("NYI"));
}

void
Expand_Mtype_Immediate (TN *dest, TN *src, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  FmtAssert((TN_is_constant(src)),
	    ("unexpected non-constant in Expand_Immediate"));
  FmtAssert((TN_has_value(src) || TN_is_symbol(src)), 
	    ("expected value or const in Expand_Immediate"));

  opc = (TOP) (TOP_mov_s8_lit + Mtype_Index(mtype));
  Build_OP (opc, dest, src, ops);
}

TN*
Expand_Mtype_Immediate_Into_Register (TN *src, TYPE_ID mtype, OPS *ops)
{
  /* load into reg and do reg case */
  TN *tmp = Build_TN_Of_Mtype (mtype);
  Expand_Mtype_Immediate (tmp, src, mtype, ops);
  return tmp;
}

TN*
Expand_Immediate_Into_Register (TN *src, BOOL is_64bit, OPS *ops)
{
  return Expand_Mtype_Immediate_Into_Register (src, 
	(is_64bit ? MTYPE_I8 : MTYPE_I4), ops);
}

void
Expand_Add (TN *result, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_add_s8 + Mtype_Index(mtype));
  INT64 val;
  FmtAssert(!TN_is_constant(src1), ("NYI"));

  // may be incrementing through memory space:
  // (propagate src2 before we replace it)
  if (TN_has_memory_space(src1))
	Set_TN_memory_space (result, TN_memory_space(src1));
  else if (TN_has_memory_space(src2))
	Set_TN_memory_space (result, TN_memory_space(src2));

  if (TN_Can_Use_Constant_Value (src1, mtype, &val)) {
  	opc = (TOP) (TOP_add_s8_lit + Mtype_Index(mtype));
	src1 = src2;
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  } else if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
	if (val < 0 && ISA_LC_Value_In_Class (-val, Lit_Class_For_Mtype(mtype)))
	{
		// change add -N to sub N
		// helps 16bit optimization as well as being clearer
		src2 = Gen_Literal_TN_Of_Mtype (-val, mtype);
		Expand_Sub (result, src1, src2, mtype, ops);
		return;
	}
	else {
  		opc = (TOP) (TOP_add_s8_lit + Mtype_Index(mtype));
		src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
	}
  }
  else if (TN_is_constant(src2)) {
        // expand add a,const into mov t,const; add a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  if (mtype == MTYPE_F4) {
    Build_OP (opc, result, Gen_Enum_TN (ECV_fround_none),
              Gen_Default_Enum_TN_ftz (),
              Gen_Enum_TN (ECV_sat_none), src1, src2, ops);
  } else {
    Build_OP (opc, result, src1, src2, ops);
  }
}

void
Expand_Sub (TN *result, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_sub_s8 + Mtype_Index(mtype));
  INT64 val;
  // may be incrementing through memory space:
  // (propagate src2 before we replace it)
  if (TN_has_memory_space(src1))
	Set_TN_memory_space (result, TN_memory_space(src1));
  else if (TN_has_memory_space(src2))
	Set_TN_memory_space (result, TN_memory_space(src2));

  if (TN_is_constant(src1)) {
	FmtAssert(TN_has_value(src1), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src1 = Expand_Mtype_Immediate_Into_Register (src1, mtype, ops);
  }
  if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
  	opc = (TOP) (TOP_sub_s8_lit + Mtype_Index(mtype));
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_is_constant(src2)) {
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  if (mtype == MTYPE_F4) {
    Build_OP (opc, result, Gen_Enum_TN (ECV_fround_none),
              Gen_Default_Enum_TN_ftz (),
              Gen_Enum_TN (ECV_sat_none), src1, src2, ops);
  } else {
    Build_OP (opc, result, src1, src2, ops);
  }
}


void
Expand_Neg (TN *result, TN *src, TYPE_ID mtype, OPS *ops)
{
  if (MTYPE_is_unsigned(mtype)) {
  	// ptx doesn't allow neg of unsigned integers, but whirl and C do,
  	// so force those to do signed negate.
	mtype = Mtype_TransferSign(MTYPE_I4, mtype);
  }
  if (TN_is_constant(src)) {
	FmtAssert(TN_has_value(src), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src = Expand_Mtype_Immediate_Into_Register (src, mtype, ops);
  }
  TOP opc = (TOP) (TOP_neg_s8 + Mtype_Index(mtype));
  if (mtype == MTYPE_F4) {
    Build_OP (opc, result, Gen_Default_Enum_TN_ftz (), src, ops);
  } else {
    Build_OP (opc, result, src, ops);
  }
}


void
Expand_Abs (TN *dest, TN *src, TYPE_ID mtype, OPS *ops)
{
  if (MTYPE_is_unsigned(mtype)) {
  	// ptx doesn't allow abs of unsigned integers, but whirl and C do,
  	// so force those to do signed negate.
	mtype = Mtype_TransferSign(MTYPE_I4, mtype);
  }
  if (TN_is_constant(src)) {
	FmtAssert(TN_has_value(src), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src = Expand_Mtype_Immediate_Into_Register (src, mtype, ops);
  }
  TOP opc = (TOP) (TOP_abs_s8 + Mtype_Index(mtype));
  if (mtype == MTYPE_F4) {
    Build_OP (opc, dest, Gen_Default_Enum_TN_ftz (), src, ops);
  } else {
    Build_OP (opc, dest, src, ops);
  }
}

void
Expand_Shift (TN *result, TN *src1, TN *src2, TYPE_ID mtype, SHIFT_DIRECTION kind, OPS *ops)
{
  TOP opc;
  INT64 val;
  switch (kind) {
  case shift_left:
    opc = TOP_shl_b8;
    break;
  case shift_aright:
    // will aright always be signed?
    // No, cause rtype is result (may be implicit convert), not desc.
    // e.g. asm_input may put rtype on shift (see opt_emit_template.h)
    if (MTYPE_is_unsigned(mtype))
      mtype = MTYPE_complement(mtype);
    opc = TOP_shr_s8;
    break;
  case shift_lright:
    // will lright always be unsigned?
    // No, cause rtype is result (may be implicit convert), not desc.
    // e.g. asm_input may put rtype on shift (see opt_emit_template.h)
    // Could avoid this situation in opt_emit, but other targets
    // ignore mtype so cleaner to put this in target-specific code.
    if (MTYPE_is_signed(mtype))
      mtype = MTYPE_complement(mtype);
    opc = TOP_shr_s8;
    break;
  }
  if (TN_Can_Use_Constant_Value (src1, mtype, &val)) {
	opc = (opc == TOP_shl_b8) ? TOP_shl_b8_lit1 : TOP_shr_s8_lit1;
	src1 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_is_constant(src1)) {
        // expand op a,const into mov t,const; op a,t;
        src1 = Expand_Mtype_Immediate_Into_Register (src1, mtype, ops);
  }
  if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
	if (val == 0) {
		// shift by zero is simply a move
    		Expand_Copy (result, src1, mtype, ops);
		return;
	}
	opc = (opc == TOP_shl_b8) ? TOP_shl_b8_lit : TOP_shr_s8_lit;
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_is_constant(src2)) {
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  if (kind == shift_left)
    opc = (TOP) (opc + Mtype_Byte_Index(mtype));
  else
    opc = (TOP) (opc + Mtype_Index(mtype));
  Build_OP (opc, result, src1, src2, ops);
}

/*
 *  Try to expand a multiply into a sequence of less expensive operations.
 */
static BOOL
Expand_Constant_Multiply (TN *result, TN *var_tn, TARG_INT constant, TYPE_ID mtype, OPS *ops)
{

  BOOL did_do_fast;
  INT16 limit;	/* maximum number of operations to replace the multiply */
  TN *x = var_tn;
  INT64 c = constant; // I don't want to depend on TARG_INT
  BOOL needs_sign_extension;

  // fast special cases
  if (c == 0) {
    Expand_Copy (result, Zero_TN, MTYPE_I8, ops);
    return TRUE;
  } else if (c == 1) {
    Expand_Copy (result, var_tn, MTYPE_I8, ops);
    return TRUE;
  } else if (c == -1) {
    Expand_Neg(result, var_tn, mtype, ops);
    return TRUE;
  }
    
  FmtAssert(FALSE, ("NYI"));
  return TRUE;
}

// is cvt.64.32
// also return mtype of original value
static BOOL
OP_is_cvt_32to64 (OP *op, TYPE_ID *mtype)
{
  switch (OP_code(op)) {
  case TOP_cvt_s64_s32:
  case TOP_cvt_sat_u64_s32:
    *mtype = MTYPE_I8; // original type was signed
    return TRUE;
  case TOP_cvt_s64_u32:
  case TOP_cvt_u64_u32:
    *mtype = MTYPE_U8; // original type was unsigned
    return TRUE;
  }
  return FALSE;
}

// Search through ops for 32bit cvt def of src;
// if not found return NULL.
// Note that this only looks in ops list, so basically in current bb.
// So may miss some cases where def is in another bb; if that case becomes
// important we could have a later ebo pass that looks across bbs, 
// but that would be more expensive, and not sure yet whether is worth it.
// Also return mtype of original value.
static TN*
Find_32bit_Def (TN *src, TYPE_ID *mtype, OPS *ops)
{
  if (TN_is_register(src)) {
    OP *op;
    FOR_ALL_OPS_OPs_REV(ops,op) {
      // look for def that is cvt
      if (OP_is_cvt_32to64(op, mtype) && src == OP_result(op,0)) {
        // make sure we don't redefine 32bit tn (unlikely but be safe)
        TN *tn32 = OP_opnd(op, OP_find_opnd_use(op, OU_opnd1));
        OP *op2 = OP_next(op);
        while (op2 != NULL) {
          if (tn32 == OP_result(op2,0)) {
            DevWarn("redefined cvt before mul");
            return NULL; // new def of 32bit tn
          }
          op2 = OP_next(op2);
        }
        return tn32;
      }
    }
    return NULL; // didn't find 32bit def
  }
  else if (TN_has_value(src)
    && ISA_LC_Value_In_Class(TN_value(src), 
       (MTYPE_signed(*mtype) ? LC_i32 : LC_u32) ) )
  {
    return src; // is 32bit literal
  }
  else {
    return NULL;
  }
}

void
Expand_Multiply (TN *result, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  FmtAssert(!TN_is_constant(src1), ("NYI"));

  if (MTYPE_is_integral(mtype)) {
	INT64 val;
  	// for int multiply, do mul_lo 
	opc = (TOP) (TOP_mul_lo_s8 + Mtype_Index(mtype));
	if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
	    	// Should we convert to shifts, or let OCG do that?
	    	// For now, we'll defer to OCG.
		opc = (TOP) (TOP_mul_lo_s8_lit + Mtype_Index(mtype));
		src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
	} 
	else if (TN_is_constant(src2)) {
        	// expand op a,const into mov t,const; op a,t;
        	src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
	}
        if (MTYPE_is_size_double(mtype)) {
          TYPE_ID src_mtype = mtype;
          // see if can change mul.lo.64 to mul.wide.32.
          TN *src1_32 = Find_32bit_Def (src1, &src_mtype, ops);
          TN *src2_32 = Find_32bit_Def (src2, &src_mtype, ops);
          if (src1_32 && src2_32) {
            if (TN_is_dedicated(src1_32)) {
              TN *tmp = Build_TN_Like(src1_32);
              Exp_COPY (tmp, src1_32, ops);
              src1_32 = tmp;
            }
            if (TN_is_dedicated(src2_32)) {
              TN *tmp = Build_TN_Like(src2_32);
              Exp_COPY (tmp, src2_32, ops);
              src2_32 = tmp;
            }
            // change to mul.wide
            switch (opc) {
            case TOP_mul_lo_s64:
            case TOP_mul_lo_u64:
              // use mtype from original value in case had 
              // cvt.s64.s32; mul.lo.u64, in which case need mul.wide.s32
              opc = (MTYPE_signed(src_mtype) ? TOP_mul_wide_s32 
                                         : TOP_mul_wide_u32);
              break;
            case TOP_mul_lo_s64_lit:
            case TOP_mul_lo_u64_lit:
              opc = (MTYPE_signed(src_mtype) ? TOP_mul_wide_s32_lit 
                                         : TOP_mul_wide_u32_lit);
              break;
            }
            if (Trace_Exp2) {
              fprintf(TFile, "change op to %s\n", TOP_Name(opc));
            }
            src1 = src1_32;
            src2 = src2_32;
          }
        }
  }
  else { // float
	// for float, do regular mul.
        opc = (mtype == MTYPE_F4) ? TOP_mul_fround_ftz_sat_f32 : TOP_mul_f64;
  	if (TN_has_value(src2)) {
        	// expand op a,const into mov t,const; op a,t;
        	src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
	}
  }
  if (mtype == MTYPE_F4) {
    Build_OP (opc, result, Gen_Enum_TN (ECV_fround_none),
              Gen_Default_Enum_TN_ftz (),
              Gen_Enum_TN (ECV_sat_none), src1, src2, ops);
  } else {
    Build_OP (opc, result, src1, src2, ops);
  }  
}

/* return high part of multiply result */
void
Expand_High_Multiply (TN *result, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_mul_hi_s8 + Mtype_Index(mtype));
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
	// Should we convert to shifts, or let OCG do that?
	// For now, we'll defer to OCG.
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  Build_OP (opc, result, src1, src2, ops);
}

static TN*
Normalize_Operand (TN *src, TYPE_ID mtype, OPS *ops)
{
    // test not equal to 0
    DevWarn("insert normalize of operand");
    TN *tmp = Build_TN_Like(src);
    Expand_Not_Equal (tmp, src, Gen_Literal_TN(0,4), V_NONE, mtype, mtype, ops);
    return tmp;
}

void Expand_Logical_Not (TN *dest, TN *src, VARIANT variant, OPS *ops)
{
  /* dest = (src == 0) ? 1 : 0 */
  TYPE_ID src_mtype = Mtype_Of_TN(src);
  TYPE_ID dest_mtype = Mtype_Of_TN(dest);
  if ( ! V_normalized_op1(variant)) {
    src = Normalize_Operand (src, src_mtype, ops);
  }
  if (dest_mtype == MTYPE_B && src_mtype == MTYPE_B) {
    // not the predicate
    Build_OP (TOP_not_pred, dest, src, ops);
  }
  else if (dest_mtype == MTYPE_B && src_mtype != MTYPE_B) {
    // dest = (src == 0 ? 1 : 0)
    TOP opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(src_mtype));
    TN *zero_tn = Expand_Mtype_Immediate_Into_Register (
	Gen_Literal_TN(0,4), MTYPE_I4, ops);
    FmtAssert(src_mtype != MTYPE_F4 && src_mtype != MTYPE_F8, ("Operator '!' applied to floating-point"));
    Build_OP (opc, dest, Gen_Enum_TN(ECV_cmp_eq), src, zero_tn, ops);
  }
  else if (dest_mtype != MTYPE_B && src_mtype == MTYPE_B) {
	// I don't think this can happen
	FmtAssert(FALSE, ("unexpected LNOT"));
  }
  else {
    // use xor rather than not so result is 1 not -1.
    TOP opc = (TOP) (TOP_xor_b8_lit + Mtype_Byte_Index(src_mtype));
    Build_OP (opc, dest, src, Gen_Literal_TN(1,4), ops);
  }
}

void Expand_Logical_And (TN *dest, TN *src1, TN *src2, VARIANT variant, OPS *ops)
{
  TYPE_ID dest_mtype = Mtype_Of_TN(dest);
  TYPE_ID src_mtype = Mtype_Of_TN(src1);
  if ( ! V_normalized_op1(variant)) {
    src1 = Normalize_Operand (src1, src_mtype, ops);
  }
  if ( ! V_normalized_op2(variant)) {
    src2 = Normalize_Operand (src2, src_mtype, ops);
  }
  if (src_mtype == MTYPE_B) {
    // and the predicate
    // assume that if src is pred, dest will be pred
    FmtAssert(dest_mtype == MTYPE_B, ("unexpected LAND"));
    Build_OP (TOP_and_pred, dest, src1, src2, ops);
  }
  else {
    // source not a pred
    TOP opc = (TOP) (TOP_and_b8 + Mtype_Byte_Index(src_mtype));
    if (dest_mtype == MTYPE_B) {
      // will do and of src type, then change to pred
      TN *tmp = Build_TN_Like(src1);
      Build_OP (opc, tmp, src1, src2, ops);
      // convert to predicate
      opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(src_mtype));
      TN *zero_tn = Expand_Mtype_Immediate_Into_Register (
	Gen_Literal_TN(0,4), MTYPE_I4, ops);
      FmtAssert(src_mtype != MTYPE_F4 && src_mtype != MTYPE_F8, ("Operator '&&' applied to floating-point"));
      Build_OP (opc, dest, Gen_Enum_TN(ECV_cmp_ne), tmp, zero_tn, ops); 
    }
    else {
      Build_OP (opc, dest, src1, src2, ops);
    }
  }
}

void Expand_Logical_Or (TN *dest, TN *src1, TN *src2, VARIANT variant, OPS *ops)
{
  TYPE_ID dest_mtype = Mtype_Of_TN(dest);
  TYPE_ID src_mtype = Mtype_Of_TN(src1);
  if ( ! V_normalized_op1(variant)) {
    src1 = Normalize_Operand (src1, src_mtype, ops);
  }
  if ( ! V_normalized_op2(variant)) {
    src2 = Normalize_Operand (src2, src_mtype, ops);
  }
  if (src_mtype == MTYPE_B) {
    // or the predicate
    // assume that if src is pred, dest will be pred
    FmtAssert(dest_mtype == MTYPE_B, ("unexpected LOR"));
    Build_OP (TOP_or_pred, dest, src1, src2, ops);
  }
  else {
    // source not a pred
    TOP opc = (TOP) (TOP_or_b8 + Mtype_Byte_Index(src_mtype));
    if (dest_mtype == MTYPE_B) {
      // will do and of src type, then change to pred
      TN *tmp = Build_TN_Like(src1);
      Build_OP (opc, tmp, src1, src2, ops);
      // convert to predicate
      opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(src_mtype));
      TN *zero_tn = Expand_Mtype_Immediate_Into_Register (
	Gen_Literal_TN(0,4), MTYPE_I4, ops);
      FmtAssert(src_mtype != MTYPE_F4 && src_mtype != MTYPE_F8, ("Operator '||' applied to floating-point"));
      Build_OP (opc, dest, Gen_Enum_TN(ECV_cmp_ne), tmp, zero_tn, ops); 
    }
    else {
      Build_OP (opc, dest, src1, src2, ops);
    }
  }
}


void Expand_Binary_Complement (TN *dest, TN *src, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_not_b8 + Mtype_Byte_Index(mtype));
  if (TN_is_constant(src)) {
	FmtAssert(TN_has_value(src), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src = Expand_Mtype_Immediate_Into_Register (src, mtype, ops);
  }
  else if (TN_is_boolean(src)) {
	Set_TN_is_boolean(dest);
  }
  Build_OP (opc, dest, src, ops);
}

void Expand_Binary_And (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_and_b8 + Mtype_Byte_Index(mtype));
  INT64 val;

  if (TN_Can_Use_Constant_Value (src1, mtype, &val)) {
  	opc = (TOP) (TOP_and_b8_lit + Mtype_Byte_Index(mtype));
	src1 = src2;
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  } else if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
  	opc = (TOP) (TOP_and_b8_lit + Mtype_Byte_Index(mtype));
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
	// if & 1 then result is 0 or 1, so is normalized boolean
	if (val == 1) {
		Set_TN_is_boolean(dest);
	}
  }
  else if (TN_is_constant(src1)) {
        // expand add a,const into mov t,const; add a,t;
        src1 = Expand_Mtype_Immediate_Into_Register (src1, mtype, ops);
  }
  else if (TN_is_constant(src2)) {
        // expand add a,const into mov t,const; add a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  else if (TN_is_boolean(src1) && TN_is_boolean(src2)) {
	Set_TN_is_boolean(dest);
  }
  Build_OP (opc, dest, src1, src2, ops);
}

void Expand_Binary_Or (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_or_b8 + Mtype_Byte_Index(mtype));
  INT64 val;

  if (TN_Can_Use_Constant_Value (src1, mtype, &val)) {
  	opc = (TOP) (TOP_or_b8_lit + Mtype_Byte_Index(mtype));
	src1 = src2;
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  } else if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
  	opc = (TOP) (TOP_or_b8_lit + Mtype_Byte_Index(mtype));
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_is_constant(src1)) {
        // expand add a,const into mov t,const; add a,t;
        src1 = Expand_Mtype_Immediate_Into_Register (src1, mtype, ops);
  }
  else if (TN_is_constant(src2)) {
        // expand add a,const into mov t,const; add a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  else if (TN_is_boolean(src1) && TN_is_boolean(src2)) {
	Set_TN_is_boolean(dest);
  }
  Build_OP (opc, dest, src1, src2, ops);
}

void Expand_Binary_Xor (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_xor_b8 + Mtype_Byte_Index(mtype));
  INT64 val;
  if (TN_Can_Use_Constant_Value (src1, mtype, &val)) {
  	opc = (TOP) (TOP_xor_b8_lit + Mtype_Byte_Index(mtype));
	src1 = src2;
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_Can_Use_Constant_Value (src2, mtype, &val)) {
  	opc = (TOP) (TOP_xor_b8_lit + Mtype_Byte_Index(mtype));
	src2 = Gen_Literal_TN_Of_Mtype (val, mtype);
  }
  else if (TN_is_constant(src1)) {
        // expand op a,const into mov t,const; op a,t;
        src1 = Expand_Mtype_Immediate_Into_Register (src1, mtype, ops);
  }
  else if (TN_is_constant(src2)) {
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  else if (TN_is_boolean(src1) && TN_is_boolean(src2)) {
	Set_TN_is_boolean(dest);
  }
  Build_OP (opc, dest, src1, src2, ops);
}

void Expand_Binary_Nor (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TN *tmp = Build_TN_Like(dest);
  Expand_Binary_Or (tmp, src1, src2, mtype, ops);
  Expand_Binary_Complement (dest, tmp, mtype, ops);
}


void Expand_Less (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_lt),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_lt),
              src1, src2, ops);
  }
  if (dest != tmp) {            
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }    
}

void Expand_Less_Equal (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_le),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_le),
              src1, src2, ops);
  }
  if (dest != tmp) {            
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }
}

void Expand_Equal (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    if (mtype == MTYPE_B) {
      // d = (p == q) is same as !xor 
      tmp = Build_TN_Like(dest);
      Build_OP (TOP_xor_pred, tmp, src1, src2, ops);
      Build_OP (TOP_not_pred, dest, tmp, ops);
      return;
    }
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_eq),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_eq),
              src1, src2, ops);
  }
  if (dest != tmp) {            
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }
}

void Expand_Not_Equal (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    if ( mtype == MTYPE_B) {
      // d = (p != q) is same as xor
      Build_OP (TOP_xor_pred, dest, src1, src2, ops);
      return;
    }  
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  // float uses neu not ne so nan's work properly
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_neu),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(rtype == MTYPE_F8? ECV_cmp_neu: ECV_cmp_ne),
              src1, src2, ops);
  }
  if (dest != tmp) {            
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }    
}

void Expand_Greater_Equal (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_ge),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_ge), src1, src2, ops);
  }  
  if (dest != tmp) {
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }
}

void Expand_Greater (TN *dest, TN *src1, TN *src2, VARIANT v, TYPE_ID rtype, TYPE_ID mtype, OPS *ops)
{
  TOP opc;
  TN *tmp = dest;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }

  if (rtype == MTYPE_B) {
    // set predicate rather than int reg
    opc = (TOP) (TOP_setp_cmp_s8 + Mtype_Index(mtype));
  }
  else {
    opc = (TOP) (TOP_set_cmp_u32_s8 + Mtype_Index(mtype));
    if ( ! V_normalized_op1(v)) {
      // set returns 0 or -1, we need 0 or 1, so negate result
      tmp = Build_TN_Like(dest);
    }
  }    
  if (mtype == MTYPE_F4) {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_gt),
              Gen_Default_Enum_TN_ftz (),
              src1, src2, ops);
  } else {
    Build_OP (opc, tmp, Gen_Enum_TN(ECV_cmp_gt), src1, src2, ops);
  }  
  if (dest != tmp) {
    Expand_Neg (dest, tmp, MTYPE_I4, ops);
    Set_TN_is_boolean(dest);
  }
}

void
Expand_Bool_To_Int (TN *dest, TN *src, VARIANT v, TYPE_ID rtype, OPS *ops)
{
  // something like I4BCVT
#if 0
  // could change to set.eq.or when that works.
  // problem is that set returns -1 or 0, not 1 or 0.
  TOP opc = TOP_set_cmpb_u32_u32_lit;
  TN *zero_tn = Gen_Literal_TN_Of_Mtype(0, rtype);
  Build_OP (opc, dest, Gen_Enum_TN(ECV_cmp_eq), Gen_Enum_TN(ECV_boolop_or), 
	zero_tn, zero_tn, src, ops);
#else
  // use selp to set dest based on predicate.
  TOP opc = (TOP) (TOP_selp_s8_lit + Mtype_Index(rtype));
  TN *true_tn = Gen_Literal_TN_Of_Mtype(1,rtype);
  TN *false_tn = Gen_Literal_TN_Of_Mtype(0,rtype);
  if (v == V_BR_FALSE) {
	// negate when converting
  	Build_OP (opc, dest, false_tn, true_tn, src, ops);
  } else {
  	Build_OP (opc, dest, true_tn, false_tn, src, ops);
  }
  // mark that it is still boolean
  Set_TN_is_boolean(dest);
#endif
}

void
Expand_Float_To_Int_Cvt (TN *dest, TN *src, TYPE_ID imtype, TYPE_ID fmtype, OPS *ops)
{
  // what rounding mode to use if none specified?
  Expand_Trunc (dest, src, imtype, fmtype, ops);
}

void
Expand_Float_To_Int_Tas (TN *dest, TN *src, TYPE_ID imtype, OPS *ops)
{
  Is_True( Is_Target_32bit(), ("Expand_Float_To_Int_Tas should not be invoked under -m64") );
  // Allocate space to store the floating point value
  const TY_IDX ty = MTYPE_To_TY( imtype );
  ST* st = Gen_Temp_Symbol( ty, "float_2_int" );
  Allocate_Temp_To_Memory( st );

  ST* base_sym = NULL;
  INT64 base_ofst = 0;

  Base_Symbol_And_Offset_For_Addressing( st, 0, &base_sym, &base_ofst );
  FmtAssert( base_sym == SP_Sym || base_sym == FP_Sym,
	     ("Expand_Float_To_Int_Tas: base symbol is on stack") );

  TN* base_tn = base_sym == SP_Sym ? SP_TN : FP_TN;
  TN* ofst_tn = Gen_Literal_TN( base_ofst, 4 );

  FmtAssert(FALSE, ("NYI"));
} 

void
Expand_Int_To_Float_Tas (TN *dest, TN *src, TYPE_ID fmtype, OPS *ops)
{
  Is_True( Is_Target_32bit(), ("Expand_Int_To_Float_Tas should not be invoked under -m64") );
  // Allocate space to store the integer point value
  const TY_IDX ty = MTYPE_To_TY( fmtype );
  ST* st = Gen_Temp_Symbol( ty, "int_2_float" );
  Allocate_Temp_To_Memory( st );

  ST* base_sym = NULL;
  INT64 base_ofst = 0;

  Base_Symbol_And_Offset_For_Addressing( st, 0, &base_sym, &base_ofst );
  FmtAssert( base_sym == SP_Sym || base_sym == FP_Sym,
	     ("Expand_Float_To_Int_Tas: base symbol is on stack") );

  TN* base_tn = base_sym == SP_Sym ? SP_TN : FP_TN;
  TN* ofst_tn = Gen_Literal_TN( base_ofst, 4 );

  FmtAssert(FALSE, ("NYI"));
} 

void 
Expand_Floor (TN *dest, TN *src, TYPE_ID rtype, TYPE_ID desc, OPS *ops)
{
  TOP top;
  if (MTYPE_is_float(rtype)) {
    FmtAssert(rtype == desc, ("differing mtypes"));
    if (rtype == MTYPE_F4) 
  	top = TOP_cvt_fround_ftz_sat_f32_f32;
    else if (rtype == MTYPE_F8)
  	top = TOP_cvt_fround_sat_f64_f64;
    else
  	FmtAssert(FALSE, ("unexpected mtype"));
  } 
  else { // float to int
    // ops are ordered s8_f32, s8_f64, s16_f32, etc.
    top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32 
	+ 9*Mtype_Index(rtype) + (Mtype_Index(desc) - Mtype_Index(MTYPE_F4)));
  }
  if (desc == MTYPE_F4) {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rmi),
              Gen_Default_Enum_TN_ftz(),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  } else {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rmi),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  }
}


void
Expand_Ceil (TN *dest, TN *src, TYPE_ID rtype, TYPE_ID desc, OPS *ops)
{
  TOP top;
  if (MTYPE_is_float(rtype)) {
    FmtAssert(rtype == desc, ("differing mtypes"));
    if (rtype == MTYPE_F4) 
  	top = TOP_cvt_fround_ftz_sat_f32_f32;
    else if (rtype == MTYPE_F8)
  	top = TOP_cvt_fround_sat_f64_f64;
    else
  	FmtAssert(FALSE, ("unexpected mtype"));
  } 
  else { // float to int
    top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32 
	+ 9*Mtype_Index(rtype) + (Mtype_Index(desc) - Mtype_Index(MTYPE_F4)));
  }
  if (desc == MTYPE_F4) {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rpi),
              Gen_Default_Enum_TN_ftz(),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  } else {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rpi),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  }
}

void
Expand_Round (TN *dest, TN *src, TYPE_ID rtype, TYPE_ID desc, OPS *ops)
{
  TOP top;
  if (MTYPE_is_float(rtype)) {
    FmtAssert(rtype == desc, ("differing mtypes"));
    if (rtype == MTYPE_F4) 
  	top = TOP_cvt_fround_ftz_sat_f32_f32;
    else if (rtype == MTYPE_F8)
  	top = TOP_cvt_fround_sat_f64_f64;
    else
  	FmtAssert(FALSE, ("unexpected mtype"));
  } 
  else { // float to int
    top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32 
	+ 9*Mtype_Index(rtype) + (Mtype_Index(desc) - Mtype_Index(MTYPE_F4)));
  }
  if (desc == MTYPE_F4) {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rni),
              Gen_Default_Enum_TN_ftz(),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  } else {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rni),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  }
}

void
Expand_Trunc (TN *dest, TN *src, TYPE_ID rtype, TYPE_ID desc, OPS *ops)
{
  TOP top;
  if (MTYPE_is_float(rtype)) {
    FmtAssert(rtype == desc, ("differing mtypes"));
    if (rtype == MTYPE_F4) 
  	top = TOP_cvt_fround_ftz_sat_f32_f32;
    else if (rtype == MTYPE_F8)
  	top = TOP_cvt_fround_sat_f64_f64;
    else
  	FmtAssert(FALSE, ("unexpected mtype"));
  } 
  else { // float to int
    if (MTYPE_size_reg(Mtype_Of_TN(dest)) == 32
	&& MTYPE_size_reg(rtype) < 32)
    {
	// may have cvt to smaller size but done in larger reg,
	// in which case we use different cvt (same eventual inst,
	// but takes different reg sizes).
	// just have s8/u8/s16/u16 rtypes for this, so not mtype_index.
	INT ridx;
	switch (rtype) {
	case MTYPE_I1: ridx = 0; break;
	case MTYPE_I2: ridx = 1; break;
	case MTYPE_U1: ridx = 2; break;
	case MTYPE_U2: ridx = 3; break;
	default: FmtAssert(FALSE, ("unexpected mtype"));
	}
	top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32_b32 
	  + 2*ridx + Mtype_Index(desc) - Mtype_Index(MTYPE_F4));
    }
    else if (MTYPE_size_reg(Mtype_Of_TN(dest)) == 64
	&& MTYPE_size_reg(rtype) < 64)
    {
	// same as 32bit case, doing cvt to smaller size but in larger reg.
	INT ridx;
	switch (rtype) {
	case MTYPE_I1: ridx = 0; break;
	case MTYPE_I2: ridx = 1; break;
	case MTYPE_I4: ridx = 2; break;
	case MTYPE_U1: ridx = 3; break;
	case MTYPE_U2: ridx = 4; break;
	case MTYPE_U4: ridx = 5; break;
	default: FmtAssert(FALSE, ("unexpected mtype"));
	}
	top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32_b64 
	  + 2*ridx + Mtype_Index(desc) - Mtype_Index(MTYPE_F4));
    }
    else {
      top = (TOP) (TOP_cvt_fround_ftz_sat_s8_f32 
	+ 9*Mtype_Index(rtype) + (Mtype_Index(desc) - Mtype_Index(MTYPE_F4)));
    }
  }
  if (desc == MTYPE_F4) {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rzi),
              Gen_Default_Enum_TN_ftz(),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  } else {
    Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rzi),
              Gen_Enum_TN(ECV_sat_none), src, ops);
  }
}


void
Expand_Float_To_Float (TN *dest, TN *src, TYPE_ID rtype, TYPE_ID dtype, OPS *ops)
{
  // if smaller->bigger no problem,
  // but if bigger->smaller then need round or trunc.
  // should only be float->double and double->float
  TOP top;
  if (rtype == MTYPE_F4 && dtype == MTYPE_F8) {
	top = TOP_cvt_fround_ftz_sat_f32_f64;
  }
  else if (rtype == MTYPE_F8 && dtype == MTYPE_F4) {
	top = TOP_cvt_fround_ftz_sat_f64_f32;
  } else {
  	FmtAssert(FALSE, ("unexpected case"));
  }
  Build_OP (top, dest, 
            Gen_Enum_TN(rtype == MTYPE_F4? ECV_fround_rn: ECV_fround_none),
            Gen_Default_Enum_TN_ftz(),
            Gen_Enum_TN(ECV_sat_none), src, ops);
}


// The params seem backwards to me, in that the mtypes are different order
// from TNs, but that is how the open-source prototypes it.
void
Expand_Int_To_Float (TN *dest, TN *src, TYPE_ID imtype, TYPE_ID fmtype, OPS *ops)
{
  TOP top;
  if (fmtype == MTYPE_F4) {
  	top = (TOP) (TOP_cvt_fround_sat_f32_s8 + Mtype_Index(imtype));
  } else if (fmtype == MTYPE_F8) {
  	top = (TOP) (TOP_cvt_fround_sat_f64_s8 + Mtype_Index(imtype));
  } else {
  	FmtAssert(FALSE, ("unexpected case"));
  }
  Build_OP (top, dest, Gen_Enum_TN(ECV_fround_rn),
            Gen_Enum_TN(ECV_sat_none), src, ops);
}


void
Expand_Select (
  TN *dest_tn, 
  TN *cond_tn, 
  TN *true_tn, 
  TN *false_tn, 
  TYPE_ID mtype, 
  VARIANT variant,
  OPS *ops)
{
  // Combine the memory spaces from a select operation.
  if (TN_has_memory_space(true_tn) && TN_has_memory_space(false_tn) &&
      TN_memory_space(true_tn) == TN_memory_space(false_tn))
    Set_TN_memory_space(dest_tn, TN_memory_space(true_tn));

  if (Mtype_Of_TN(cond_tn) == MTYPE_B) {
    TOP opc = (TOP) (TOP_selp_s8 + Mtype_Index(mtype));
    Build_OP (opc, dest_tn, true_tn, false_tn, cond_tn, ops);
    return;
  }

  TN *tmp;
  if (!V_normalized_op1(variant)) {
    // ptx slct does (c>=0) ? a : b
    // whereas whirl select is c ? a : b
    // so negate c (makes 0 or -1) and swap operands so get -c ? b : a
    tmp = Build_TN_Like(cond_tn);
    Expand_Neg(tmp, cond_tn, Mtype_Of_TN(cond_tn), ops);
    cond_tn = tmp;
  }

  // dest and src must be of same type size;
  // for now, assert if not same, eventually may need to convert.
  FmtAssert( MTYPE_bit_size(Mtype_Of_TN(dest_tn)) == MTYPE_bit_size(mtype), ("select mtype doesn't match"));
  FmtAssert( MTYPE_bit_size(Mtype_Of_TN(true_tn)) == MTYPE_bit_size(mtype), ("select mtype doesn't match"));
  FmtAssert( MTYPE_bit_size(Mtype_Of_TN(false_tn)) == MTYPE_bit_size(mtype), ("select mtype doesn't match"));
  // new ptx slct requires condition to be either s32 or f32
  FmtAssert( MTYPE_bit_size(Mtype_Of_TN(cond_tn)) == MTYPE_bit_size(MTYPE_I4), ("select condition must be 32bits"));
  FmtAssert(!TN_is_constant(cond_tn), ("select const NYI"));
  FmtAssert(!TN_is_constant(true_tn), ("select const NYI"));
  FmtAssert(!TN_is_constant(false_tn), ("select const NYI"));

  // whirl always puts condition in int, so float condition version
  // will never be generated (should we check for cvt of float under cond?)
  TOP opc = (TOP) (TOP_slct_s8_s32 + Mtype_Index(mtype));
  // cond is 0 or -1, so swap operands to get c>=0 ? b : a
  Build_OP (opc, dest_tn, false_tn, true_tn, cond_tn, ops);
}
  
void
Expand_Min (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{
  TOP opc = (TOP) (TOP_min_s8 + Mtype_Index(mtype));
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  if (mtype == MTYPE_F4) {
    Build_OP (opc, dest, Gen_Default_Enum_TN_ftz (), src1, src2, ops);
  } else {
    Build_OP (opc, dest, src1, src2, ops);
  }  
}

void
Expand_Max (TN *dest, TN *src1, TN *src2, TYPE_ID mtype, OPS *ops)
{     
  TOP opc = (TOP) (TOP_max_s8 + Mtype_Index(mtype));
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  if (mtype == MTYPE_F4) {
    Build_OP (opc, dest, Gen_Default_Enum_TN_ftz (), src1, src2, ops);
  } else {
    Build_OP (opc, dest, src1, src2, ops);
  }  
}

void
Expand_MinMax (TN *dest_min, TN *dest_max,
	       TN *src1, TN *src2,
	       TYPE_ID mtype, OPS *ops)
{ 
  if (TN_is_constant(src2)) {
	FmtAssert(TN_has_value(src2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        src2 = Expand_Mtype_Immediate_Into_Register (src2, mtype, ops);
  }
  Expand_Min(dest_min, src1, src2, mtype, ops);
  Expand_Max(dest_max, src1, src2, mtype, ops);
}

/* check whether to eval condition before select */
extern BOOL
Check_Select_Expansion (OPCODE compare)
{
  // in order to get optimal code,
  // don't evaluate the condition first,
  // but pass the condition and kids to exp_select,
  // which will do the compare and use the predicate results.
  return FALSE;
}


extern void 
Exp_Select_And_Condition (
        OPCODE select, TN *result, TN *true_tn, TN *false_tn,
        OPCODE compare, TN *cmp_kid1, TN *cmp_kid2, VARIANT variant, OPS *ops)
{
  OPS new_ops = OPS_EMPTY;
  const TYPE_ID mtype = OPCODE_rtype(select);
  TN *cond = Build_TN_Of_Mtype(OPCODE_rtype(compare));

  // ptx compare and select both will use -1/0 as condition,
  // whereas whirl uses 0/1.  Rather than switch back and forth 2 times,
  // use variant to mark that we should leave it in -1/0 format.
  VARIANT v = V_NONE;
  Set_V_normalized_op1(v);

  // just do compare separately, then select
  Exp_OP2v (compare, cond, cmp_kid1, cmp_kid2, v, ops);

  if (Trace_Exp) {
    fprintf(TFile, "expand %s: ", OPCODE_name(select));
    if (result) Print_TN(result,FALSE);
    fprintf(TFile, " :- (");
    if (cond) Print_TN(cond,FALSE);
    fprintf(TFile, ") ? ");
    if (true_tn) Print_TN(true_tn,FALSE);
    fprintf(TFile, " : ");
    if (false_tn) Print_TN(false_tn,FALSE);
    fprintf(TFile, " ");
    if (variant) fprintf(TFile, "(0x%llx)", (INT64)variant);
    fprintf(TFile, "\n");
  }

  Expand_Select (result, cond, true_tn, false_tn, mtype, v, &new_ops);

  if( Trace_Exp ){
    Print_OPS( &new_ops );
  }
  OPS_Append_Ops(ops, &new_ops);
}

void
Expand_Vote (TOP opcode, TN *result, TN *src, OPS *ops) {
  TYPE_ID mtype = Mtype_Of_TN (src);

  // Convert argument into a predicate
  if (mtype != MTYPE_B) {
    TN *cond = Build_TN_Of_Mtype (MTYPE_B);
    Expand_Not_Equal (cond, src, Gen_Literal_TN (0, 4), V_NONE, MTYPE_B, mtype, ops);
    src = cond;
  }

  // Emit vote insn, converting back to integer if needed
  mtype = Mtype_Of_TN (result);
  if (mtype == MTYPE_B || 
    opcode == TOP_vote_ballot_b32 ||
    opcode == TOP_vote_ballot_b32_not) 
  {
    Build_OP (opcode, result, src, ops);
  } else {  
    TN *cond = Build_TN_Of_Mtype (MTYPE_B);
    Build_OP (opcode, cond, src, ops);
    Expand_Bool_To_Int (result, cond, V_NONE, mtype, ops);
  }

}

static TN *
Build_Vote (TOP opcode, TN *src, OPS *ops) {
  TN *result = Build_TN_Of_Mtype(MTYPE_U4);
  Expand_Vote (opcode, result, src, ops);
  return result; 
}

static void
Expand_PMEvent (TN *counter, OPS *ops) {
  INT64 val;
  if (!TN_Can_Use_Constant_Value (counter, MTYPE_I4, &val)) {
    FmtAssert(FALSE, ("Argument to pmevent must be an immediate"));
    val = 0;
  }
  counter = Gen_Literal_TN_Of_Mtype (val, MTYPE_I4);
  Build_OP (TOP_pmevent, counter, ops);
}

static TN *
Build_BitCount (TOP opcode, TN *src, OPS *ops) {
  TN *result = Build_TN_Of_Mtype(MTYPE_U4);
  Build_OP (opcode, result, src, ops);
  return result; 
}

static void
Build_FMA (INTRINSIC id, TN *result, TN *op0, TN *op1, TN *op2, OPS *ops) {
  TN *fround_tn;
  TN *ftz_tn = Gen_Enum_TN (id >= INTRN_F4MA_IEEE_ROUND && id <= INTRN_F4MA_IEEE_CEIL
                            ? ECV_ftz_none
                            : TN_enum (Gen_Default_Enum_TN_ftz ()));
  TN *sat_tn = Gen_Enum_TN (ECV_sat_none);
  switch (id) {
    case INTRN_F4MA_ROUND:
    case INTRN_F4MA_IEEE_ROUND:
      fround_tn = Gen_Enum_TN (ECV_fround_rn);
      break;
    case INTRN_F4MA_TRUNC:
    case INTRN_F4MA_IEEE_TRUNC:
      fround_tn = Gen_Enum_TN (ECV_fround_rz);
      break;
    case INTRN_F4MA_FLOOR:
    case INTRN_F4MA_IEEE_FLOOR:
      fround_tn = Gen_Enum_TN (ECV_fround_rm);
      break;
    case INTRN_F4MA_CEIL:
    case INTRN_F4MA_IEEE_CEIL:
      fround_tn = Gen_Enum_TN (ECV_fround_rp);
      break;
    default:
      FmtAssert(FALSE, ("Unsupported FMA variant"));
  }    
  
  Build_OP (TOP_fma_fround_ftz_sat_f32, result,
            fround_tn, ftz_tn, sat_tn,
            op0, op1, op2, ops);
}

static TN *
Expand_Bar_red (TOP opcode, TN *src, OPS *ops) {
  TN *result = Build_TN_Of_Mtype(MTYPE_U4);
  TYPE_ID mtype = Mtype_Of_TN (src);

  if (mtype != MTYPE_B) {
    TN *cond = Build_TN_Of_Mtype (MTYPE_B);
    Expand_Not_Equal (cond, src, Gen_Literal_TN (0, 4), V_NONE, MTYPE_B, mtype, ops);
    src = cond;
  }

  // Currently the barrier source number would always be zero. 
  TN *Bar_Res_TN = Build_TN_Of_Mtype (MTYPE_U4);
  Build_OP (TOP_mov_u32_lit, Bar_Res_TN, Gen_Literal_TN (0, 4), ops);

  mtype = Mtype_Of_TN (result);
  if (opcode != TOP_bar_red_popc_u32 &&
      opcode != TOP_bar_red_popc_u32_not && 
      mtype != MTYPE_B) {
    TN *cond = Build_TN_Of_Mtype (MTYPE_B);
    Build_OP (opcode, cond, Bar_Res_TN, src, ops);
    Expand_Bool_To_Int (result, cond, V_NONE, mtype, ops);
  } else {  
    Build_OP (opcode, result, Bar_Res_TN, src, ops);
  }

  return result; 
}


void
Expand_Sqrt (TN *result, TN *src, TYPE_ID mtype, OPS *ops)
{
  FmtAssert( MTYPE_is_float(mtype), ("Unimplemented sqrt for integer") );
  FmtAssert(!TN_is_constant(src), ("NYI"));
  if (mtype == MTYPE_F4) {
  	Build_OP( TOP_sqrt_fround_ftz_f32, result,
  	          Gen_Default_Enum_TN_prec_sqrt (),
  	          Gen_Default_Enum_TN_ftz (),
  	          src, ops);
  }
  else if (mtype == MTYPE_F8) {
  	Build_OP( TOP_sqrt_rn_f64, result, src, ops);
  }
  else {
  	FmtAssert(FALSE, ("unexpected mtype"));
  }
}


void Expand_Flop ( OPCODE opcode, TN *result, TN *src1, TN *src2, TN *src3, OPS *ops )
{
  FmtAssert(FALSE, ("NYI"));
}

void Expand_Madd ( OPCODE opcode, TN *result, TN *src1, TN *src2, TN *src3, OPS *ops )
{
  TOP top;
  TN *tmp;
  TYPE_ID mtype = OPCODE_rtype(opcode);
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  FmtAssert(!TN_is_constant(src2), ("NYI"));
  FmtAssert(!TN_is_constant(src3), ("NYI"));
  if (MTYPE_is_float(mtype))
     top = (mtype == MTYPE_F4)
           ? Target_ISA >= TARGET_ISA_compute_20
             ? TOP_fma_fround_ftz_sat_f32
             : TOP_mad_f32
           : TOP_mad_rn_f64;
  else
     top = (TOP) (TOP_mad_lo_s8 + Mtype_Index(mtype));

  // may be incrementing through memory space:
  if (TN_has_memory_space(src1))
	Set_TN_memory_space (result, TN_memory_space(src1));

  switch (OPCODE_operator(opcode)) {
  // WHIRL MADD/etc have addend in src1, multipliers in src2,src3;
  // PTX puts addend last.
  case OPR_MADD:
        if (top == TOP_fma_fround_ftz_sat_f32) {
          Build_OP( top, result,
                    Gen_Enum_TN (ECV_fround_rn),
                    Gen_Default_Enum_TN_ftz (),
                    Gen_Enum_TN (ECV_sat_none),
                    src2, src3, src1, ops);
        } else {
          Build_OP( top, result, src2, src3, src1, ops);
        }  
	break;
  case OPR_NMADD:
	tmp = Build_TN_Like(result);
        if (top == TOP_fma_fround_ftz_sat_f32) {
          Build_OP( top, tmp,
                    Gen_Enum_TN (ECV_fround_rn),
                    Gen_Default_Enum_TN_ftz (),
                    Gen_Enum_TN (ECV_sat_none),
                    src2, src3, src1, ops);
        } else {
          Build_OP( top, tmp, src2, src3, src1, ops);
        }
	Expand_Neg (result, tmp, mtype, ops);
	break;
  case OPR_MSUB:
  	// only mad, no msub or nm* in ptx, so break up into mul/sub
  	// and let OCG optimize this, until OCG fixes optimization.
	tmp = Build_TN_Like(src2);
	Expand_Multiply (tmp, src2, src3, mtype, ops);
	Expand_Sub (result, tmp, src1, mtype, ops);
	break;
  case OPR_NMSUB:
	tmp = Build_TN_Like(src2);
	Expand_Multiply (tmp, src2, src3, mtype, ops);
	Expand_Sub (result, src1, tmp, mtype, ops);
	break;
  default:
    #pragma mips_frequency_hint NEVER
    FmtAssert(FALSE, ("Unimplemented madd: %s", OPCODE_name(opcode)));
  }
}

void Expand_Recip (TN *result, TN *src1, TYPE_ID mtype, OPS *ops)
{
  TOP top = (mtype == MTYPE_F4)
	     ? TOP_rcp_fround_ftz_f32
             : TOP_rcp_rn_f64;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (mtype == MTYPE_F4) {
    Build_OP( top, result,
              Gen_Default_Enum_TN_prec_rcp (),
              Gen_Default_Enum_TN_ftz (),
              src1, ops);
  } else {
    Build_OP( top, result, src1, ops);
  }  
}

void Expand_Recip_Sqrt (TN *result, TN *src1, TYPE_ID mtype, OPS *ops)
{
  TOP top = (mtype == MTYPE_F4) ? TOP_rsqrt_fround_ftz_f32 : TOP_rsqrt_fround_f64;
  FmtAssert(!TN_is_constant(src1), ("NYI"));
  if (mtype == MTYPE_F4) {
    Build_OP( top, result,
              Gen_Enum_TN (ECV_fround_approx),
              Gen_Default_Enum_TN_ftz (),
              src1, ops);
  } else {
    Build_OP( top, result,
              Gen_Enum_TN (ECV_fround_approx),
              src1, ops);
  }
}

extern void
Init_CG_Expand (void)
{
  static BOOL Initialized = FALSE;

  // per PU:
  Trace_Exp = Get_Trace (TP_CGEXP, 1);
  /* whirl2ops uses -ttexp:2 */
  Trace_Exp2 = Get_Trace (TP_CGEXP, 4);
  Disable_Const_Mult_Opt = Get_Trace (TP_CGEXP, 32);
  Exp_Ldst_Init();
  
  if (Initialized) return;
  Initialized = TRUE;
  // once per file:
  Initialize_Branch_Variants();
}


/* ======================================================================
 * Exp_COPY_Ext
 * 
 * Generate a register transfer copy from 'src_tn' to 'tgt_tn' with
 * appropriate sign/zero extension.
 * ======================================================================*/
void 
Exp_COPY_Ext (TOP opcode, TN *tgt_tn, TN *src_tn, OPS *ops)
{
  TOP new_op;
  FmtAssert(FALSE, ("NYI"));
}

/* ======================================================================
 * Exp_COPY
 * 
 * Generate a register transfer copy from 'src_tn' to 'tgt_tn'. 
 * ======================================================================*/
void 
Exp_COPY (TN *tgt_tn, TN *src_tn, OPS *ops)
{
  OPS newops = OPS_EMPTY;
  // src_tn and tgt_tn have to be of same size; we will use src_tn to derive
  // move type.
  TYPE_ID mtype;
  if (TN_size(src_tn) != TN_size(tgt_tn)) {
	// sizes don't match, so do convert rather than simple copy
	Expand_Convert (tgt_tn, Mtype_Of_TN(tgt_tn), src_tn, Mtype_Of_TN(src_tn), ops);
  }
  else if (TN_register_class(src_tn) != TN_register_class(tgt_tn)) {
	// do bit-move of int<->float regs
	TOP opc;
	switch (TN_register_class(tgt_tn)) {
	case ISA_REGISTER_CLASS_integer: opc = TOP_mov_b32_f2i; break;
	case ISA_REGISTER_CLASS_integer64: opc = TOP_mov_b64_f2i; break;
	case ISA_REGISTER_CLASS_float: opc = TOP_mov_b32_i2f; break;
	case ISA_REGISTER_CLASS_float64: opc = TOP_mov_b64_i2f; break;
	default: FmtAssert(FALSE, ("unexpected regclass"));
	}
  	Build_OP (opc, tgt_tn, src_tn, &newops);
	Set_OP_copy (OPS_last(&newops));
	if (TN_has_memory_space(src_tn))
		Set_TN_memory_space (tgt_tn, TN_memory_space(src_tn));
  }
  else if( TN_is_constant(src_tn) ){
    FmtAssert (TN_has_value(src_tn), ("Exp_COPY: illegal source tn"));
    /* expansion for INTCONST doesn't depend on size */
    Exp_OP1 (OPC_I4INTCONST, tgt_tn, src_tn, &newops);

  } else {
    switch (TN_register_class(tgt_tn)) {
    case ISA_REGISTER_CLASS_integer:
    case ISA_REGISTER_CLASS_integer16:
    case ISA_REGISTER_CLASS_integer64:
	switch (TN_size(tgt_tn)) {
	case 1: mtype = MTYPE_I1; break;
	case 2: mtype = MTYPE_I2; break;
	case 4: mtype = MTYPE_I4; break;
	case 8: mtype = MTYPE_I8; break;
	default: FmtAssert(FALSE, ("unexpected size"));
	}
    	Expand_Copy (tgt_tn, src_tn, mtype, &newops);
	break;
    case ISA_REGISTER_CLASS_float:
    case ISA_REGISTER_CLASS_float64:
	switch (TN_size(tgt_tn)) {
	case 4: mtype = MTYPE_F4; break;
	case 8: mtype = MTYPE_F8; break;
	default: FmtAssert(FALSE, ("unexpected size"));
	}
    	Expand_Copy (tgt_tn, src_tn, mtype, &newops);
	break;
    case ISA_REGISTER_CLASS_predicate:
	Build_OP (TOP_mov_pred, tgt_tn, src_tn, &newops);
 	Set_OP_copy (OPS_last(&newops));
	break;
    default: 
	FmtAssert(FALSE, ("unexpected regclass"));
    }
  }

  if (Trace_Exp2) {
    OP *op;
    FOR_ALL_OPS_OPs (&newops, op) {
      fprintf(TFile, "exp_copy into "); Print_OP (op);
    }
  }
  /* Add the new OPs to the end of the list passed in */
  OPS_Append_Ops(ops, &newops);
}

void
Exp_Intrinsic_Op (INTRINSIC id, TN *result, TN *op0, TN *op1, TN *op2, TYPE_ID mtype, OPS *ops)
{
  TN *tmp;
  switch (id) {
  case INTRN_F4SIN:
  	Build_OP (TOP_sin_fround_ftz_f32, result,
  	          Gen_Enum_TN (ECV_fround_approx),
  	          Gen_Default_Enum_TN_ftz (),
  	          op0, op1, ops);
	break;
  case INTRN_F4COS:
  	Build_OP (TOP_cos_fround_ftz_f32, result,
  	          Gen_Enum_TN (ECV_fround_approx),
  	          Gen_Default_Enum_TN_ftz (),
  	          op0, op1, ops);
	break;
  case INTRN_C_F4CEIL:
  	Build_OP (TOP_cvt_fround_ftz_sat_f32_f32, result,
  	          Gen_Enum_TN(ECV_fround_rpi),
  	          Gen_Default_Enum_TN_ftz (),
  	          Gen_Enum_TN(ECV_sat_none),
                  op0, ops);
	break;
  case INTRN_C_F8CEIL:
  	Build_OP (TOP_cvt_fround_sat_f64_f64, result, 
  	          Gen_Enum_TN(ECV_fround_rpi),
  	          Gen_Enum_TN(ECV_sat_none),
  	          op0, ops);
	break;
  case INTRN_C_F4TRUNC:
  	Build_OP (TOP_cvt_fround_ftz_sat_f32_f32, result, 
  	          Gen_Enum_TN(ECV_fround_rzi),
  	          Gen_Default_Enum_TN_ftz (),
  	          Gen_Enum_TN(ECV_sat_none),
                  op0, ops);
	break;
  case INTRN_C_F8TRUNC:
  	Build_OP (TOP_cvt_fround_sat_f64_f64, result, 
  	          Gen_Enum_TN(ECV_fround_rzi),
  	          Gen_Enum_TN(ECV_sat_none),
  	          op0, ops);
	break;
  case INTRN_C_F4ROUND:
  	Build_OP (TOP_cvt_fround_ftz_sat_f32_f32, result,
  	          Gen_Enum_TN(ECV_fround_rni),
  	          Gen_Default_Enum_TN_ftz (),
  	          Gen_Enum_TN(ECV_sat_none),
                  op0, ops);
	break;
  case INTRN_C_F8ROUND:
  	Build_OP (TOP_cvt_fround_sat_f64_f64, result,
  	          Gen_Enum_TN(ECV_fround_rni),
  	          Gen_Enum_TN(ECV_sat_none),
  	          op0, ops);
	break;
  case INTRN_F4EXP2:
  	Build_OP (TOP_ex2_fround_ftz_f32, result, 
  	          Gen_Enum_TN (ECV_fround_approx),
  	          Gen_Default_Enum_TN_ftz (),
  	          op0, ops);
	break;
  case INTRN_F8EXP2:
  	Build_OP (TOP_ex2_f64, result, op0, ops);
	break;
  case INTRN_F4LOG2:
  	Build_OP (TOP_lg2_fround_ftz_f32, result,
  	          Gen_Enum_TN (ECV_fround_approx),
  	          Gen_Default_Enum_TN_ftz (),
  	          op0, ops);
	break;
  case INTRN_F8LOG2:
  	Build_OP (TOP_lg2_f64, result, op0, ops);
	break;
  case INTRN_F4SATURATE:
  	Build_OP (TOP_cvt_fround_ftz_sat_f32_f32, result,
  	          Gen_Enum_TN(ECV_fround_none),
  	          Gen_Default_Enum_TN_ftz (),
  	          Gen_Enum_TN(ECV_sat_sat), op0, ops);
	break;
  case INTRN_F8SATURATE:
  	Build_OP (TOP_cvt_fround_sat_f64_f64, result,
  	          Gen_Enum_TN(ECV_fround_none),
  	          Gen_Enum_TN(ECV_sat_sat), op0, ops);
	break;
  case INTRN_MUL24:
        Build_OP (TOP_mul24_lo_s32, result, op0, op1, ops);
        break;
  case INTRN_UMUL24:
        Build_OP (TOP_mul24_lo_u32, result, op0, op1, ops);
        break;
  case INTRN_F4MA:
        if (Target_ISA >= TARGET_ISA_compute_20) {
          Build_OP (TOP_fma_fround_ftz_sat_f32, result,
                    Gen_Enum_TN (ECV_fround_rn),
                    Gen_Default_Enum_TN_ftz (),
                    Gen_Enum_TN (ECV_sat_none),
                    op0, op1, op2, ops);
        } else {
          Build_OP (TOP_mad_f32, result, op0, op1, op2, ops);
        }  
        break;
  case INTRN_F4DIV_APPROX:
        Build_OP (TOP_div_fround_ftz_f32, result,
                  Gen_Enum_TN (ECV_fround_approx),
                  Gen_Default_Enum_TN_ftz (),
                  op0, op1, ops);
        break;
  case INTRN_F4MA_ROUND:
  case INTRN_F4MA_TRUNC:
  case INTRN_F4MA_FLOOR:
  case INTRN_F4MA_CEIL:
  case INTRN_F4MA_IEEE_ROUND:
  case INTRN_F4MA_IEEE_TRUNC:
  case INTRN_F4MA_IEEE_FLOOR:
  case INTRN_F4MA_IEEE_CEIL:
        Build_FMA (id, result, op0, op1, op2, ops);
        break;
  case INTRN_BREV:
        Build_OP (TOP_brev_b32, result, op0, ops);
        break;
  case INTRN_BREVLL:
        Build_OP (TOP_brev_b64, result, op0, ops);
        break;
  case INTRN_F8MA_ROUND:
        Build_OP (TOP_mad_rn_f64, result, op0, op1, op2, ops);
        break;
  case INTRN_F8MA_TRUNC:
	Build_OP (TOP_mad_rz_f64, result, op0, op1, op2, ops);
        break;	
  case INTRN_F8MA_FLOOR:
	Build_OP (TOP_mad_rm_f64, result, op0, op1, op2, ops);
        break;	
  case INTRN_F8MA_CEIL:
	Build_OP (TOP_mad_rp_f64, result, op0, op1, op2, ops);
        break;
  case INTRN_F4ADD_ROUND:
	Build_OP (TOP_add_fround_ftz_sat_f32, result, 
	          Gen_Enum_TN(ECV_fround_rn),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;	
  case INTRN_F4ADD_TRUNC:
	Build_OP (TOP_add_fround_ftz_sat_f32, result,
	          Gen_Enum_TN(ECV_fround_rz),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;
  case INTRN_F4ADD_FLOOR:
	Build_OP (TOP_add_fround_ftz_sat_f32, result,
	          Gen_Enum_TN(ECV_fround_rm),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;
  case INTRN_F4ADD_CEIL:
	Build_OP (TOP_add_fround_ftz_sat_f32, result,
	          Gen_Enum_TN(ECV_fround_rp),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;
  case INTRN_F8ADD_ROUND:
	Build_OP (TOP_add_rn_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8ADD_TRUNC:
	Build_OP (TOP_add_rz_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8ADD_FLOOR:
	Build_OP (TOP_add_rm_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8ADD_CEIL:
	Build_OP (TOP_add_rp_f64, result, op0, op1, ops);
        break;	
  case INTRN_F4MUL_ROUND:
	Build_OP (TOP_mul_fround_ftz_sat_f32, result, 
	          Gen_Enum_TN(ECV_fround_rn),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;	
  case INTRN_F4MUL_TRUNC:
	Build_OP (TOP_mul_fround_ftz_sat_f32, result, 
	          Gen_Enum_TN(ECV_fround_rz),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;	
  case INTRN_F4MUL_FLOOR:
	Build_OP (TOP_mul_fround_ftz_sat_f32, result, 
	          Gen_Enum_TN(ECV_fround_rm),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;	
  case INTRN_F4MUL_CEIL:
	Build_OP (TOP_mul_fround_ftz_sat_f32, result, 
	          Gen_Enum_TN(ECV_fround_rp),
                  Gen_Default_Enum_TN_ftz (),
	          Gen_Enum_TN(ECV_sat_none),
	          op0, op1, ops);
        break;
  case INTRN_F8MUL_ROUND:
	Build_OP (TOP_mul_rn_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8MUL_TRUNC:
	Build_OP (TOP_mul_rz_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8MUL_FLOOR:
	Build_OP (TOP_mul_rm_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8MUL_CEIL:
	Build_OP (TOP_mul_rp_f64, result, op0, op1, ops);
        break;	
  case INTRN_F4DIV_ROUND:
	Build_OP (TOP_div_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, op1, ops);
        break;	
  case INTRN_F4DIV_TRUNC:
	Build_OP (TOP_div_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, op1, ops);
        break;	
  case INTRN_F4DIV_FLOOR:
	Build_OP (TOP_div_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, op1, ops);
        break;	
  case INTRN_F4DIV_CEIL:
	Build_OP (TOP_div_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, op1, ops);
        break;	
  case INTRN_F4RCP_ROUND:
	Build_OP (TOP_rcp_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4RCP_TRUNC:
	Build_OP (TOP_rcp_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4RCP_FLOOR:
	Build_OP (TOP_rcp_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4RCP_CEIL:
	Build_OP (TOP_rcp_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4SQRT_ROUND:
	Build_OP (TOP_sqrt_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4SQRT_TRUNC:
	Build_OP (TOP_sqrt_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4SQRT_FLOOR:
	Build_OP (TOP_sqrt_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F4SQRT_CEIL:
	Build_OP (TOP_sqrt_fround_ftz_f32, result, Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(CG_use_ftz? ECV_ftz_ftz: ECV_ftz_none), op0, ops);
        break;	
  case INTRN_F8DIV_ROUND:
	Build_OP (TOP_div_rn_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8DIV_TRUNC:
	Build_OP (TOP_div_rz_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8DIV_FLOOR:
	Build_OP (TOP_div_rm_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8DIV_CEIL:
	Build_OP (TOP_div_rp_f64, result, op0, op1, ops);
        break;	
  case INTRN_F8RCP_ROUND:
	Build_OP (TOP_rcp_rn_f64, result, op0, ops);
        break;	
  case INTRN_F8RCP_TRUNC:
	Build_OP (TOP_rcp_rz_f64, result, op0, ops);
        break;	
  case INTRN_F8RCP_FLOOR:
	Build_OP (TOP_rcp_rm_f64, result, op0, ops);
        break;	
  case INTRN_F8RCP_CEIL:
	Build_OP (TOP_rcp_rp_f64, result, op0, ops);
        break;	
  case INTRN_F8RCP64H:
	Build_OP (TOP_rcp_approx_ftz_f64, result, op0, ops);
        break;	
  case INTRN_F8SQRT_ROUND:
	Build_OP (TOP_sqrt_rn_f64, result, op0, ops);
        break;	
  case INTRN_F8SQRT_TRUNC:
	Build_OP (TOP_sqrt_rz_f64, result, op0, ops);
        break;	
  case INTRN_F8SQRT_FLOOR:
	Build_OP (TOP_sqrt_rm_f64, result, op0, ops);
        break;	
  case INTRN_F8SQRT_CEIL:
	Build_OP (TOP_sqrt_rp_f64, result, op0, ops);
        break;	
  case INTRN_I4SAD:
	Build_OP (TOP_sad_s32, result, op0, op1, op2, ops);
        break;	
  case INTRN_U4SAD:
	Build_OP (TOP_sad_u32, result, op0, op1, op2, ops);
        break;	
  case INTRN_F4F8CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_f32_f64, result,
	          Gen_Enum_TN(ECV_fround_rn),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F4F8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_ftz_sat_f32_f64, result,
	          Gen_Enum_TN(ECV_fround_rz),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F4F8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_ftz_sat_f32_f64, result,
	          Gen_Enum_TN(ECV_fround_rm),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F4F8CVT_CEIL:
	Build_OP (TOP_cvt_fround_ftz_sat_f32_f64, result,
	          Gen_Enum_TN(ECV_fround_rp),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_I4F8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_s32_f64, result,
	          Gen_Enum_TN(ECV_fround_rni),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_I4F8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_s32_f64, result,
	          Gen_Enum_TN(ECV_fround_rzi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_I4F8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_s32_f64, result,
	          Gen_Enum_TN(ECV_fround_rmi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_I4F8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_s32_f64, result,
	          Gen_Enum_TN(ECV_fround_rpi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_U4F8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_u32_f64, result,
	          Gen_Enum_TN(ECV_fround_rni),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_U4F8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_u32_f64, result,
	          Gen_Enum_TN(ECV_fround_rzi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_U4F8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_u32_f64, result,
	          Gen_Enum_TN(ECV_fround_rmi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_U4F8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_u32_f64, result,
	          Gen_Enum_TN(ECV_fround_rpi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8I4CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f64_s32, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8I4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f64_s32, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8I4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f64_s32, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8I4CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f64_s32, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8U4CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f64_u32, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8U4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f64_u32, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8U4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f64_u32, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F8U4CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f64_u32, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
        break;	
  case INTRN_F4I4CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f32_s32, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f32_s32, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f32_s32, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I4CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f32_s32, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U4CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f32_u32, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f32_u32, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f32_u32, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U4CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f32_u32, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I4F4CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_s32_f32, result,
	          Gen_Enum_TN(ECV_fround_rni),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I4F4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_ftz_sat_s32_f32, result,
	          Gen_Enum_TN(ECV_fround_rzi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I4F4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_ftz_sat_s32_f32, result,
	          Gen_Enum_TN(ECV_fround_rmi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I4F4CVT_CEIL:
	Build_OP (TOP_cvt_fround_ftz_sat_s32_f32, result,
	          Gen_Enum_TN(ECV_fround_rpi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U4F4CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_u32_f32, result,
	          Gen_Enum_TN(ECV_fround_rni),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U4F4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_ftz_sat_u32_f32, result,
	          Gen_Enum_TN(ECV_fround_rzi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U4F4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_ftz_sat_u32_f32, result,
	          Gen_Enum_TN(ECV_fround_rmi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U4F4CVT_CEIL:
	Build_OP (TOP_cvt_fround_ftz_sat_u32_f32, result,
	          Gen_Enum_TN(ECV_fround_rpi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f32_s64, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f32_s64, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f32_s64, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4I8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f32_s64, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f32_u64, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f32_u64, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f32_u64, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4U8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f32_u64, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8I8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f64_s64, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8I8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f64_s64, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8I8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f64_s64, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8I8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f64_s64, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8U8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_f64_u64, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8U8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_f64_u64, result,
	          Gen_Enum_TN(ECV_fround_rz),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8U8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_f64_u64, result,
	          Gen_Enum_TN(ECV_fround_rm),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8U8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_f64_u64, result,
	          Gen_Enum_TN(ECV_fround_rp),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F4CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_s64_f32, result,
	          Gen_Enum_TN(ECV_fround_rni),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_ftz_sat_s64_f32, result,
	          Gen_Enum_TN(ECV_fround_rzi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_ftz_sat_s64_f32, result,
	          Gen_Enum_TN(ECV_fround_rmi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F4CVT_CEIL:
	Build_OP (TOP_cvt_fround_ftz_sat_s64_f32, result,
	          Gen_Enum_TN(ECV_fround_rpi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F4CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_u64_f32, result,
	          Gen_Enum_TN(ECV_fround_rni),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F4CVT_TRUNC:
	Build_OP (TOP_cvt_fround_ftz_sat_u64_f32, result,
	          Gen_Enum_TN(ECV_fround_rzi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F4CVT_FLOOR:
	Build_OP (TOP_cvt_fround_ftz_sat_u64_f32, result,
	          Gen_Enum_TN(ECV_fround_rmi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F4CVT_CEIL:
	Build_OP (TOP_cvt_fround_ftz_sat_u64_f32, result,
	          Gen_Enum_TN(ECV_fround_rpi),
                  Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_s64_f64, result,
	          Gen_Enum_TN(ECV_fround_rni),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_s64_f64, result,
	          Gen_Enum_TN(ECV_fround_rzi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_s64_f64, result,
	          Gen_Enum_TN(ECV_fround_rmi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_I8F8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_s64_f64, result,
	          Gen_Enum_TN(ECV_fround_rpi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F8CVT_ROUND:
	Build_OP (TOP_cvt_fround_sat_u64_f64, result,
	          Gen_Enum_TN(ECV_fround_rni),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F8CVT_TRUNC:
	Build_OP (TOP_cvt_fround_sat_u64_f64, result,
	          Gen_Enum_TN(ECV_fround_rzi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F8CVT_FLOOR:
	Build_OP (TOP_cvt_fround_sat_u64_f64, result,
	          Gen_Enum_TN(ECV_fround_rmi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_U8F8CVT_CEIL:
	Build_OP (TOP_cvt_fround_sat_u64_f64, result,
	          Gen_Enum_TN(ECV_fround_rpi),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F2F4CVT_ROUND:
	Build_OP (TOP_cvt_fround_ftz_sat_f16_f32, result,
	          Gen_Enum_TN(ECV_fround_rn),
	          Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F4F2CVT:
	Build_OP (TOP_cvt_fround_ftz_sat_f32_f16, result,
	          Gen_Enum_TN(ECV_fround_none),
	          Gen_Default_Enum_TN_ftz(),
	          Gen_Enum_TN(ECV_sat_none), op0, ops);
	break;
  case INTRN_F8HLI2D:
	// little endian means we write it as {lo,hi}
	Build_OP (TOP_mov_b64_i22f, result, op1, op0, ops);
	break;
  case INTRN_I4D2LI:
	tmp = Build_TN_Like(result); 
	Build_OP (TOP_mov_b64_f2i2, result, tmp, op0, ops);
	break;
  case INTRN_I4D2HI:
	tmp = Build_TN_Like(result); 
	Build_OP (TOP_mov_b64_f2i2, tmp, result, op0, ops);
	break;
  case INTRN_I4I4CLZ:
        Build_OP (TOP_clz_b32, result, op0, ops);
        break;
  case INTRN_I4I8CLZ:
        Build_OP (TOP_clz_b64, result, op0, ops);
        break;
  case INTRN_I4U4POPC:
        Build_OP (TOP_popc_b32, result, op0, ops);
        break;
  case INTRN_I4U8POPC:
        Build_OP (TOP_popc_b64, result, op0, ops);
        break;
  case INTRN_U4U4BYTEPERM:
        Build_OP (TOP_and_b32_lit, op2, op2, Gen_Literal_TN(0x00007777,4), ops);
        Build_OP (TOP_prmt_b32, result, op0, op1, op2, ops);
        break;
  default:
  	FmtAssert(FALSE, ("intrinsic NYI"));
  }
  return;
}

/* ======================================================================
 * Expand_TOP_intrncall
 * 
 * Given a TOP_intrncall <op>, expand it into the sequence of instructions 
 * that must be generated. If <get_sequence_length> is TRUE, return only
 * the number of instructions in the sequence and don't actually do the 
 * expansion.
 * ======================================================================*/
static INT
Expand_TOP_intrncall (
  const OP *op, 
  OPS *ops, 
  BOOL get_sequence_length,
  INT pc_value)
{
  ErrMsg( EC_Unimplemented, "Expand_TOP_intrncall: NYI" );
  return 0;
}


// initial expansion of intrinsic call (may not be complete lowering).
// return result TN (if set).
// If the intrinsic requires a label and loop (2 bb's)
// then ops is for first bb and ops2 is for bb after the label.
// Otherwise only ops is filled in.
TN *
Exp_Intrinsic_Call (INTRINSIC id, VARIANT v, TN *op0, TN *op1, TN *op2,
                    OPS *ops, LABEL_IDX *label, OPS *loop_ops)
{
  TN *call_iresult, *call_fresult, *call_dresult;
  TOP  at_opc;
  BOOL addr_is64 = op0 != NULL && TN_size(op0) == 8;

  // for now put constants into registers just to simplify the targ_info;
  // ocg will propagate these anyway.
  if (op1 && TN_is_constant(op1)) {
	FmtAssert(TN_has_value(op1), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        op1 = Expand_Mtype_Immediate_Into_Register (op1, Mtype_Of_TN(op1), ops);
  }
  if (op2 && TN_is_constant(op2)) {
	FmtAssert(TN_has_value(op2), ("NYI"));
        // expand op a,const into mov t,const; op a,t;
        op2 = Expand_Mtype_Immediate_Into_Register (op2, Mtype_Of_TN(op2), ops);
  }

  ST_MEMORY mem = MEMORY_UNKNOWN;
  TN *space_tn = 0;
  switch (id) {
  case INTRN_I4ATOMICADD:
  case INTRN_U4ATOMICADD:
  case INTRN_U8ATOMICADD:
  case INTRN_F4ATOMICADD:
  case INTRN_F8ATOMICADD:
  case INTRN_I4ATOMICMIN:
  case INTRN_U4ATOMICMIN:
  case INTRN_F4ATOMICMIN:
  case INTRN_I4ATOMICMAX:
  case INTRN_U4ATOMICMAX:
  case INTRN_F4ATOMICMAX:
  case INTRN_I4ATOMICEXCH:
  case INTRN_U4ATOMICEXCH:
  case INTRN_U8ATOMICEXCH:
  case INTRN_F4ATOMICEXCH:
  case INTRN_F8ATOMICEXCH:
  case INTRN_U4ATOMICAND:
  case INTRN_U4ATOMICOR:
  case INTRN_U4ATOMICXOR:
  case INTRN_U4ATOMICINC:
  case INTRN_U4ATOMICDEC:
  case INTRN_I4ATOMICCAS:
  case INTRN_U4ATOMICCAS:
  case INTRN_U8ATOMICCAS:
  case INTRN_F4ATOMICCAS:
  case INTRN_F8ATOMICCAS:
    if (TN_has_spill(op0))
      mem = BE_ST_memory_pointed_to(TN_spill(op0));
    {
      ISA_ENUM_CLASS_VALUE ecv = ECV_space_none;
      if (mem == MEMORY_GLOBAL)
        ecv = ECV_space_global;
      else if (mem == MEMORY_SHARED)
        ecv = ECV_space_shared;
      // Check the variant only after the memory space information from
      // the TN has been checked.
      else if (V_global_mem(v))
        ecv = ECV_space_global;
      else if (V_shared_mem(v))
        ecv = ECV_space_shared;
      else if (Target_ISA < TARGET_ISA_compute_20) {
        ErrMsgSrcpos (EC_Ptr_Assumed_Global, current_srcpos);
        ecv = ECV_space_global;
      }
      space_tn = Gen_Enum_TN(ecv);
    }  
    break;
  }

  switch (id) {
  case INTRN_SYNCTHREADS:
  	Build_OP (TOP_bar_sync, Gen_Literal_TN(0, 4), ops);
	return NULL;
  case INTRN_THREADFENCE:
        Build_OP (TOP_membar_gl, ops);
        return NULL;
  case INTRN_THREADFENCE_BLOCK:
        Build_OP (TOP_membar_cta, ops);
        return NULL;
  case INTRN_THREADFENCE_SYS:
        Build_OP (TOP_membar_sys, ops);
        return NULL;
  case INTRN_BRKPT:
  	Build_OP (TOP_brkpt, ops);
	return NULL;
  case INTRN_TRAP:
  	Build_OP (TOP_trap, ops);
	return NULL;
  case INTRN_CLOCK:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
  	Build_OP (TOP_mov_u32, call_iresult, Clock_TN(), ops);
	return call_iresult;
  case INTRN_CLOCK64:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U8);
  	Build_OP (TOP_mov_u64, call_iresult, Clock64_TN(), ops);
	return call_iresult;
  case INTRN_I4ATOMICADD:
        at_opc = addr_is64 ? TOP_atom_space_add_s32_a64 : TOP_atom_space_add_s32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_I4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICADD:
        at_opc = addr_is64 ? TOP_atom_space_add_u32_a64 : TOP_atom_space_add_u32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U8ATOMICADD:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_add_u64_a64 : TOP_atom_space_add_u64;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U8);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_F4ATOMICADD:
        at_opc = addr_is64 ? TOP_atom_space_add_f32_a64 : TOP_atom_space_add_f32;
	call_fresult = Build_TN_Of_Mtype(MTYPE_F4);
	Build_OP (at_opc, call_fresult, space_tn, op0, op1, ops);
	return call_fresult;
  case INTRN_F8ATOMICADD:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_add_f64_a64 : TOP_atom_space_add_f64;
	call_dresult = Build_TN_Of_Mtype(MTYPE_F8);
	Build_OP (at_opc, call_dresult, space_tn, op0, op1, ops);
	return call_dresult;
  case INTRN_I4ATOMICMIN:
        at_opc = addr_is64 ? TOP_atom_space_min_s32_a64 : TOP_atom_space_min_s32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_I4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICMIN:
        at_opc = addr_is64 ? TOP_atom_space_min_u32_a64 : TOP_atom_space_min_u32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_F4ATOMICMIN:
        at_opc = addr_is64 ? TOP_atom_space_min_f32_a64 : TOP_atom_space_min_f32;
	call_fresult = Build_TN_Of_Mtype(MTYPE_F4);
	Build_OP (at_opc, call_fresult, space_tn, op0, op1, ops);
	return call_fresult;
  case INTRN_I4ATOMICMAX:
        at_opc = addr_is64 ? TOP_atom_space_max_s32_a64 : TOP_atom_space_max_s32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_I4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICMAX:
        at_opc = addr_is64 ? TOP_atom_space_max_u32_a64 : TOP_atom_space_max_u32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_F4ATOMICMAX:
        at_opc = addr_is64 ? TOP_atom_space_max_f32_a64 : TOP_atom_space_max_f32;
	call_fresult = Build_TN_Of_Mtype(MTYPE_F4);
	Build_OP (at_opc, call_fresult, space_tn, op0, op1, ops);
	return call_fresult;
  case INTRN_I4ATOMICEXCH:
  case INTRN_U4ATOMICEXCH:
        at_opc = addr_is64 ? TOP_atom_space_exch_b32_a64 : TOP_atom_space_exch_b32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U8ATOMICEXCH:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_exch_b64_a64 : TOP_atom_space_exch_b64;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U8);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_F4ATOMICEXCH:
        at_opc = addr_is64 ? TOP_atom_space_exch_b32_a64_f : TOP_atom_space_exch_b32_f;
	call_fresult = Build_TN_Of_Mtype(MTYPE_F4);
	Build_OP (at_opc, call_fresult, space_tn, op0, op1, ops);
	return call_fresult;
  case INTRN_F8ATOMICEXCH:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_exch_b64_a64_f : TOP_atom_space_exch_b64_f;
	call_dresult = Build_TN_Of_Mtype(MTYPE_F8);
	Build_OP (at_opc, call_dresult, space_tn, op0, op1, ops);
	return call_dresult;
  case INTRN_U4ATOMICAND:
        at_opc = addr_is64 ? TOP_atom_space_and_b32_a64 : TOP_atom_space_and_b32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICOR:
        at_opc = addr_is64 ? TOP_atom_space_or_b32_a64 : TOP_atom_space_or_b32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICXOR:
        at_opc = addr_is64 ? TOP_atom_space_xor_b32_a64 : TOP_atom_space_xor_b32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICINC:
        at_opc = addr_is64 ? TOP_atom_space_inc_u32_a64 : TOP_atom_space_inc_u32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_U4ATOMICDEC:
        at_opc = addr_is64 ? TOP_atom_space_dec_u32_a64 : TOP_atom_space_dec_u32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, ops);
	return call_iresult;
  case INTRN_I4ATOMICCAS:
  case INTRN_U4ATOMICCAS:
        at_opc = addr_is64 ? TOP_atom_space_cas_b32_a64 : TOP_atom_space_cas_b32;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, op2, ops);
	return call_iresult;
  case INTRN_U8ATOMICCAS:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_cas_b64_a64 : TOP_atom_space_cas_b64;
	call_iresult = Build_TN_Of_Mtype(MTYPE_U8);
	Build_OP (at_opc, call_iresult, space_tn, op0, op1, op2, ops);
	return call_iresult;
  case INTRN_F4ATOMICCAS:
        at_opc = addr_is64 ? TOP_atom_space_cas_b32_a64_f : TOP_atom_space_cas_b32_f;
	call_fresult = Build_TN_Of_Mtype(MTYPE_F4);
	Build_OP (at_opc, call_fresult, space_tn, op0, op1, op2, ops);
	return call_fresult;
  case INTRN_F8ATOMICCAS:
        if (mem == MEMORY_SHARED && Target_ISA < TARGET_ISA_compute_20)
          ErrMsgSrcpos(EC_Shared_Atomic64_Opnd, current_srcpos);
        at_opc = addr_is64 ? TOP_atom_space_cas_b64_a64_f : TOP_atom_space_cas_b64_f;
	call_dresult = Build_TN_Of_Mtype(MTYPE_F8);
	Build_OP (at_opc, call_dresult, space_tn, op0, op1, op2, ops);
	return call_dresult;

  case INTRN_VOTEALL:
        return Build_Vote (TOP_vote_all_pred, op0, ops);
  case INTRN_VOTEANY:
        return Build_Vote (TOP_vote_any_pred, op0, ops);
  case INTRN_VOTEUNI:
        return Build_Vote (TOP_vote_uni_pred, op0, ops);
  case INTRN_VOTEALL_NOT:
        return Build_Vote (TOP_vote_all_pred_not, op0, ops);
  case INTRN_VOTEANY_NOT:
        return Build_Vote (TOP_vote_any_pred_not, op0, ops);
  case INTRN_VOTEBALLOT:
        return Build_Vote (TOP_vote_ballot_b32, op0, ops);
  case INTRN_BARREDPOPC:
        return Expand_Bar_red (TOP_bar_red_popc_u32, op0, ops);
  case INTRN_BARREDAND:
        return Expand_Bar_red (TOP_bar_red_and_pred, op0, ops);
  case INTRN_BARREDOR:
        return Expand_Bar_red (TOP_bar_red_or_pred, op0, ops);

  case INTRN_PMEVENT:
        Expand_PMEvent (op0, ops);
        return NULL;
  case INTRN_PM0:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
  	Build_OP (TOP_mov_u32, call_iresult, PM_TN(0), ops);
	return call_iresult;
  case INTRN_PM1:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
  	Build_OP (TOP_mov_u32, call_iresult, PM_TN(1), ops);
	return call_iresult;
  case INTRN_PM2:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
  	Build_OP (TOP_mov_u32, call_iresult, PM_TN(2), ops);
	return call_iresult;
  case INTRN_PM3:
	call_iresult = Build_TN_Of_Mtype(MTYPE_U4);
  	Build_OP (TOP_mov_u32, call_iresult, PM_TN(3), ops);
	return call_iresult;

  default:  
    FmtAssert( FALSE, ("NYI intrinsic call") );
  }
}


/* ======================================================================
 * Exp_Simulated_Op
 *
 * Given a simulated <op>, expand it into the sequence of instructions
 * supported by the target.
 * ======================================================================*/
void Exp_Simulated_Op(const OP *op, OPS *ops, INT pc_value)
{
  TOP top = OP_code(op);
  BB *bb = OP_bb(op);

  switch (top)
  {
  default:
    FmtAssert(FALSE, ("simulated OP %s not handled", TOP_Name(top)));
  }
}


/* ======================================================================
 * Simulated_Op_Real_Ops
 *
 * Return the number of instructions that will be generated by Exp_Simulated_Op
 * ======================================================================*/
INT
Simulated_Op_Real_Ops(const OP *op)
{
  switch (OP_code(op)) {
  case TOP_intrncall:
    return Expand_TOP_intrncall (op, NULL, TRUE, 0);
  default:

    /* Anything other than the above is presumed to be removed by
     * emit time, therefore we just say the expansion generates 0 ops.
     * (we used to assert, but that isn't a good solution -- see pv 652898).
     */
    return 0;
  }
}


/* ======================================================================
 * Simulated_Op_Real_Inst_Words
 *
 * Return the number of instruction words that will ultimately be emitted
 * for the expansion generated by Exp_Simulated_Op
 * ======================================================================*/
INT
Simulated_Op_Real_Inst_Words(const OP *op)
{
  TOP top = OP_code(op);
  switch (top)
  {
  case TOP_asm:
    /* We don't know how many instructions are "within" the asm, so we
       just assume 3 bytes. */
    return 3;
  case TOP_call:
  case TOP_call_uni:
    return 1;

  default:
    FmtAssert(FALSE, ("simulated OP %s not handled", TOP_Name(OP_code(op))));
    return 0;
  }
}


/* ======================================================================
 * Exp_Is_Large_Stack_Sym
 *
 * determine if a given symbol is a stack relative reference that will
 * require multiple instructions to load or store.
 * ======================================================================*/
BOOL
Exp_Is_Large_Stack_Sym(ST* sym,  INT64 ofst)
{
  ST *base_sym;
  INT64 base_ofst;
  
  if (sym == NULL)
    return FALSE;

  Allocate_Object(sym);
  Base_Symbol_And_Offset_For_Addressing (sym, ofst, &base_sym, &base_ofst);

  /* We can assume that 'sym' is a spill location for an integer
     register, so we can check for l32i/s32i range. */
  
  return FALSE;
}

void Exp_Noop (OPS *ops)
{
  Build_OP (CGTARG_Noop_Top(), ops);
}

void Expand_Const (TN *dest, TN *src, TYPE_ID mtype, OPS *ops)
{
  // fp lit is stored as symbol, do mov of symbol
  TOP opc = (TOP) (TOP_mov_s8_lit + Mtype_Index(mtype));
  FmtAssert( TN_is_symbol(src), ("Expand_Const: src not a symbol TN") );
  Build_OP (opc, dest, src, ops);
}

static BB* last_bb = NULL;
static TN *last_true_tn = NULL, *last_false_tn = NULL;
void
HB_Reinit_Pred ()
{
  last_true_tn = NULL;
  last_false_tn = NULL;
  last_bb = NULL;
}

void
Exp_True_False_Preds_For_Block(BB *bb, TN* &true_tn, TN * &false_tn)
{ 
  if (last_bb != bb)
    last_bb = bb;
  else {
    true_tn = last_true_tn;
    false_tn = last_false_tn;
    return;
  }
  OP* br_op = BB_branch_op(bb);
  if (!br_op)
    return;

  FmtAssert( FALSE, ("UNIMPLEMENTED") );
}

BOOL
Target_Has_Immediate_Operand (WN *parent, WN *expr)
{
  OPERATOR opr = WN_operator(parent);
  return opr == OPR_ADD || opr == OPR_SUB || opr == OPR_EQ ||
         opr == OPR_BAND || opr == OPR_BIOR || opr == OPR_BXOR ||
         opr == OPR_LT || opr == OPR_LE || opr == OPR_GT || opr == OPR_GE ||
         opr == OPR_LSHR || opr == OPR_ASHR || opr == OPR_SHL;
}

void 
Exp_Spadjust (TN *dest, TN *size, VARIANT variant, OPS *ops)
{
  Build_OP (TOP_spadjust, dest, SP_TN, size, ops);
  OP_variant(OPS_last(ops)) = variant;
}

/* Return a unique name for a symbol representing a literal. */
char *
Exp_Unique_Literal_Name (void)
{
  static int unique;
  static char name[32];

  sprintf(name, ".LC%d", unique);
  unique++;
  return name;
}


/* Expand FETCH_AND_ADD intrinsic into the following format
      lock (addr) = (addr) + opnd1
 */
void Exp_Fetch_and_Add( TN* addr, TN* opnd1, TYPE_ID mtype, OPS* ops )
{
  TOP top = TOP_UNDEFINED;

  FmtAssert(FALSE, ("NYI"));
}

TN* Exp_Compare_and_Swap( TN* addr, TN* opnd1, TN* opnd2, TYPE_ID mtype, OPS* ops )
{
  FmtAssert(FALSE, ("NYI"));
  return NULL;
}

// Expand FETCH_AND_AND intrinsic into the following format
//        lock (addr) = (addr) & opnd1
void Exp_Fetch_and_And( TN* addr, TN* opnd1, TYPE_ID mtype, OPS* ops )
{
  TOP top = TOP_UNDEFINED;
  FmtAssert(FALSE, ("NYI"));

}

// Expand FETCH_AND_OR intrinsic into the following format
//        lock (addr) = (addr) | opnd1
void Exp_Fetch_and_Or( TN* addr, TN* opnd1, TYPE_ID mtype, OPS* ops )
{
  TOP top = TOP_UNDEFINED;
  FmtAssert(FALSE, ("NYI"));

}

// Expand FETCH_AND_XOR intrinsic into the following format
//        lock (addr) = (addr) ^ opnd1
void Exp_Fetch_and_Xor( TN* addr, TN* opnd1, TYPE_ID mtype, OPS* ops )
{
  TOP top = TOP_UNDEFINED;
  FmtAssert(FALSE, ("NYI"));
}

// Expand FETCH_AND_SUB intrinsic into the following format
//        lock (addr) = (addr) - opnd1
void Exp_Fetch_and_Sub( TN* addr, TN* opnd1, TYPE_ID mtype, OPS* ops )
{
  TOP top = TOP_UNDEFINED;
  FmtAssert(FALSE, ("NYI"));
}
