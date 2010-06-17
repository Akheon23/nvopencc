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

//
// Generate an ISA containing and group TOPS with similar operands/results 
// format.
/////////////////////////////////////////////////////////
// The instructions are listed below in the order as shown below. 
//
//   1. Real and pseudo instructions
//   2. Simulated instructions.
//
// Within each category, the instructions are arranged roughly in order
// of increasing numbers of operands.
/////////////////////////////////////
//
//  $Revision: 1.47 $
//  $Date: 2001/03/10 01:16:39 $
//  $Author: mtibuild $
//  $Source: /osprey.src/osprey1.0/common/targ_info/isa/ia64/RCS/isa_operands.cxx,v $

#include <stddef.h>
#include "topcode.h"
#include "isa_operands_gen.h"

int main()
{
  OPERAND_VALUE_TYPE fp32, fp64, int16, int32, int64, pred;
  OPERAND_VALUE_TYPE fround, sat, cmp, boolop, space, qualifier, ftz;
  OPERAND_VALUE_TYPE lit8, lit16, lit32, lit64, ulit8, ulit16, ulit32, ulit64; 
  OPERAND_VALUE_TYPE flit8, flit32, flit64;
  OPERAND_VALUE_TYPE addr16, addr32, addr64;

  OPERAND_USE_TYPE
	  predicate,	// a qualifying predicate
	  base,		// a base address (for memory insts)
	  offset,	// an offset added to a base (implies immed val)
	  postincr,	// a post increment applied to a base address
	  target,	// the target of a branch
	  storeval,	// value to be stored
	  opnd1,	// first/left operand of an alu operator
	  opnd2,	// second/right operand of an alu operator
	  maddend;	// addend/subtrahend operand of a madd

  ISA_Operands_Begin("nvisa");

  /* Create the register operand types:
   */
  fp32  = ISA_Reg_Opnd_Type_Create("fp32", ISA_REGISTER_CLASS_float,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   32, SIGNED, INVALID);
  fp64  = ISA_Reg_Opnd_Type_Create("fp64", ISA_REGISTER_CLASS_float64,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   64, SIGNED, INVALID);
  int16 = ISA_Reg_Opnd_Type_Create("int16", ISA_REGISTER_CLASS_integer16,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   16, SIGNED, INVALID);
  int32 = ISA_Reg_Opnd_Type_Create("int32", ISA_REGISTER_CLASS_integer,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   32, SIGNED, INVALID);
  int64 = ISA_Reg_Opnd_Type_Create("int64", ISA_REGISTER_CLASS_integer64,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   64, SIGNED, INVALID);
  pred = ISA_Reg_Opnd_Type_Create("pred", ISA_REGISTER_CLASS_predicate,
				   ISA_REGISTER_SUBCLASS_UNDEFINED,
				   1, UNSIGNED, INVALID);

  /* Create the enum operand types:
   */
  fround = ISA_Enum_Opnd_Type_Create("fround", 8, UNSIGNED, EC_fround);
  sat = ISA_Enum_Opnd_Type_Create("sat", 8, UNSIGNED, EC_sat);
  cmp = ISA_Enum_Opnd_Type_Create("cmp", 8, UNSIGNED, EC_cmp);
  boolop = ISA_Enum_Opnd_Type_Create("boolop", 8, UNSIGNED, EC_boolop);
  space = ISA_Enum_Opnd_Type_Create("space", 8, UNSIGNED, EC_space);
  qualifier = ISA_Enum_Opnd_Type_Create("qualifier", 8, UNSIGNED, EC_qualifier);
  ftz = ISA_Enum_Opnd_Type_Create("ftz", 8, UNSIGNED, EC_ftz);  

  /* Create the literal operand types:
   */
  lit8   = ISA_Lit_Opnd_Type_Create("lit8",   8, SIGNED, LC_i8);
  lit16  = ISA_Lit_Opnd_Type_Create("lit16", 16, SIGNED, LC_i16);
  lit32  = ISA_Lit_Opnd_Type_Create("lit32", 32, SIGNED, LC_i32);
  lit64  = ISA_Lit_Opnd_Type_Create("lit64", 64, SIGNED, LC_i64);
  ulit8  = ISA_Lit_Opnd_Type_Create("ulit8",   8, UNSIGNED, LC_u8);
  ulit16 = ISA_Lit_Opnd_Type_Create("ulit16", 16, UNSIGNED, LC_u16);
  ulit32 = ISA_Lit_Opnd_Type_Create("ulit32", 32, UNSIGNED, LC_u32);
  ulit64 = ISA_Lit_Opnd_Type_Create("ulit64", 64, UNSIGNED, LC_u64);
  flit32 = ISA_Lit_Opnd_Type_Create("flit32", 32, UNSIGNED, LC_f32);
  flit64 = ISA_Lit_Opnd_Type_Create("flit64", 64, UNSIGNED, LC_f64);
  addr16 = ISA_Lit_Opnd_Type_Create("addr16", 16, UNSIGNED, LC_u16);
  addr32 = ISA_Lit_Opnd_Type_Create("addr32", 32, UNSIGNED, LC_u32);
  addr64 = ISA_Lit_Opnd_Type_Create("addr64", 64, UNSIGNED, LC_u64);

  /* Create the operand uses:
   */
  predicate  = Create_Operand_Use("predicate");
  offset     = Create_Operand_Use("offset");
  base       = Create_Operand_Use("base");
  postincr   = Create_Operand_Use("postincr");
  target     = Create_Operand_Use("target");
  storeval   = Create_Operand_Use("storeval");
  opnd1      = Create_Operand_Use("opnd1");
  opnd2      = Create_Operand_Use("opnd2");
  maddend    = Create_Operand_Use("maddend");

  // Don't define predicate for now, except on bra;
  // may have to add later if we ever decide to predicate.
  // also don't bother with rounding/saturation modes
  // that we won't be setting.

/* =====  ===== */
  Instruction_Group("O_name",
	TOP_exit,
	TOP_ret,
	TOP_nop, TOP_noop,
	TOP_membar_gl,
	TOP_membar_cta,
	TOP_membar_sys,
	TOP_trap, TOP_brkpt,
	TOP_asm,
	TOP_begin_pregtn, TOP_end_pregtn,
	TOP_UNDEFINED);

  Instruction_Group("O_unary16",
	TOP_abs_s8, TOP_abs_u8,
	TOP_not_b8, TOP_cnot_b8,
	TOP_neg_s8, TOP_neg_u8,
	TOP_abs_s16, TOP_abs_u16, 
	TOP_not_b16, TOP_cnot_b16,
	TOP_neg_s16, TOP_neg_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);

  Instruction_Group("O_unary32",
	TOP_abs_s32, TOP_abs_u32, 
	TOP_not_b32, TOP_cnot_b32,
	TOP_neg_s32, TOP_neg_u32,
	TOP_brev_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);

  Instruction_Group("O_unary64",
	TOP_abs_s64, TOP_abs_u64, 
	TOP_not_b64, TOP_cnot_b64,
	TOP_neg_s64, TOP_neg_u64,
	TOP_brev_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);

  Instruction_Group("O_unaryf32ftz",
	TOP_abs_ftz_f32,
	TOP_neg_ftz_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, ftz);
  Operand(1, fp32, opnd1);

  Instruction_Group("O_unaryf32froundftz",
	TOP_rcp_fround_ftz_f32, TOP_sqrt_fround_ftz_f32, TOP_rsqrt_fround_ftz_f32,
	TOP_sin_fround_ftz_f32, TOP_cos_fround_ftz_f32, 
	TOP_lg2_fround_ftz_f32, TOP_ex2_fround_ftz_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, fp32, opnd1);

  Instruction_Group("O_unaryf64",
	TOP_abs_f64, TOP_rcp_f64, TOP_sqrt_f64,
	TOP_sin_f64, TOP_cos_f64, 
	TOP_lg2_f64, TOP_ex2_f64,
	TOP_neg_f64,
	TOP_rcp_rn_f64, TOP_sqrt_rn_f64, 
	TOP_rcp_rz_f64,
	TOP_rcp_rm_f64, TOP_rcp_rp_f64,
	TOP_sqrt_rz_f64,
	TOP_sqrt_rm_f64, TOP_sqrt_rp_f64,
        TOP_rcp_approx_ftz_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);

  Instruction_Group("O_unaryf64fround",
	TOP_rsqrt_fround_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, fp64, opnd1);

  Instruction_Group("O_binary16",
	TOP_div_s8, TOP_div_u8,
	TOP_rem_s8, TOP_rem_u8,
	TOP_min_s8, TOP_min_u8,
	TOP_max_s8, TOP_max_u8,
	TOP_and_b8,
	TOP_or_b8,
	TOP_xor_b8,
	TOP_shl_b8,
	TOP_shr_s8, TOP_shr_u8,
	TOP_div_s16, TOP_div_u16, 
	TOP_rem_s16, TOP_rem_u16,
	TOP_min_s16, TOP_min_u16,
	TOP_max_s16, TOP_max_u16,
	TOP_and_b16,
	TOP_or_b16,
	TOP_xor_b16,
	TOP_shl_b16,
	TOP_shr_s16, TOP_shr_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binary32",
	TOP_div_s32, TOP_div_u32, 
	TOP_rem_s32, TOP_rem_u32,
	TOP_min_s32, TOP_min_u32,
	TOP_max_s32, TOP_max_u32,
	TOP_and_b32,
	TOP_or_b32,
	TOP_xor_b32,
	TOP_shl_b32,
	TOP_shr_s32, TOP_shr_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binary64",
	TOP_div_s64, TOP_div_u64, 
	TOP_rem_s64, TOP_rem_u64,
	TOP_min_s64, TOP_min_u64,
	TOP_max_s64, TOP_max_u64,
	TOP_and_b64,
	TOP_or_b64,
	TOP_xor_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);

  /* use 32bit shift amount */
  Instruction_Group("O_binary64shift",
	TOP_shl_b64,
	TOP_shr_s64, TOP_shr_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binaryf32ftz",
	TOP_min_ftz_f32, TOP_max_ftz_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, ftz);
  Operand(1, fp32, opnd1);
  Operand(2, fp32, opnd2);

  Instruction_Group("O_binaryf32froundftz",
	TOP_div_fround_ftz_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_binaryf32froundftzsat",
        TOP_add_fround_ftz_sat_f32,
        TOP_sub_fround_ftz_sat_f32,
        TOP_mul_fround_ftz_sat_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);
  Operand(4, fp32, opnd2);

  Instruction_Group("O_binaryf64",
	TOP_div_f64, TOP_min_f64, TOP_max_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);
  Operand(1, fp64, opnd2);

  Instruction_Group("O_unaryp",
	TOP_not_pred, TOP_mov_pred,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, pred, opnd1);

  Instruction_Group("O_binaryp",
	TOP_and_pred, TOP_or_pred, TOP_xor_pred,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, pred, opnd1);
  Operand(1, pred, opnd2);

  Instruction_Group("O_set16",
	TOP_set_cmp_u32_s8, TOP_set_cmp_u32_u8,
	TOP_set_cmp_u32_s16, TOP_set_cmp_u32_u16,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, int16, opnd1);
  Operand(2, int16, opnd2);

  Instruction_Group("O_set32",
	TOP_set_cmp_u32_s32, TOP_set_cmp_u32_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, int32, opnd1);
  Operand(2, int32, opnd2);

  Instruction_Group("O_set64",
	TOP_set_cmp_u32_s64, TOP_set_cmp_u32_u64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, int64, opnd1);
  Operand(2, int64, opnd2);

  Instruction_Group("O_setf32",
	TOP_set_cmp_ftz_u32_f32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, ftz);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_setf64",
	TOP_set_cmp_u32_f64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, fp64, opnd1);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_set1616",
	TOP_set_cmp_u16_s8, TOP_set_cmp_u16_u8,
	TOP_set_cmp_u16_s16, TOP_set_cmp_u16_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, int16, opnd1);
  Operand(2, int16, opnd2);

  Instruction_Group("O_set1632",
	TOP_set_cmp_u16_s32, TOP_set_cmp_u16_u32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, int32, opnd1);
  Operand(2, int32, opnd2);

  Instruction_Group("O_set1664",
	TOP_set_cmp_u16_s64, TOP_set_cmp_u16_u64,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, int64, opnd1);
  Operand(2, int64, opnd2);

  Instruction_Group("O_set16f32",
	TOP_set_cmp_ftz_u16_f32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, ftz);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_set16f64",
	TOP_set_cmp_u16_f64,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, fp64, opnd1);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_binary8lit",
	TOP_shl_b8_lit,
	TOP_shr_s8_lit,
	TOP_add_s8_lit,
	TOP_sub_s8_lit,
	TOP_mul_lo_s8_lit,
	TOP_div_s8_lit,
	TOP_rem_s8_lit,
	TOP_and_b8_lit,
	TOP_or_b8_lit,
	TOP_xor_b8_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, lit8, opnd2);

  Instruction_Group("O_binary8ulit",
	TOP_shr_u8_lit,
	TOP_add_u8_lit,
	TOP_sub_u8_lit,
	TOP_mul_lo_u8_lit,
	TOP_div_u8_lit,
	TOP_rem_u8_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, ulit8, opnd2);

  Instruction_Group("O_binary16lit",
	TOP_shl_b16_lit,
	TOP_shr_s16_lit,
	TOP_add_s16_lit,
	TOP_sub_s16_lit,
	TOP_mul_lo_s16_lit,
	TOP_div_s16_lit,
	TOP_rem_s16_lit,
	TOP_and_b16_lit,
	TOP_or_b16_lit,
	TOP_xor_b16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, lit16, opnd2);

  Instruction_Group("O_binary16ulit",
	TOP_shr_u16_lit,
	TOP_add_u16_lit,
	TOP_sub_u16_lit,
	TOP_mul_lo_u16_lit,
	TOP_div_u16_lit,
	TOP_rem_u16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, ulit16, opnd2);

  Instruction_Group("O_binary32lit",
	TOP_shl_b32_lit,
	TOP_shr_s32_lit,
	TOP_add_s32_lit,
	TOP_sub_s32_lit,
	TOP_mul_lo_s32_lit,
	TOP_mul24_lo_s32_lit,
	TOP_div_s32_lit,
	TOP_rem_s32_lit,
	TOP_and_b32_lit,
	TOP_or_b32_lit,
	TOP_xor_b32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, lit32, opnd2);

  Instruction_Group("O_binary32ulit",
	TOP_shr_u32_lit,
	TOP_add_u32_lit,
	TOP_sub_u32_lit,
	TOP_mul_lo_u32_lit,
	TOP_mul24_lo_u32_lit,
	TOP_div_u32_lit,
	TOP_rem_u32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, ulit32, opnd2);

  Instruction_Group("O_binary64lit",
	TOP_add_s64_lit,
	TOP_sub_s64_lit,
	TOP_mul_lo_s64_lit,
	TOP_div_s64_lit,
	TOP_rem_s64_lit,
	TOP_and_b64_lit,
	TOP_or_b64_lit,
	TOP_xor_b64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, lit64, opnd2);

  Instruction_Group("O_binary64ulit",
	TOP_add_u64_lit,
	TOP_sub_u64_lit,
	TOP_mul_lo_u64_lit,
	TOP_div_u64_lit,
	TOP_rem_u64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, ulit64, opnd2);

  Instruction_Group("O_binary64shiftulit",
	TOP_shl_b64_lit,
	TOP_shr_s64_lit,
	TOP_shr_u64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, ulit32, opnd2);

  Instruction_Group("O_binary8lit1",
	TOP_shl_b8_lit1,
	TOP_shr_s8_lit1,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, lit8, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binary8ulit1",
	TOP_shr_u8_lit1,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, ulit8, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binary16lit1",
	TOP_shl_b16_lit1,
	TOP_shr_s16_lit1,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, lit16, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binary16ulit1",
	TOP_shr_u16_lit1,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, ulit16, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binary32lit1",
	TOP_shl_b32_lit1,
	TOP_shr_s32_lit1,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, lit32, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binary32ulit1",
	TOP_shr_u32_lit1,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, ulit32, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binary64lit1",
	TOP_shl_b64_lit1,
	TOP_shr_s64_lit1,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, lit64, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binary64ulit1",
	TOP_shr_u64_lit1,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, ulit64, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binary16p",
	TOP_setp_cmp_s8, TOP_setp_cmp_u8,
	TOP_setp_cmp_s16, TOP_setp_cmp_u16,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, int16, opnd1);
  Operand(2, int16, opnd2);

  Instruction_Group("O_binary32p",
	TOP_setp_cmp_s32, TOP_setp_cmp_u32,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, int32, opnd1);
  Operand(2, int32, opnd2);

  Instruction_Group("O_binary64p",
	TOP_setp_cmp_s64, TOP_setp_cmp_u64,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, int64, opnd1);
  Operand(2, int64, opnd2);

  Instruction_Group("O_binaryf32p",
	TOP_setp_cmp_ftz_f32,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, ftz);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_binaryf64p",
	TOP_setp_cmp_f64,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, fp64, opnd1);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_setp32p",
	TOP_setp_eq_u32_p, TOP_setp_ne_u32_p,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, pred, predicate);
  Operand(1, int32, opnd1);
  Operand(2, int32, opnd2);

  Instruction_Group("O_set1616cmpblit",
	TOP_set_cmp_boolop_u16_s16_lit,
	TOP_set_cmp_boolop_u16_u16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, boolop);
  Operand(2, lit16, opnd1);
  Operand(3, lit16, opnd2);
  Operand(4, pred, predicate);

  Instruction_Group("O_set1632cmpblit",
	TOP_set_cmp_boolop_u16_s32_lit,
	TOP_set_cmp_boolop_u16_u32_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, cmp);
  Operand(1, boolop);
  Operand(2, lit32, opnd1);
  Operand(3, lit32, opnd2);
  Operand(4, pred, predicate);

  Instruction_Group("O_set3216cmpblit",
	TOP_set_cmp_boolop_u32_s16_lit,
	TOP_set_cmp_boolop_u32_u16_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, boolop);
  Operand(2, lit16, opnd1);
  Operand(3, lit16, opnd2);
  Operand(4, pred, predicate);

  Instruction_Group("O_set3232cmpblit",
	TOP_set_cmp_boolop_u32_s32_lit,
	TOP_set_cmp_boolop_u32_u32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, cmp);
  Operand(1, boolop);
  Operand(2, lit32, opnd1);
  Operand(3, lit32, opnd2);
  Operand(4, pred, predicate);

  Instruction_Group("O_setp32cmpblit",
	TOP_setp_cmp_boolop_u32_lit,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, cmp);
  Operand(1, boolop);
  Operand(2, lit32, opnd1);
  Operand(3, lit32, opnd2);
  Operand(4, pred, predicate);

  Instruction_Group("O_binaryr16",
	TOP_add_s8, TOP_add_u8,
	TOP_sub_s8, TOP_sub_u8,
	TOP_mul_lo_s8, TOP_mul_lo_u8,
	TOP_mul_hi_s8, TOP_mul_hi_u8,
	TOP_add_s16, TOP_add_u16,
	TOP_sub_s16, TOP_sub_u16,
	TOP_mul_lo_s16, TOP_mul_lo_u16,
	TOP_mul_hi_s16, TOP_mul_hi_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binaryr32",
	TOP_add_s32, TOP_add_u32,
	TOP_sub_s32, TOP_sub_u32,
	TOP_mul_lo_s32, TOP_mul_lo_u32,
	TOP_mul_hi_s32, TOP_mul_hi_u32,
	TOP_mul24_lo_s32, TOP_mul24_lo_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binaryr32p",
    TOP_sub_s32_p, TOP_sub_s32_np, 
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, pred, predicate);
  
  Instruction_Group("O_binaryr64p",
	TOP_sub_s64_p, TOP_sub_s64_np, 
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);
  Operand(2, pred, predicate);
  
  Instruction_Group("O_binaryr64",
	TOP_add_s64, TOP_add_u64,
	TOP_sub_s64, TOP_sub_u64,
	TOP_mul_lo_s64, TOP_mul_lo_u64,
	TOP_mul_hi_s64, TOP_mul_hi_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);

  Instruction_Group("O_binaryrf64",
	TOP_add_f64, TOP_sub_f64, TOP_mul_f64,
	TOP_add_rn_f64, TOP_add_rz_f64,
	TOP_add_rm_f64, TOP_add_rp_f64,
	TOP_mul_rn_f64, TOP_mul_rz_f64,
	TOP_mul_rm_f64, TOP_mul_rp_f64,
	TOP_div_rn_f64, TOP_div_rz_f64,
	TOP_div_rm_f64, TOP_div_rp_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);
  Operand(1, fp64, opnd2);

  Instruction_Group("O_binaryw16",
	TOP_mul_wide_s16, TOP_mul_wide_u16,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);

  Instruction_Group("O_binaryw32",
	TOP_mul_wide_s32, TOP_mul_wide_u32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);

  Instruction_Group("O_binaryw16lit",
	TOP_mul_wide_s16_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int16, opnd1);
  Operand(1, lit16, opnd2);

  Instruction_Group("O_binaryw16ulit",
	TOP_mul_wide_u16_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int16, opnd1);
  Operand(1, ulit16, opnd2);

  Instruction_Group("O_binaryw32lit",
	TOP_mul_wide_s32_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int32, opnd1);
  Operand(1, lit32, opnd2);

  Instruction_Group("O_binaryw32ulit",
	TOP_mul_wide_u32_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int32, opnd1);
  Operand(1, ulit32, opnd2);

  Instruction_Group("O_triple16",
	TOP_mad_lo_s8, TOP_mad_lo_u8,
	TOP_mad_hi_s8, TOP_mad_hi_u8,
	TOP_mad_lo_s16, TOP_mad_lo_u16,
	TOP_mad_hi_s16, TOP_mad_hi_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);
  Operand(2, int16, maddend);

  Instruction_Group("O_triple32",
	TOP_mad_lo_s32, TOP_mad_lo_u32,
	TOP_mad24_lo_s32, TOP_mad24_lo_u32,
	TOP_mad_hi_s32, TOP_mad_hi_u32,
	TOP_sad_s32, TOP_sad_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, int32, maddend);

  Instruction_Group("O_triple64",
	TOP_mad_lo_s64, TOP_mad_lo_u64,
	TOP_mad_hi_s64, TOP_mad_hi_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);
  Operand(2, int64, maddend);

  Instruction_Group("O_triplef32",
	TOP_mad_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fp32, opnd1);
  Operand(1, fp32, opnd2);
  Operand(2, fp32, maddend);

  Instruction_Group("O_fma32",
	TOP_fma_fround_ftz_sat_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);
  Operand(4, fp32, opnd2);
  Operand(5, fp32, maddend);

  Instruction_Group("O_fma64",
	TOP_fma_fround_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, fp64, opnd1);
  Operand(2, fp64, opnd2);
  Operand(3, fp64, maddend);

  Instruction_Group("O_triplef64",
	TOP_mad_f64,
	TOP_mad_rn_f64,
	TOP_mad_rz_f64,
	TOP_mad_rm_f64,
	TOP_mad_rp_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);
  Operand(1, fp64, opnd2);
  Operand(2, fp64, maddend);

  Instruction_Group("O_triplew16",
	TOP_mad_wide_s16, TOP_mad_wide_u16,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);
  Operand(2, int32, maddend);

  Instruction_Group("O_triplew32",
	TOP_mad_wide_s32, TOP_mad_wide_u32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, int64, maddend);

  Instruction_Group("O_slct16",
	TOP_slct_s8_s32, TOP_slct_u8_s32,
	TOP_slct_s16_s32, TOP_slct_u16_s32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);
  Operand(2, int32);

  Instruction_Group("O_slct32",
	TOP_slct_s32_s32, TOP_slct_u32_s32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, int32);

  Instruction_Group("O_slct64",
	TOP_slct_s64_s32, TOP_slct_u64_s32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);
  Operand(2, int32);

  Instruction_Group("O_slctf32",
	TOP_slct_f32_s32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fp32, opnd1);
  Operand(1, fp32, opnd2);
  Operand(2, int32);

  Instruction_Group("O_slctf64",
	TOP_slct_f64_s32,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);
  Operand(1, fp64, opnd2);
  Operand(2, int32);

  Instruction_Group("O_slct16f",
	TOP_slct_ftz_s8_f32, TOP_slct_ftz_u8_f32,
	TOP_slct_ftz_s16_f32, TOP_slct_ftz_u16_f32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, ftz);
  Operand(1, int16, opnd1);
  Operand(2, int16, opnd2);
  Operand(3, fp32);

  Instruction_Group("O_slct32f",
	TOP_slct_ftz_s32_f32, TOP_slct_ftz_u32_f32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, ftz);
  Operand(1, int32, opnd1);
  Operand(2, int32, opnd2);
  Operand(3, fp32);

  Instruction_Group("O_slct64f",
	TOP_slct_ftz_s64_f32, TOP_slct_ftz_u64_f32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, ftz);
  Operand(1, int64, opnd1);
  Operand(2, int64, opnd2);
  Operand(3, fp32);

  Instruction_Group("O_slctf32f",
	TOP_slct_ftz_f32_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, ftz);
  Operand(1, fp32, opnd1);
  Operand(2, fp32, opnd2);
  Operand(3, fp32);

  Instruction_Group("O_slctf64f",
	TOP_slct_ftz_f64_f32,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, ftz);
  Operand(1, fp64, opnd1);
  Operand(2, fp64, opnd2);
  Operand(3, fp32);

  Instruction_Group("O_selp16",
	TOP_selp_s8, TOP_selp_u8,
	TOP_selp_s16, TOP_selp_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);
  Operand(1, int16, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selp32",
	TOP_selp_s32, TOP_selp_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selp64",
	TOP_selp_s64, TOP_selp_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);
  Operand(1, int64, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selpf32",
	TOP_selp_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fp32, opnd1);
  Operand(1, fp32, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selpf64",
	TOP_selp_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64, opnd1);
  Operand(1, fp64, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selp16l",
	TOP_selp_s8_lit, TOP_selp_u8_lit,
	TOP_selp_s16_lit, TOP_selp_u16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, lit16, opnd1);
  Operand(1, lit16, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selp32l",
	TOP_selp_s32_lit, TOP_selp_u32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, lit32, opnd1);
  Operand(1, lit32, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selp64l",
	TOP_selp_s64_lit, TOP_selp_u64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, lit64, opnd1);
  Operand(1, lit64, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selpf32l",
	TOP_selp_f32_lit,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, flit32, opnd1);
  Operand(1, flit32, opnd2);
  Operand(2, pred);

  Instruction_Group("O_selpf64l",
	TOP_selp_f64_lit,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, flit64, opnd1);
  Operand(1, flit64, opnd2);
  Operand(2, pred);

  Instruction_Group("O_mov16",
	TOP_mov_s8, TOP_mov_u8,
	TOP_mov_s16, TOP_mov_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16);

  Instruction_Group("O_mov32",
	TOP_mov_s32, TOP_mov_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32);

  Instruction_Group("O_mov64",
	TOP_mov_s64, TOP_mov_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64);

  Instruction_Group("O_movf32",
	TOP_mov_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fp32);

  Instruction_Group("O_movf64",
	TOP_mov_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fp64);

  Instruction_Group("O_movb32i",
	TOP_mov_b32_f2i,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fp32);

  Instruction_Group("O_movb32f",
	TOP_mov_b32_i2f,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, int32);

  Instruction_Group("O_movb64i",
	TOP_mov_b64_f2i,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fp64);

  Instruction_Group("O_movb64f",
	TOP_mov_b64_i2f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, int64);

  Instruction_Group("O_movb64i2",
	TOP_mov_b64_f2i2,
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Operand(0, fp64);

  Instruction_Group("O_movb64f2",
	TOP_mov_b64_i22f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, int32);
  Operand(1, int32);

// What should be size of address?
// nv50 global takes 32bit reg, rest take 16bit reg+offset
// For future compatibility, have it always be 32-bit addr.

  Instruction_Group("O_ldi16",
	TOP_ld_qualifier_space_s8, TOP_ld_qualifier_space_u8, 
	TOP_ld_qualifier_space_s16, TOP_ld_qualifier_space_u16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);

  Instruction_Group("O_ldi32",
	TOP_ld_qualifier_space_s32, TOP_ld_qualifier_space_u32,
	TOP_ld_qualifier_space_s8_b32, TOP_ld_qualifier_space_u8_b32,
	TOP_ld_qualifier_space_s16_b32, TOP_ld_qualifier_space_u16_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);

  Instruction_Group("O_ldi64",
	TOP_ld_qualifier_space_s64, TOP_ld_qualifier_space_u64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);

  Instruction_Group("O_ldf32",
	TOP_ld_qualifier_space_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);

  Instruction_Group("O_ldf64",
	TOP_ld_qualifier_space_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);

  Instruction_Group("O_ldi16o",
	TOP_ld_qualifier_space_s8_o, TOP_ld_qualifier_space_u8_o, 
	TOP_ld_qualifier_space_s16_o, TOP_ld_qualifier_space_u16_o,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldi32o",
	TOP_ld_qualifier_space_s32_o, TOP_ld_qualifier_space_u32_o,
	TOP_ld_qualifier_space_s8_b32_o, TOP_ld_qualifier_space_u8_b32_o,
	TOP_ld_qualifier_space_s16_b32_o, TOP_ld_qualifier_space_u16_b32_o,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldi64o",
	TOP_ld_qualifier_space_s64_o, TOP_ld_qualifier_space_u64_o,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldf32o",
	TOP_ld_qualifier_space_f32_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldf64o",
	TOP_ld_qualifier_space_f64_o,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldi16r",
	TOP_ld_qualifier_space_s8_r, TOP_ld_qualifier_space_u8_r, 
	TOP_ld_qualifier_space_s16_r, TOP_ld_qualifier_space_u16_r,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldi32r",
	TOP_ld_qualifier_space_s32_r, TOP_ld_qualifier_space_u32_r,
	TOP_ld_qualifier_space_s8_b32_r, TOP_ld_qualifier_space_u8_b32_r,
	TOP_ld_qualifier_space_s16_b32_r, TOP_ld_qualifier_space_u16_b32_r,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldi64r",
	TOP_ld_qualifier_space_s64_r, TOP_ld_qualifier_space_u64_r,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldf32r",
	TOP_ld_qualifier_space_f32_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldf64r",
	TOP_ld_qualifier_space_f64_r,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei16",
	TOP_ld_qualifier_space_s8_a64, TOP_ld_qualifier_space_u8_a64, 
	TOP_ld_qualifier_space_s16_a64, TOP_ld_qualifier_space_u16_a64,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);

  Instruction_Group("O_ldei32",
	TOP_ld_qualifier_space_s32_a64, TOP_ld_qualifier_space_u32_a64,
	TOP_ld_qualifier_space_s8_b32_a64, TOP_ld_qualifier_space_u8_b32_a64,
	TOP_ld_qualifier_space_s16_b32_a64, TOP_ld_qualifier_space_u16_b32_a64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);

  Instruction_Group("O_ldei64",
	TOP_ld_qualifier_space_s64_a64, TOP_ld_qualifier_space_u64_a64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);

  Instruction_Group("O_ldef32",
	TOP_ld_qualifier_space_f32_a64,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);

  Instruction_Group("O_ldef64",
	TOP_ld_qualifier_space_f64_a64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);

  Instruction_Group("O_ldei16o",
	TOP_ld_qualifier_space_s8_a64_o, TOP_ld_qualifier_space_u8_a64_o, 
	TOP_ld_qualifier_space_s16_a64_o, TOP_ld_qualifier_space_u16_a64_o,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei32o",
	TOP_ld_qualifier_space_s32_a64_o, TOP_ld_qualifier_space_u32_a64_o,
	TOP_ld_qualifier_space_s8_b32_a64_o, TOP_ld_qualifier_space_u8_b32_a64_o, 
	TOP_ld_qualifier_space_s16_b32_a64_o, TOP_ld_qualifier_space_u16_b32_a64_o,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei64o",
	TOP_ld_qualifier_space_s64_a64_o, TOP_ld_qualifier_space_u64_a64_o,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldef32o",
	TOP_ld_qualifier_space_f32_a64_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldef64o",
	TOP_ld_qualifier_space_f64_a64_o,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei16r",
	TOP_ld_qualifier_space_s8_a64_r, TOP_ld_qualifier_space_u8_a64_r, 
	TOP_ld_qualifier_space_s16_a64_r, TOP_ld_qualifier_space_u16_a64_r,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei32r",
	TOP_ld_qualifier_space_s32_a64_r, TOP_ld_qualifier_space_u32_a64_r,
	TOP_ld_qualifier_space_s8_b32_a64_r, TOP_ld_qualifier_space_u8_b32_a64_r, 
	TOP_ld_qualifier_space_s16_b32_a64_r, TOP_ld_qualifier_space_u16_b32_a64_r,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldei64r",
	TOP_ld_qualifier_space_s64_a64_r, TOP_ld_qualifier_space_u64_a64_r,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldef32r",
	TOP_ld_qualifier_space_f32_a64_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldef64r",
	TOP_ld_qualifier_space_f64_a64_r,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  // vector loads _r suffix is for *(r+o) addresses
  Instruction_Group("O_ldv2i16r",
	TOP_ld_qualifier_space_v2_s8_r, TOP_ld_qualifier_space_v2_u8_r, 
	TOP_ld_qualifier_space_v2_s16_r, TOP_ld_qualifier_space_v2_u16_r, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4i16r",
	TOP_ld_qualifier_space_v4_s8_r, TOP_ld_qualifier_space_v4_u8_r, 
	TOP_ld_qualifier_space_v4_s16_r, TOP_ld_qualifier_space_v4_u16_r, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Result(2, int16);
  Result(3, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2i32r",
	TOP_ld_qualifier_space_v2_s8_b32_r, TOP_ld_qualifier_space_v2_u8_b32_r, 
	TOP_ld_qualifier_space_v2_s16_b32_r, TOP_ld_qualifier_space_v2_u16_b32_r,
	TOP_ld_qualifier_space_v2_s32_r, TOP_ld_qualifier_space_v2_u32_r, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4i32r",
	TOP_ld_qualifier_space_v4_s8_b32_r, TOP_ld_qualifier_space_v4_u8_b32_r, 
	TOP_ld_qualifier_space_v4_s16_b32_r, TOP_ld_qualifier_space_v4_u16_b32_r,
	TOP_ld_qualifier_space_v4_s32_r, TOP_ld_qualifier_space_v4_u32_r, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Result(2, int32);
  Result(3, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2i64r",
	TOP_ld_qualifier_space_v2_s64_r, TOP_ld_qualifier_space_v2_u64_r, 
	TOP_UNDEFINED);
  Result(0, int64);
  Result(1, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2f32r",
	TOP_ld_qualifier_space_v2_f32_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4f32r",
	TOP_ld_qualifier_space_v4_f32_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Result(2, fp32);
  Result(3, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2f64r",
	TOP_ld_qualifier_space_v2_f64_r,
	TOP_UNDEFINED);
  Result(0, fp64);
  Result(1, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i16r",
	TOP_ld_qualifier_space_v2_s8_a64_r, TOP_ld_qualifier_space_v2_u8_a64_r, 
	TOP_ld_qualifier_space_v2_s16_a64_r, TOP_ld_qualifier_space_v2_u16_a64_r, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4i16r",
	TOP_ld_qualifier_space_v4_s8_a64_r, TOP_ld_qualifier_space_v4_u8_a64_r, 
	TOP_ld_qualifier_space_v4_s16_a64_r, TOP_ld_qualifier_space_v4_u16_a64_r, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Result(2, int16);
  Result(3, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i32r",
	TOP_ld_qualifier_space_v2_s8_b32_a64_r, TOP_ld_qualifier_space_v2_u8_b32_a64_r, 
	TOP_ld_qualifier_space_v2_s16_b32_a64_r, TOP_ld_qualifier_space_v2_u16_b32_a64_r,
	TOP_ld_qualifier_space_v2_s32_a64_r, TOP_ld_qualifier_space_v2_u32_a64_r, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4i32r",
	TOP_ld_qualifier_space_v4_s8_b32_a64_r, TOP_ld_qualifier_space_v4_u8_b32_a64_r, 
	TOP_ld_qualifier_space_v4_s16_b32_a64_r, TOP_ld_qualifier_space_v4_u16_b32_a64_r,
	TOP_ld_qualifier_space_v4_s32_a64_r, TOP_ld_qualifier_space_v4_u32_a64_r, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Result(2, int32);
  Result(3, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i64r",
	TOP_ld_qualifier_space_v2_s64_a64_r, TOP_ld_qualifier_space_v2_u64_a64_r, 
	TOP_UNDEFINED);
  Result(0, int64);
  Result(1, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2f32r",
	TOP_ld_qualifier_space_v2_f32_a64_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4f32r",
	TOP_ld_qualifier_space_v4_f32_a64_r,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Result(2, fp32);
  Result(3, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2f64r",
	TOP_ld_qualifier_space_v2_f64_a64_r,
	TOP_UNDEFINED);
  Result(0, fp64);
  Result(1, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2i16o",
	TOP_ld_qualifier_space_v2_s8_o, TOP_ld_qualifier_space_v2_u8_o, 
	TOP_ld_qualifier_space_v2_s16_o, TOP_ld_qualifier_space_v2_u16_o, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4i16o",
	TOP_ld_qualifier_space_v4_s8_o, TOP_ld_qualifier_space_v4_u8_o, 
	TOP_ld_qualifier_space_v4_s16_o, TOP_ld_qualifier_space_v4_u16_o, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Result(2, int16);
  Result(3, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2i32o",
	TOP_ld_qualifier_space_v2_s8_b32_o, TOP_ld_qualifier_space_v2_u8_b32_o, 
	TOP_ld_qualifier_space_v2_s16_b32_o, TOP_ld_qualifier_space_v2_u16_b32_o,
	TOP_ld_qualifier_space_v2_s32_o, TOP_ld_qualifier_space_v2_u32_o, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4i32o",
	TOP_ld_qualifier_space_v4_s8_b32_o, TOP_ld_qualifier_space_v4_u8_b32_o, 
	TOP_ld_qualifier_space_v4_s16_b32_o, TOP_ld_qualifier_space_v4_u16_b32_o,
	TOP_ld_qualifier_space_v4_s32_o, TOP_ld_qualifier_space_v4_u32_o, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Result(2, int32);
  Result(3, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2i64o",
	TOP_ld_qualifier_space_v2_s64_o, TOP_ld_qualifier_space_v2_u64_o, 
	TOP_UNDEFINED);
  Result(0, int64);
  Result(1, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2f32o",
	TOP_ld_qualifier_space_v2_f32_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv4f32o",
	TOP_ld_qualifier_space_v4_f32_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Result(2, fp32);
  Result(3, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldv2f64o",
	TOP_ld_qualifier_space_v2_f64_o,
	TOP_UNDEFINED);
  Result(0, fp64);
  Result(1, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i16o",
	TOP_ld_qualifier_space_v2_s8_a64_o, TOP_ld_qualifier_space_v2_u8_a64_o, 
	TOP_ld_qualifier_space_v2_s16_a64_o, TOP_ld_qualifier_space_v2_u16_a64_o, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4i16o",
	TOP_ld_qualifier_space_v4_s8_a64_o, TOP_ld_qualifier_space_v4_u8_a64_o, 
	TOP_ld_qualifier_space_v4_s16_a64_o, TOP_ld_qualifier_space_v4_u16_a64_o, 
	TOP_UNDEFINED);
  Result(0, int16);
  Result(1, int16);
  Result(2, int16);
  Result(3, int16);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i32o",
	TOP_ld_qualifier_space_v2_s8_b32_a64_o, TOP_ld_qualifier_space_v2_u8_b32_a64_o, 
	TOP_ld_qualifier_space_v2_s16_b32_a64_o, TOP_ld_qualifier_space_v2_u16_b32_a64_o,
	TOP_ld_qualifier_space_v2_s32_a64_o, TOP_ld_qualifier_space_v2_u32_a64_o, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4i32o",
	TOP_ld_qualifier_space_v4_s8_b32_a64_o, TOP_ld_qualifier_space_v4_u8_b32_a64_o, 
	TOP_ld_qualifier_space_v4_s16_b32_a64_o, TOP_ld_qualifier_space_v4_u16_b32_a64_o,
	TOP_ld_qualifier_space_v4_s32_a64_o, TOP_ld_qualifier_space_v4_u32_a64_o, 
	TOP_UNDEFINED);
  Result(0, int32);
  Result(1, int32);
  Result(2, int32);
  Result(3, int32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2i64o",
	TOP_ld_qualifier_space_v2_s64_a64_o, TOP_ld_qualifier_space_v2_u64_a64_o, 
	TOP_UNDEFINED);
  Result(0, int64);
  Result(1, int64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2f32o",
	TOP_ld_qualifier_space_v2_f32_a64_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev4f32o",
	TOP_ld_qualifier_space_v4_f32_a64_o,
	TOP_UNDEFINED);
  Result(0, fp32);
  Result(1, fp32);
  Result(2, fp32);
  Result(3, fp32);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_ldev2f64o",
	TOP_ld_qualifier_space_v2_f64_a64_o,
	TOP_UNDEFINED);
  Result(0, fp64);
  Result(1, fp64);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);

  Instruction_Group("O_sti16",
	TOP_st_qualifier_space_s8, TOP_st_qualifier_space_u8, 
	TOP_st_qualifier_space_s16, TOP_st_qualifier_space_u16,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, int16, storeval);

  Instruction_Group("O_sti32",
	TOP_st_qualifier_space_s32, TOP_st_qualifier_space_u32,
	TOP_st_qualifier_space_s8_b32, TOP_st_qualifier_space_u8_b32,
	TOP_st_qualifier_space_s16_b32, TOP_st_qualifier_space_u16_b32,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, int32, storeval);

  Instruction_Group("O_sti64",
	TOP_st_qualifier_space_s64, TOP_st_qualifier_space_u64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, int64, storeval);

  Instruction_Group("O_stf32",
	TOP_st_qualifier_space_f32,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, fp32, storeval);

  Instruction_Group("O_stf64",
	TOP_st_qualifier_space_f64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, fp64, storeval);

  Instruction_Group("O_sti16o",
	TOP_st_qualifier_space_s8_o, TOP_st_qualifier_space_u8_o, 
	TOP_st_qualifier_space_s16_o, TOP_st_qualifier_space_u16_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);

  Instruction_Group("O_sti32o",
	TOP_st_qualifier_space_s32_o, TOP_st_qualifier_space_u32_o,
	TOP_st_qualifier_space_s8_b32_o, TOP_st_qualifier_space_u8_b32_o,
	TOP_st_qualifier_space_s16_b32_o, TOP_st_qualifier_space_u16_b32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);

  Instruction_Group("O_sti64o",
	TOP_st_qualifier_space_s64_o, TOP_st_qualifier_space_u64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);

  Instruction_Group("O_stf32o",
	TOP_st_qualifier_space_f32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);

  Instruction_Group("O_stf64o",
	TOP_st_qualifier_space_f64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);

  Instruction_Group("O_sti16r",
	TOP_st_qualifier_space_s8_r, TOP_st_qualifier_space_u8_r, 
	TOP_st_qualifier_space_s16_r, TOP_st_qualifier_space_u16_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);

  Instruction_Group("O_sti32r",
	TOP_st_qualifier_space_s32_r, TOP_st_qualifier_space_u32_r,
	TOP_st_qualifier_space_s8_b32_r, TOP_st_qualifier_space_u8_b32_r,
	TOP_st_qualifier_space_s16_b32_r, TOP_st_qualifier_space_u16_b32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);

  Instruction_Group("O_sti64r",
	TOP_st_qualifier_space_s64_r, TOP_st_qualifier_space_u64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);

  Instruction_Group("O_stf32r",
	TOP_st_qualifier_space_f32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);

  Instruction_Group("O_stf64r",
	TOP_st_qualifier_space_f64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);

  Instruction_Group("O_stei16",
	TOP_st_qualifier_space_s8_a64, TOP_st_qualifier_space_u8_a64, 
	TOP_st_qualifier_space_s16_a64, TOP_st_qualifier_space_u16_a64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, int16, storeval);

  Instruction_Group("O_stei32",
	TOP_st_qualifier_space_s32_a64, TOP_st_qualifier_space_u32_a64,
	TOP_st_qualifier_space_s8_b32_a64, TOP_st_qualifier_space_u8_b32_a64, 
	TOP_st_qualifier_space_s16_b32_a64, TOP_st_qualifier_space_u16_b32_a64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, int32, storeval);

  Instruction_Group("O_stei64",
	TOP_st_qualifier_space_s64_a64, TOP_st_qualifier_space_u64_a64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, int64, storeval);

  Instruction_Group("O_stef32",
	TOP_st_qualifier_space_f32_a64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, fp32, storeval);

  Instruction_Group("O_stef64",
	TOP_st_qualifier_space_f64_a64,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, fp64, storeval);

  Instruction_Group("O_stei16o",
	TOP_st_qualifier_space_s8_a64_o, TOP_st_qualifier_space_u8_a64_o, 
	TOP_st_qualifier_space_s16_a64_o, TOP_st_qualifier_space_u16_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);

  Instruction_Group("O_stei32o",
	TOP_st_qualifier_space_s32_a64_o, TOP_st_qualifier_space_u32_a64_o,
	TOP_st_qualifier_space_s8_b32_a64_o, TOP_st_qualifier_space_u8_b32_a64_o, 
	TOP_st_qualifier_space_s16_b32_a64_o, TOP_st_qualifier_space_u16_b32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);

  Instruction_Group("O_stei64o",
	TOP_st_qualifier_space_s64_a64_o, TOP_st_qualifier_space_u64_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);

  Instruction_Group("O_stef32o",
	TOP_st_qualifier_space_f32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);

  Instruction_Group("O_stef64o",
	TOP_st_qualifier_space_f64_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);

  Instruction_Group("O_stei16r",
	TOP_st_qualifier_space_s8_a64_r, TOP_st_qualifier_space_u8_a64_r, 
	TOP_st_qualifier_space_s16_a64_r, TOP_st_qualifier_space_u16_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);

  Instruction_Group("O_stei32r",
	TOP_st_qualifier_space_s32_a64_r, TOP_st_qualifier_space_u32_a64_r,
	TOP_st_qualifier_space_s8_b32_a64_r, TOP_st_qualifier_space_u8_b32_a64_r, 
	TOP_st_qualifier_space_s16_b32_a64_r, TOP_st_qualifier_space_u16_b32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);

  Instruction_Group("O_stei64r",
	TOP_st_qualifier_space_s64_a64_r, TOP_st_qualifier_space_u64_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);

  Instruction_Group("O_stef32r",
	TOP_st_qualifier_space_f32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);

  Instruction_Group("O_stef64r",
	TOP_st_qualifier_space_f64_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);

  // vector stores
  Instruction_Group("O_stv2i16r",
	TOP_st_qualifier_space_v2_s8_r, TOP_st_qualifier_space_v2_u8_r, 
	TOP_st_qualifier_space_v2_s16_r, TOP_st_qualifier_space_v2_u16_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);

  Instruction_Group("O_stv4i16r",
	TOP_st_qualifier_space_v4_s8_r, TOP_st_qualifier_space_v4_u8_r, 
	TOP_st_qualifier_space_v4_s16_r, TOP_st_qualifier_space_v4_u16_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);
  Operand(6, int16, storeval);
  Operand(7, int16, storeval);

  Instruction_Group("O_stv2i32r",
	TOP_st_qualifier_space_v2_s8_b32_r, TOP_st_qualifier_space_v2_u8_b32_r, 
	TOP_st_qualifier_space_v2_s16_b32_r, TOP_st_qualifier_space_v2_u16_b32_r,
	TOP_st_qualifier_space_v2_s32_r, TOP_st_qualifier_space_v2_u32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);

  Instruction_Group("O_stv4i32r",
	TOP_st_qualifier_space_v4_s8_b32_r, TOP_st_qualifier_space_v4_u8_b32_r, 
	TOP_st_qualifier_space_v4_s16_b32_r, TOP_st_qualifier_space_v4_u16_b32_r,
	TOP_st_qualifier_space_v4_s32_r, TOP_st_qualifier_space_v4_u32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);
  Operand(6, int32, storeval);
  Operand(7, int32, storeval);

  Instruction_Group("O_stv2i64r",
	TOP_st_qualifier_space_v2_s64_r, TOP_st_qualifier_space_v2_u64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);
  Operand(5, int64, storeval);

  Instruction_Group("O_stv2f32r",
	TOP_st_qualifier_space_v2_f32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);

  Instruction_Group("O_stv4f32r",
	TOP_st_qualifier_space_v4_f32_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);
  Operand(6, fp32, storeval);
  Operand(7, fp32, storeval);

  Instruction_Group("O_stv2f64r",
	TOP_st_qualifier_space_v2_f64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int32, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);
  Operand(5, fp64, storeval);

  Instruction_Group("O_stev2i16r",
	TOP_st_qualifier_space_v2_s8_a64_r, TOP_st_qualifier_space_v2_u8_a64_r, 
	TOP_st_qualifier_space_v2_s16_a64_r, TOP_st_qualifier_space_v2_u16_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);

  Instruction_Group("O_stev4i16r",
	TOP_st_qualifier_space_v4_s8_a64_r, TOP_st_qualifier_space_v4_u8_a64_r, 
	TOP_st_qualifier_space_v4_s16_a64_r, TOP_st_qualifier_space_v4_u16_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);
  Operand(6, int16, storeval);
  Operand(7, int16, storeval);

  Instruction_Group("O_stev2i32r",
	TOP_st_qualifier_space_v2_s8_b32_a64_r, TOP_st_qualifier_space_v2_u8_b32_a64_r, 
	TOP_st_qualifier_space_v2_s16_b32_a64_r, TOP_st_qualifier_space_v2_u16_b32_a64_r,
	TOP_st_qualifier_space_v2_s32_a64_r, TOP_st_qualifier_space_v2_u32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);

  Instruction_Group("O_stev4i32r",
	TOP_st_qualifier_space_v4_s8_b32_a64_r, TOP_st_qualifier_space_v4_u8_b32_a64_r, 
	TOP_st_qualifier_space_v4_s16_b32_a64_r, TOP_st_qualifier_space_v4_u16_b32_a64_r,
	TOP_st_qualifier_space_v4_s32_a64_r, TOP_st_qualifier_space_v4_u32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);
  Operand(6, int32, storeval);
  Operand(7, int32, storeval);

  Instruction_Group("O_stev2i64r",
	TOP_st_qualifier_space_v2_s64_a64_r, TOP_st_qualifier_space_v2_u64_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);
  Operand(5, int64, storeval);

  Instruction_Group("O_stev2f32r",
	TOP_st_qualifier_space_v2_f32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);

  Instruction_Group("O_stev4f32r",
	TOP_st_qualifier_space_v4_f32_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);
  Operand(6, fp32, storeval);
  Operand(7, fp32, storeval);

  Instruction_Group("O_stev2f64r",
	TOP_st_qualifier_space_v2_f64_a64_r,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, int64, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);
  Operand(5, fp64, storeval);

  Instruction_Group("O_stv2i16o",
	TOP_st_qualifier_space_v2_s8_o, TOP_st_qualifier_space_v2_u8_o, 
	TOP_st_qualifier_space_v2_s16_o, TOP_st_qualifier_space_v2_u16_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);

  Instruction_Group("O_stv4i16o",
	TOP_st_qualifier_space_v4_s8_o, TOP_st_qualifier_space_v4_u8_o, 
	TOP_st_qualifier_space_v4_s16_o, TOP_st_qualifier_space_v4_u16_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);
  Operand(6, int16, storeval);
  Operand(7, int16, storeval);

  Instruction_Group("O_stv2i32o",
	TOP_st_qualifier_space_v2_s8_b32_o, TOP_st_qualifier_space_v2_u8_b32_o, 
	TOP_st_qualifier_space_v2_s16_b32_o, TOP_st_qualifier_space_v2_u16_b32_o,
	TOP_st_qualifier_space_v2_s32_o, TOP_st_qualifier_space_v2_u32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);

  Instruction_Group("O_stv4i32o",
	TOP_st_qualifier_space_v4_s8_b32_o, TOP_st_qualifier_space_v4_u8_b32_o, 
	TOP_st_qualifier_space_v4_s16_b32_o, TOP_st_qualifier_space_v4_u16_b32_o,
	TOP_st_qualifier_space_v4_s32_o, TOP_st_qualifier_space_v4_u32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);
  Operand(6, int32, storeval);
  Operand(7, int32, storeval);

  Instruction_Group("O_stv2i64o",
	TOP_st_qualifier_space_v2_s64_o, TOP_st_qualifier_space_v2_u64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);
  Operand(5, int64, storeval);

  Instruction_Group("O_stv2f32o",
	TOP_st_qualifier_space_v2_f32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);

  Instruction_Group("O_stv4f32o",
	TOP_st_qualifier_space_v4_f32_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);
  Operand(6, fp32, storeval);
  Operand(7, fp32, storeval);

  Instruction_Group("O_stv2f64o",
	TOP_st_qualifier_space_v2_f64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr32, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);
  Operand(5, fp64, storeval);

  Instruction_Group("O_stev2i16o",
	TOP_st_qualifier_space_v2_s8_a64_o, TOP_st_qualifier_space_v2_u8_a64_o, 
	TOP_st_qualifier_space_v2_s16_a64_o, TOP_st_qualifier_space_v2_u16_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);

  Instruction_Group("O_stev4i16o",
	TOP_st_qualifier_space_v4_s8_a64_o, TOP_st_qualifier_space_v4_u8_a64_o, 
	TOP_st_qualifier_space_v4_s16_a64_o, TOP_st_qualifier_space_v4_u16_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int16, storeval);
  Operand(5, int16, storeval);
  Operand(6, int16, storeval);
  Operand(7, int16, storeval);

  Instruction_Group("O_stev2i32o",
	TOP_st_qualifier_space_v2_s8_b32_a64_o, TOP_st_qualifier_space_v2_u8_b32_a64_o, 
	TOP_st_qualifier_space_v2_s16_b32_a64_o, TOP_st_qualifier_space_v2_u16_b32_a64_o,
	TOP_st_qualifier_space_v2_s32_a64_o, TOP_st_qualifier_space_v2_u32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);

  Instruction_Group("O_stev4i32o",
	TOP_st_qualifier_space_v4_s8_b32_a64_o, TOP_st_qualifier_space_v4_u8_b32_a64_o, 
	TOP_st_qualifier_space_v4_s16_b32_a64_o, TOP_st_qualifier_space_v4_u16_b32_a64_o,
	TOP_st_qualifier_space_v4_s32_a64_o, TOP_st_qualifier_space_v4_u32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int32, storeval);
  Operand(5, int32, storeval);
  Operand(6, int32, storeval);
  Operand(7, int32, storeval);

  Instruction_Group("O_stev2i64o",
	TOP_st_qualifier_space_v2_s64_a64_o, TOP_st_qualifier_space_v2_u64_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, int64, storeval);
  Operand(5, int64, storeval);

  Instruction_Group("O_stev2f32o",
	TOP_st_qualifier_space_v2_f32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);

  Instruction_Group("O_stev4f32o",
	TOP_st_qualifier_space_v4_f32_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, fp32, storeval);
  Operand(5, fp32, storeval);
  Operand(6, fp32, storeval);
  Operand(7, fp32, storeval);

  Instruction_Group("O_stev2f64o",
	TOP_st_qualifier_space_v2_f64_a64_o,
	TOP_UNDEFINED);
  Operand(0, qualifier);
  Operand(1, space);
  Operand(2, addr64, base);
  Operand(3, lit32, offset);
  Operand(4, fp64, storeval);
  Operand(5, fp64, storeval);

  Instruction_Group("O_movi8l",
	TOP_mov_s8_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, lit8, opnd1);

  Instruction_Group("O_movi8ul",
	TOP_mov_u8_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, ulit8, opnd1);

  Instruction_Group("O_movi16l",
	TOP_mov_s16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, lit16, opnd1);

  Instruction_Group("O_movi16ul",
	TOP_mov_u16_lit,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, ulit16, opnd1);

  Instruction_Group("O_movi32l",
	TOP_mov_s32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, lit32, opnd1);

  Instruction_Group("O_movi32ul",
	TOP_mov_u32_lit,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, ulit32, opnd1);

  Instruction_Group("O_movi64l",
	TOP_mov_s64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, lit64, opnd1);

  Instruction_Group("O_movi64ul",
	TOP_mov_u64_lit,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, ulit64, opnd1);

  Instruction_Group("O_movf32l",
	TOP_mov_f32_lit,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, flit32, opnd1);

  Instruction_Group("O_movf64l",
	TOP_mov_f64_lit,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, flit64, opnd1);

  Instruction_Group("O_movu32a",
	TOP_mov_u32_a,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, addr32, base);

  Instruction_Group("O_movu32ao",
	TOP_mov_u32_ao,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, addr32, base);
  Operand(1, lit32, offset);

  Instruction_Group("O_movu64a",
	TOP_mov_u64_a,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, addr64, base);

  Instruction_Group("O_movu64ao",
	TOP_mov_u64_ao,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, addr64, base);
  Operand(1, lit64, offset);

  Instruction_Group("O_cvt1616",
	TOP_cvt_s16_s8, TOP_cvt_s16_u8,
	TOP_cvt_u16_u8,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, int16, opnd1);

  Instruction_Group("O_cvt1616_sat",
	TOP_cvt_sat_s8_u8, TOP_cvt_sat_u8_s8,
	TOP_cvt_sat_s8_s16, TOP_cvt_sat_s8_u16,
	TOP_cvt_sat_u8_s16, TOP_cvt_sat_u8_u16,
	TOP_cvt_sat_u16_s8, 
	TOP_cvt_sat_s16_u16, TOP_cvt_sat_u16_s16,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, sat);
  Operand(1, int16, opnd1);

  Instruction_Group("O_cvt1632_sat",
	TOP_cvt_sat_s8_s32, TOP_cvt_sat_s8_u32,
	TOP_cvt_sat_u8_s32, TOP_cvt_sat_u8_u32,
	TOP_cvt_sat_s16_s32, TOP_cvt_sat_s16_u32,
	TOP_cvt_sat_u16_s32, TOP_cvt_sat_u16_u32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, sat);
  Operand(1, int32, opnd1);

  Instruction_Group("O_cvt1664_sat",
	TOP_cvt_sat_s8_s64, TOP_cvt_sat_s8_u64,
	TOP_cvt_sat_u8_s64, TOP_cvt_sat_u8_u64,
	TOP_cvt_sat_s16_s64, TOP_cvt_sat_s16_u64,
	TOP_cvt_sat_u16_s64, TOP_cvt_sat_u16_u64,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt3216",
	TOP_cvt_s32_s8, TOP_cvt_s32_u8,
	TOP_cvt_u32_u8,
	TOP_cvt_s32_s16, TOP_cvt_s32_u16,
	TOP_cvt_u32_u16,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int16, opnd1);

  Instruction_Group("O_cvt3216_sat",
	TOP_cvt_sat_u32_s8,
	TOP_cvt_sat_u32_s16,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, sat);
  Operand(1, int16, opnd1);

  Instruction_Group("O_clz32",
	TOP_clz_b32, TOP_popc_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);

  Instruction_Group("O_prmt",
	TOP_prmt_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, int32, opnd2);
  Operand(2, int32);

  Instruction_Group("O_cvt3232_sat",
	TOP_cvt_sat_s32_u32, TOP_cvt_sat_u32_s32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, sat);
  Operand(1, int32, opnd1);

  Instruction_Group("O_clz64",
	TOP_clz_b64, TOP_popc_b64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int64, opnd1);

  Instruction_Group("O_cvt3264_sat",
	TOP_cvt_sat_s32_s64, TOP_cvt_sat_s32_u64,
	TOP_cvt_sat_u32_s64, TOP_cvt_sat_u32_u64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt6416",
	TOP_cvt_s64_s8, TOP_cvt_s64_u8,
	TOP_cvt_u64_u8,
	TOP_cvt_s64_s16, TOP_cvt_s64_u16,
	TOP_cvt_u64_u16,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int16, opnd1);

  Instruction_Group("O_cvt6416_sat",
	TOP_cvt_sat_u64_s8,
	TOP_cvt_sat_u64_s16,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int16, opnd1);

  Instruction_Group("O_cvt6432",
	TOP_cvt_s64_s32, TOP_cvt_s64_u32,
	TOP_cvt_u64_u32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int32, opnd1);

  Instruction_Group("O_cvt6432_sat",
	TOP_cvt_sat_u64_s32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int32, opnd1);

  Instruction_Group("O_cvt6464_sat",
	TOP_cvt_sat_s64_u64, TOP_cvt_sat_u64_s64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt16f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s8_f32,
	TOP_cvt_fround_ftz_sat_u8_f32,
	TOP_cvt_fround_ftz_sat_s16_f32,
	TOP_cvt_fround_ftz_sat_u16_f32,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt16f32b32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s8_f32_b32,
	TOP_cvt_fround_ftz_sat_u8_f32_b32,
	TOP_cvt_fround_ftz_sat_s16_f32_b32,
	TOP_cvt_fround_ftz_sat_u16_f32_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt16f32b64_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s8_f32_b64,
	TOP_cvt_fround_ftz_sat_u8_f32_b64,
	TOP_cvt_fround_ftz_sat_s16_f32_b64,
	TOP_cvt_fround_ftz_sat_u16_f32_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt16f64_fround_sat",
	TOP_cvt_fround_sat_s8_f64,
	TOP_cvt_fround_sat_u8_f64,
	TOP_cvt_fround_sat_s16_f64,
	TOP_cvt_fround_sat_u16_f64,
	TOP_UNDEFINED);
  Result(0, int16);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvt16f64b32_fround_sat",
	TOP_cvt_fround_sat_s8_f64_b32,
	TOP_cvt_fround_sat_u8_f64_b32,
	TOP_cvt_fround_sat_s16_f64_b32,
	TOP_cvt_fround_sat_u16_f64_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvt16f64b64_fround_sat",
	TOP_cvt_fround_sat_s8_f64_b64,
	TOP_cvt_fround_sat_u8_f64_b64,
	TOP_cvt_fround_sat_s16_f64_b64,
	TOP_cvt_fround_sat_u16_f64_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvt32f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s32_f32,
	TOP_cvt_fround_ftz_sat_u32_f32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt32f32b64_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s32_f32_b64,
	TOP_cvt_fround_ftz_sat_u32_f32_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt32f64_fround_sat",
	TOP_cvt_fround_sat_s32_f64,
	TOP_cvt_fround_sat_u32_f64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvt32f64b64_fround_sat",
	TOP_cvt_fround_sat_s32_f64_b64,
	TOP_cvt_fround_sat_u32_f64_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvt64f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_s64_f32,
	TOP_cvt_fround_ftz_sat_u64_f32,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvt64f64_fround_sat",
	TOP_cvt_fround_sat_s64_f64,
	TOP_cvt_fround_sat_u64_f64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvtf32f64_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_f32_f64,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp64, opnd1);

  Instruction_Group("O_cvtf16f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_f16_f32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvtf32f16_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_f32_f16,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, int32, opnd1);

  Instruction_Group("O_cvtf32f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_f32_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvtf64f32_fround_ftz_sat",
	TOP_cvt_fround_ftz_sat_f64_f32, 
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, ftz);
  Operand(2, sat);
  Operand(3, fp32, opnd1);

  Instruction_Group("O_cvtf64f64_fround_sat",
	TOP_cvt_fround_sat_f64_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, fp64, opnd1);

  Instruction_Group("O_cvtf3216_fround_sat",
	TOP_cvt_fround_sat_f32_s8,
	TOP_cvt_fround_sat_f32_u8,
	TOP_cvt_fround_sat_f32_s16,
	TOP_cvt_fround_sat_f32_u16,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, sat);
  Operand(1, int16, opnd1);

  Instruction_Group("O_cvtf3232_fround_sat",
	TOP_cvt_fround_sat_f32_s32,
	TOP_cvt_fround_sat_f32_u32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, int32, opnd1);

  Instruction_Group("O_cvtf3264_fround_sat",
	TOP_cvt_fround_sat_f32_s64,
	TOP_cvt_fround_sat_f32_u64,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, int64, opnd1);

  Instruction_Group("O_cvtf6416_fround_sat",
	TOP_cvt_fround_sat_f64_s8,
	TOP_cvt_fround_sat_f64_u8,
	TOP_cvt_fround_sat_f64_s16,
	TOP_cvt_fround_sat_f64_u16,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, int16, opnd1);

  Instruction_Group("O_cvtf6432_fround_sat",
	TOP_cvt_fround_sat_f64_s32,
	TOP_cvt_fround_sat_f64_u32,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, int32, opnd1);

  Instruction_Group("O_cvtf6464_fround_sat",
	TOP_cvt_fround_sat_f64_s64,
	TOP_cvt_fround_sat_f64_u64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, fround);
  Operand(1, sat);
  Operand(2, int64, opnd1);

  Instruction_Group("O_cvt161632",
	TOP_cvt_s16_s8_b32, TOP_cvt_s16_u8_b32,
	TOP_cvt_u16_u8_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);	// say both sides reside in 32-bit reg

  Instruction_Group("O_cvt161632_sat",
	TOP_cvt_sat_s8_u8_b32, TOP_cvt_sat_u8_s8_b32,
	TOP_cvt_sat_s8_s16_b32, TOP_cvt_sat_s8_u16_b32,
	TOP_cvt_sat_u8_s16_b32, TOP_cvt_sat_u8_u16_b32,
	TOP_cvt_sat_s16_u16_b32, TOP_cvt_sat_u16_s8_b32,
	TOP_cvt_sat_u16_s16_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, sat);
  Operand(1, int32, opnd1);	// say both sides reside in 32-bit reg

  Instruction_Group("O_cvt163232_sat",
	TOP_cvt_sat_s8_s32_b32, TOP_cvt_sat_s8_u32_b32,
	TOP_cvt_sat_u8_s32_b32, TOP_cvt_sat_u8_u32_b32,
	TOP_cvt_sat_s16_s32_b32, TOP_cvt_sat_s16_u32_b32,
	TOP_cvt_sat_u16_s32_b32, TOP_cvt_sat_u16_u32_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, sat);
  Operand(1, int32, opnd1);

  Instruction_Group("O_cvt161664",
	TOP_cvt_s16_s8_b64, TOP_cvt_s16_u8_b64,
	TOP_cvt_u16_u8_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);	// say both sides reside in 32-bit reg

  Instruction_Group("O_cvt161664_sat",
	TOP_cvt_sat_s8_u8_b64, TOP_cvt_sat_u8_s8_b64,
	TOP_cvt_sat_s8_s16_b64, TOP_cvt_sat_s8_u16_b64,
	TOP_cvt_sat_u8_s16_b64, TOP_cvt_sat_u8_u16_b64,
	TOP_cvt_sat_s16_u16_b64, TOP_cvt_sat_u16_s8_b64,
	TOP_cvt_sat_u16_s16_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);	// say both sides reside in 32-bit reg

  Instruction_Group("O_cvt163264_sat",
	TOP_cvt_sat_s8_s32_b64, TOP_cvt_sat_s8_u32_b64,
	TOP_cvt_sat_u8_s32_b64, TOP_cvt_sat_u8_u32_b64,
	TOP_cvt_sat_s16_s32_b64, TOP_cvt_sat_s16_u32_b64,
	TOP_cvt_sat_u16_s32_b64, TOP_cvt_sat_u16_u32_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt166464_sat",
	TOP_cvt_sat_s8_s64_b64, TOP_cvt_sat_s8_u64_b64,
	TOP_cvt_sat_u8_s64_b64, TOP_cvt_sat_u8_u64_b64,
	TOP_cvt_sat_s16_s64_b64, TOP_cvt_sat_s16_u64_b64,
	TOP_cvt_sat_u16_s64_b64, TOP_cvt_sat_u16_u64_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt321664",
	TOP_cvt_s32_s8_b64, TOP_cvt_s32_s16_b64,
	TOP_cvt_s32_u8_b64, TOP_cvt_s32_u16_b64,
	TOP_cvt_u32_u8_b64, TOP_cvt_u32_u16_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, int64, opnd1);

  Instruction_Group("O_cvt323264_sat",
	TOP_cvt_sat_s32_u32_b64, TOP_cvt_sat_u32_s32_b64,
	TOP_cvt_sat_u32_s8_b64, TOP_cvt_sat_u32_s16_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvt326464_sat",
	TOP_cvt_sat_s32_s64_b64, TOP_cvt_sat_s32_u64_b64,
	TOP_cvt_sat_u32_s64_b64, TOP_cvt_sat_u32_u64_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, sat);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvta32",
	TOP_cvta_space_u32, TOP_cvta_to_space_u32, 
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, int32, opnd1);

  Instruction_Group("O_cvta64",
	TOP_cvta_space_u64, TOP_cvta_to_space_u64, 
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, int64, opnd1);

  Instruction_Group("O_cvta32a",
	TOP_cvta_space_u32_a,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, addr32, base);

  Instruction_Group("O_cvta32ao",
	TOP_cvta_space_u32_ao,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, addr32, base);
  Operand(2, lit32, offset);

  Instruction_Group("O_cvta64a",
	TOP_cvta_space_u64_a,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, addr64, base);

  Instruction_Group("O_cvta64ao",
	TOP_cvta_space_u64_ao,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, addr64, base);
  Operand(2, lit64, offset);

  Instruction_Group("O_call",
	TOP_call, TOP_call_uni,
	TOP_UNDEFINED);
  Operand(0, addr32, target);

  Instruction_Group("O_bra",
	TOP_bra, TOP_bra_uni,
	TOP_UNDEFINED);
  Operand(0, addr32, target);

  Instruction_Group("O_brap",
	TOP_bra_p, TOP_bra_uni_p,
	TOP_bra_np, TOP_bra_uni_np,
	TOP_UNDEFINED);
  Operand(0, pred, predicate);
  Operand(1, addr32, target);

  Instruction_Group("O_bar",
	TOP_bar_sync,
	TOP_pmevent,
	TOP_UNDEFINED);
  Operand(0, lit32, opnd1);

  // atomic ops have implicit def of *op0, like a store
  Instruction_Group("O_atomic32",
	TOP_atom_space_add_s32, TOP_atom_space_add_u32,
	TOP_atom_space_min_s32, TOP_atom_space_min_u32,
	TOP_atom_space_max_s32, TOP_atom_space_max_u32,
	TOP_atom_space_exch_b32,
	TOP_atom_space_and_b32,
	TOP_atom_space_or_b32,
	TOP_atom_space_xor_b32,
	TOP_atom_space_inc_u32, TOP_atom_space_dec_u32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, int32, opnd2);

  // hack to allow float registers in b operation
  Instruction_Group("O_atomic32f",
	TOP_atom_space_exch_b32_f,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp32, opnd2);

  Instruction_Group("O_atomic32e",
	TOP_atom_space_add_s32_a64, TOP_atom_space_add_u32_a64,
	TOP_atom_space_min_s32_a64, TOP_atom_space_min_u32_a64,
	TOP_atom_space_max_s32_a64, TOP_atom_space_max_u32_a64,
	TOP_atom_space_exch_b32_a64,
	TOP_atom_space_and_b32_a64,
	TOP_atom_space_or_b32_a64,
	TOP_atom_space_xor_b32_a64,
	TOP_atom_space_inc_u32_a64, TOP_atom_space_dec_u32_a64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, int32, opnd2);

  Instruction_Group("O_atomic32ef",
	TOP_atom_space_exch_b32_a64_f,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp32, opnd2);

  Instruction_Group("O_atomicas32",
	TOP_atom_space_cas_b32,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, int32, opnd1);
  Operand(3, int32, opnd2);

  Instruction_Group("O_atomicas32f",
	TOP_atom_space_cas_b32_f,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_atomicas32e",
	TOP_atom_space_cas_b32_a64,
	TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, int32, opnd1);
  Operand(3, int32, opnd2);

  Instruction_Group("O_atomicas32ef",
	TOP_atom_space_cas_b32_a64_f,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp32, opnd1);
  Operand(3, fp32, opnd2);

  Instruction_Group("O_atomic64",
	TOP_atom_space_add_s64, TOP_atom_space_add_u64,
	TOP_atom_space_exch_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, int64, opnd2);

  Instruction_Group("O_atomic64f",
	TOP_atom_space_exch_b64_f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_atomic64e",
	TOP_atom_space_add_s64_a64, TOP_atom_space_add_u64_a64,
	TOP_atom_space_exch_b64_a64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, int64, opnd2);

  Instruction_Group("O_atomic64ef",
	TOP_atom_space_exch_b64_a64_f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_atomicas64",
	TOP_atom_space_cas_b64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, int64, opnd1);
  Operand(3, int64, opnd2);

  Instruction_Group("O_atomicas64f",
	TOP_atom_space_cas_b64_f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp64, opnd1);
  Operand(3, fp64, opnd2);

  Instruction_Group("O_atomicas64e",
	TOP_atom_space_cas_b64_a64,
	TOP_UNDEFINED);
  Result(0, int64);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, int64, opnd1);
  Operand(3, int64, opnd2);

  Instruction_Group("O_atomicas64ef",
	TOP_atom_space_cas_b64_a64_f,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp64, opnd1);
  Operand(3, fp64, opnd2);

  Instruction_Group("O_atomicf32",
	TOP_atom_space_add_f32,
	TOP_atom_space_min_f32,
	TOP_atom_space_max_f32,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp32, opnd2);

  Instruction_Group("O_atomicf32e",
	TOP_atom_space_add_f32_a64,
	TOP_atom_space_min_f32_a64,
	TOP_atom_space_max_f32_a64,
	TOP_UNDEFINED);
  Result(0, fp32);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp32, opnd2);

  Instruction_Group("O_atomicf64",
	TOP_atom_space_add_f64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int32, base);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_atomicf64e",
	TOP_atom_space_add_f64_a64,
	TOP_UNDEFINED);
  Result(0, fp64);
  Operand(0, space);
  Operand(1, int64, base);
  Operand(2, fp64, opnd2);

  Instruction_Group("O_vote",
	TOP_vote_all_pred,
	TOP_vote_any_pred,
	TOP_vote_uni_pred,
	TOP_vote_all_pred_not,
	TOP_vote_any_pred_not,
	TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, pred, opnd1);

  Instruction_Group ("O_voteballotb32",
    TOP_vote_ballot_b32,
    TOP_vote_ballot_b32_not,
    TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, pred, opnd1);

  Instruction_Group ("O_barred",
    TOP_bar_red_and_pred,
    TOP_bar_red_or_pred,
    TOP_bar_red_and_pred_not,
    TOP_bar_red_or_pred_not,
    TOP_UNDEFINED);
  Result(0, pred);
  Operand(0, int32, opnd1);
  Operand(1, pred, opnd1);

  Instruction_Group ("O_barredpopc",
    TOP_bar_red_popc_u32,
    TOP_bar_red_popc_u32_not,
    TOP_UNDEFINED);
  Result(0, int32);
  Operand(0, int32, opnd1);
  Operand(1, pred, opnd1);

  Instruction_Group (   "O_intrncall",
                        TOP_intrncall,
                        TOP_UNDEFINED);
  Operand (0, lit32);   // the intrinsic ID
  Result  (0, int32);

  Instruction_Group("O_spadjust",
        TOP_spadjust,
        TOP_UNDEFINED);
  Operand(0, int32, opnd1);
  Operand(1, lit32, opnd2);
  Result(0, int32);

  ISA_Operands_End();
  return 0;
}
