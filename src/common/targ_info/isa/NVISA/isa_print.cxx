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
// Group TOPS with similar Printing format together. 
/////////////////////////////////////////////////////////
// Within each Print_Type instructions are listed in alphabetical order and
// as shown in the ISA manual
/////////////////////////////////////
//
//  $Revision: 1.2 $
//  $Date: 2002/03/26 07:54:05 $
//  $Author: xlp $
//  $Source: /u/merge/src/osprey1.0/common/targ_info/isa/ia64/isa_print.cxx,v $

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "topcode.h"
#include "isa_print_gen.h"

// Multiple topcodes map to the same assembly name. To disambiguate the 
// topcodes, we append a suffix to the basename. By convention, the 
// suffix starts with an underscore. To get the assembly name we strip off
// the suffix.
// In NVISA we can also have enumerations in the middle of the name,
// with type size after the enumeration, e.g. "set.<cmp>.s32_lit"
// To handle this case, we treat everything preceding the enumerations
// (e.g., "set.") as the name, and everything following the last enumeration
// (e.g., ".s32" -- not including the suffix, discussed above) as the type.

static const char *asmname(TOP topcode)
{
  const char *name = TOP_Name(topcode);
  int i = 0;
  while (name[i] && name[i] != '_' && name[i] != '<') {
    ++i;
  }
  
  char *n = (char *)malloc(i + 1);
  strncpy(n, name, i);
  n[i] = 0;
  
  return n;
}

static const char *asmtype(TOP topcode)
{    
  const char *name = TOP_Name(topcode);
  while (*name && *name != '_' && *name != '<') {
    ++name;
  }
  // skip over <enum> sequence (if any)
  while(*name == '<') {
    do {
      ++name;
    } while (*name && *name != '>');
    if (*name) {
      ++name;
    }
  }      
  int i = 0;
  while (name[i] && name[i] != '_') {
    ++i;
  }
  
  char *n = (char *)malloc(i + 1);
  strncpy(n, name, i);
  n[i] = 0;
  
  return n;
}

int main()
{
  ISA_Print_Begin("nvisa");

  Set_AsmName_Func(asmname);
  Set_AsmType_Func(asmtype);  

  Define_Macro("END_GROUP", ";");       // end-of-group marker
  Define_Macro("PREDICATE", "%s");	// predicate operand format
  /* If predicate is empty, will print blank space for it */

  ISA_PRINT_TYPE print_inst = ISA_Print_Type_Create(
	"print_inst", "%s;");
  Name();
  Instruction_Print_Group(print_inst,
	TOP_ret, TOP_nop, TOP_exit, TOP_trap, TOP_brkpt, TOP_membar_gl, 
	TOP_membar_cta,
	TOP_membar_sys,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_bar = ISA_Print_Type_Create(
	"print_bar", "%s \t%s;");
  Name();
  Operand(0);		// d
  Instruction_Print_Group(print_bar,
	TOP_bar_sync,
	TOP_pmevent,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_bra = ISA_Print_Type_Create(
	"print_bra", "%s \t%s;");
  Name();
  Operand(0);		// label
  Instruction_Print_Group(print_bra,
	TOP_bra, TOP_bra_uni,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_brap = ISA_Print_Type_Create(
	"print_brap", "@%s %s \t%s;");
  Operand(0);		// pred
  Name();
  Operand(1);		// label
  Instruction_Print_Group(print_brap,
	TOP_bra_p, TOP_bra_uni_p,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_branp = ISA_Print_Type_Create(
	"print_branp", "@!%s %s \t%s;");
  Operand(0);		// pred
  Name();
  Operand(1);		// label
  Instruction_Print_Group(print_branp,
	TOP_bra_np, TOP_bra_uni_np,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_pbinary = ISA_Print_Type_Create(
	"print_pbinary", "@%s %s \t%s, %s, %s;");
  Operand(2);		// Predicate
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// r3
  Instruction_Print_Group(print_pbinary,
  TOP_sub_s32_p, TOP_sub_s64_p, TOP_UNDEFINED);

  ISA_PRINT_TYPE print_npbinary = ISA_Print_Type_Create(
	"print_npbinary", "@!%s %s \t%s, %s, %s;");
  Operand(2);		// Predicate
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// r3
  Instruction_Print_Group(print_npbinary,
  TOP_sub_s32_np, TOP_sub_s64_np, TOP_UNDEFINED);

  
  ISA_PRINT_TYPE print_unary = ISA_Print_Type_Create(
	"print_unary", "%s \t%s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Instruction_Print_Group(print_unary,
	TOP_abs_s8, TOP_abs_s16, TOP_abs_s32, TOP_abs_s64,
	TOP_abs_u8, TOP_abs_u16, TOP_abs_u32, TOP_abs_u64,
	TOP_abs_f64,
	TOP_sin_f64,
	TOP_cos_f64,
	TOP_lg2_f64,
	TOP_ex2_f64,
	TOP_rcp_f64, TOP_rcp_rn_f64,
        TOP_rcp_approx_ftz_f64,
	TOP_sqrt_f64, TOP_sqrt_rn_f64,
	TOP_not_b8, TOP_not_b16, TOP_not_b32, TOP_not_b64,
	TOP_not_pred,
	TOP_cnot_b8, TOP_cnot_b16, TOP_cnot_b32, TOP_cnot_b64,
	TOP_neg_s8, TOP_neg_s16, TOP_neg_s32, TOP_neg_s64,
	TOP_neg_u8, TOP_neg_u16, TOP_neg_u32, TOP_neg_u64,
	TOP_neg_f64,
	TOP_rcp_rz_f64,
	TOP_rcp_rm_f64, TOP_rcp_rp_f64,
	TOP_sqrt_rz_f64,
	TOP_sqrt_rm_f64, TOP_sqrt_rp_f64,
	TOP_brev_b32, TOP_brev_b64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_unary_ftz = ISA_Print_Type_Create(
	"print_unary_ftz", "%s%s.%s \t%s, %s;");
  Name();		// sin, cos, etc.
  Operand(0);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(1);		// r2
  Instruction_Print_Group(print_unary_ftz,
	TOP_abs_ftz_f32,
	TOP_neg_ftz_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_unary_fround = ISA_Print_Type_Create(
	"print_unary_fround", "%s%s.%s \t%s, %s;");
  Name();		// sin, cos, etc.
  Operand(0);		// fround
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(1);		// r2
  Instruction_Print_Group(print_unary_fround,
	TOP_rsqrt_fround_f64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_unary_froundftz = ISA_Print_Type_Create(
	"print_unary_froundftz", "%s%s%s.%s \t%s, %s;");
  Name();		// sin, cos, etc.
  Operand(0);		// fround
  Operand(1);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Instruction_Print_Group(print_unary_froundftz,
	TOP_sin_fround_ftz_f32,
	TOP_cos_fround_ftz_f32,
	TOP_lg2_fround_ftz_f32,
	TOP_ex2_fround_ftz_f32,
	TOP_rcp_fround_ftz_f32,
	TOP_sqrt_fround_ftz_f32,
	TOP_rsqrt_fround_ftz_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary = ISA_Print_Type_Create(
	"print_binary", "%s \t%s, %s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// r3
  Instruction_Print_Group(print_binary,
	TOP_add_s8, TOP_add_s16, TOP_add_s32, TOP_add_s64,
	TOP_add_u8, TOP_add_u16, TOP_add_u32, TOP_add_u64,
	TOP_add_f64,
	TOP_add_rn_f64, TOP_add_rz_f64,
	TOP_add_rm_f64, TOP_add_rp_f64,
	TOP_add_s8_lit, TOP_add_s16_lit, TOP_add_s32_lit, TOP_add_s64_lit,
	TOP_add_u8_lit, TOP_add_u16_lit, TOP_add_u32_lit, TOP_add_u64_lit,
	TOP_sub_s8, TOP_sub_s16, TOP_sub_s32, TOP_sub_s64,
	TOP_sub_u8, TOP_sub_u16, TOP_sub_u32, TOP_sub_u64,
	TOP_sub_f64,
	TOP_sub_s8_lit, TOP_sub_s16_lit, TOP_sub_s32_lit, TOP_sub_s64_lit,
	TOP_sub_u8_lit, TOP_sub_u16_lit, TOP_sub_u32_lit, TOP_sub_u64_lit,
	TOP_mul_f64,
	TOP_mul_rn_f64, TOP_mul_rz_f64,
	TOP_mul_rm_f64, TOP_mul_rp_f64,
	TOP_mul_lo_s8, TOP_mul_lo_s16, TOP_mul_lo_s32, TOP_mul_lo_s64,
	TOP_mul_lo_u8, TOP_mul_lo_u16, TOP_mul_lo_u32, TOP_mul_lo_u64,
	TOP_mul_lo_s8_lit, TOP_mul_lo_s16_lit, TOP_mul_lo_s32_lit, TOP_mul_lo_s64_lit,
	TOP_mul_lo_u8_lit, TOP_mul_lo_u16_lit, TOP_mul_lo_u32_lit, TOP_mul_lo_u64_lit,
	TOP_mul_hi_s8, TOP_mul_hi_s16, TOP_mul_hi_s32, TOP_mul_hi_s64,
	TOP_mul_hi_u8, TOP_mul_hi_u16, TOP_mul_hi_u32, TOP_mul_hi_u64,
	TOP_mul_wide_s16, TOP_mul_wide_u16, 
	TOP_mul_wide_s32, TOP_mul_wide_u32,
	TOP_mul_wide_s16_lit, TOP_mul_wide_u16_lit, 
	TOP_mul_wide_s32_lit, TOP_mul_wide_u32_lit,
	TOP_mul24_lo_s32, TOP_mul24_lo_u32, 
	TOP_mul24_lo_s32_lit, TOP_mul24_lo_u32_lit,
	TOP_div_s8, TOP_div_s16, TOP_div_s32, TOP_div_s64,
	TOP_div_u8, TOP_div_u16, TOP_div_u32, TOP_div_u64,
	TOP_div_f64,
	TOP_div_s8_lit, TOP_div_s16_lit, TOP_div_s32_lit, TOP_div_s64_lit,
	TOP_div_u8_lit, TOP_div_u16_lit, TOP_div_u32_lit, TOP_div_u64_lit,
	TOP_div_rn_f64, TOP_div_rz_f64,
	TOP_div_rm_f64, TOP_div_rp_f64,
	TOP_rem_s8, TOP_rem_s16, TOP_rem_s32, TOP_rem_s64,
	TOP_rem_u8, TOP_rem_u16, TOP_rem_u32, TOP_rem_u64,
	TOP_rem_s8_lit, TOP_rem_s16_lit, TOP_rem_s32_lit, TOP_rem_s64_lit,
	TOP_rem_u8_lit, TOP_rem_u16_lit, TOP_rem_u32_lit, TOP_rem_u64_lit,
	TOP_min_s8, TOP_min_s16, TOP_min_s32, TOP_min_s64,
	TOP_min_u8, TOP_min_u16, TOP_min_u32, TOP_min_u64,
	TOP_min_f64,
	TOP_max_s8, TOP_max_s16, TOP_max_s32, TOP_max_s64,
	TOP_max_u8, TOP_max_u16, TOP_max_u32, TOP_max_u64,
	TOP_max_f64,
	TOP_and_b8, TOP_and_b16, TOP_and_b32, TOP_and_b64,
	TOP_and_b8_lit, TOP_and_b16_lit, TOP_and_b32_lit, TOP_and_b64_lit,
	TOP_or_b8, TOP_or_b16, TOP_or_b32, TOP_or_b64,
	TOP_or_b8_lit, TOP_or_b16_lit, TOP_or_b32_lit, TOP_or_b64_lit,
	TOP_xor_b8, TOP_xor_b16, TOP_xor_b32, TOP_xor_b64,
	TOP_xor_b8_lit, TOP_xor_b16_lit, TOP_xor_b32_lit, TOP_xor_b64_lit,
	TOP_and_pred, TOP_or_pred, TOP_xor_pred,
	TOP_shl_b8, TOP_shl_b16, TOP_shl_b32, TOP_shl_b64,
	TOP_shr_s8, TOP_shr_s16, TOP_shr_s32, TOP_shr_s64,
	TOP_shr_u8, TOP_shr_u16, TOP_shr_u32, TOP_shr_u64,
	TOP_shl_b8_lit, TOP_shl_b16_lit, TOP_shl_b32_lit, TOP_shl_b64_lit,
	TOP_shr_s8_lit, TOP_shr_s16_lit, TOP_shr_s32_lit, TOP_shr_s64_lit,
	TOP_shr_u8_lit, TOP_shr_u16_lit, TOP_shr_u32_lit, TOP_shr_u64_lit,
	TOP_shl_b8_lit1, TOP_shl_b16_lit1, TOP_shl_b32_lit1, TOP_shl_b64_lit1,
	TOP_shr_s8_lit1, TOP_shr_s16_lit1, TOP_shr_s32_lit1, TOP_shr_s64_lit1,
	TOP_shr_u8_lit1, TOP_shr_u16_lit1, TOP_shr_u32_lit1, TOP_shr_u64_lit1,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary_cmp = ISA_Print_Type_Create(
	"print_binary_cmp", "%s.%s.%s \t%s, %s, %s;");
  Name();		// div, etc.
  Operand(0);		// cmp
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Instruction_Print_Group(print_binary_cmp,
	TOP_set_cmp_u16_s8, TOP_set_cmp_u16_s16, TOP_set_cmp_u16_s32, TOP_set_cmp_u16_s64,
	TOP_set_cmp_u16_u8, TOP_set_cmp_u16_u16, TOP_set_cmp_u16_u32, TOP_set_cmp_u16_u64,
	TOP_set_cmp_u16_f64,
	TOP_set_cmp_u32_s8, TOP_set_cmp_u32_s16, TOP_set_cmp_u32_s32, TOP_set_cmp_u32_s64,
	TOP_set_cmp_u32_u8, TOP_set_cmp_u32_u16, TOP_set_cmp_u32_u32, TOP_set_cmp_u32_u64,
	TOP_set_cmp_u32_f64,
	TOP_setp_cmp_s8, TOP_setp_cmp_s16, TOP_setp_cmp_s32, TOP_setp_cmp_s64,
	TOP_setp_cmp_u8, TOP_setp_cmp_u16, TOP_setp_cmp_u32, TOP_setp_cmp_u64,
	TOP_setp_cmp_f64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary_ftz = ISA_Print_Type_Create(
	"print_binary_ftz", "%s%s.%s \t%s, %s, %s;");
  Name();		// div, etc.
  Operand(0);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Instruction_Print_Group(print_binary_ftz,
	TOP_min_ftz_f32,
	TOP_max_ftz_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary_cmpftz = ISA_Print_Type_Create(
	"print_binary_cmpftz", "%s.%s%s.%s \t%s, %s, %s;");
  Name();		// div, etc.
  Operand(0);		// cmp
  Operand(1);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Operand(3);		// r3
  Instruction_Print_Group(print_binary_cmpftz,
	TOP_set_cmp_ftz_u16_f32,
	TOP_set_cmp_ftz_u32_f32,
	TOP_setp_cmp_ftz_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary_froundftz = ISA_Print_Type_Create(
	"print_binary_froundftz", "%s%s%s.%s \t%s, %s, %s;");
  Name();		// div, etc.
  Operand(0);		// fround
  Operand(1);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Operand(3);		// r3
  Instruction_Print_Group(print_binary_froundftz,
	TOP_div_fround_ftz_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_binary_froundftzsat = ISA_Print_Type_Create(
	"print_binary_froundftzsat", "%s%s%s%s.%s \t%s, %s, %s;");
  Name();		// div, etc.
  Operand(0);		// fround
  Operand(1);		// ftz
  Operand(2);		// sat
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(3);		// r2
  Operand(4);		// r3
  Instruction_Print_Group(print_binary_froundftzsat,
	TOP_add_fround_ftz_sat_f32,
	TOP_sub_fround_ftz_sat_f32,
	TOP_mul_fround_ftz_sat_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_psetp = ISA_Print_Type_Create(
	"print_psetp", "@%s %s \t%s, %s, %s;");
  Operand(0);		// pred
  Name();
  Result(0);		// p 
  Operand(1);		// r1
  Operand(2);		// r2
  Instruction_Print_Group(print_psetp,
	TOP_setp_eq_u32_p, TOP_setp_ne_u32_p,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_triple = ISA_Print_Type_Create(
	"print_triple", "%s \t%s, %s, %s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// r3
  Operand(2);		// r4
  Instruction_Print_Group(print_triple,
	TOP_mad_f32, TOP_mad_f64,
	TOP_mad_rn_f64, TOP_mad_rz_f64,
	TOP_mad_rm_f64, TOP_mad_rp_f64, 
	TOP_mad_lo_s8, TOP_mad_lo_s16, TOP_mad_lo_s32, TOP_mad_lo_s64,
	TOP_mad_lo_u8, TOP_mad_lo_u16, TOP_mad_lo_u32, TOP_mad_lo_u64,
	TOP_mad_hi_s8, TOP_mad_hi_s16, TOP_mad_hi_s32, TOP_mad_hi_s64,
	TOP_mad_hi_u8, TOP_mad_hi_u16, TOP_mad_hi_u32, TOP_mad_hi_u64,
	TOP_mad24_lo_s32, TOP_mad24_lo_u32,
	TOP_mad_wide_s16, TOP_mad_wide_u16, 
	TOP_mad_wide_s32, TOP_mad_wide_u32,
	TOP_sad_s32, TOP_sad_u32,
	TOP_slct_s8_s32, TOP_slct_u8_s32,
	TOP_slct_s16_s32, TOP_slct_u16_s32,
	TOP_slct_s32_s32, TOP_slct_u32_s32,
	TOP_slct_s64_s32, TOP_slct_u64_s32,
	TOP_slct_f32_s32, TOP_slct_f64_s32,
	TOP_selp_s8, TOP_selp_u8,
	TOP_selp_s16, TOP_selp_u16,
	TOP_selp_s32, TOP_selp_u32,
	TOP_selp_s64, TOP_selp_u64,
	TOP_selp_f32, TOP_selp_f64,
	TOP_selp_s8_lit, TOP_selp_u8_lit,
	TOP_selp_s16_lit, TOP_selp_u16_lit,
	TOP_selp_s32_lit, TOP_selp_u32_lit,
	TOP_selp_s64_lit, TOP_selp_u64_lit,
	TOP_selp_f32_lit, TOP_selp_f64_lit,
	TOP_prmt_b32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_triple_ftz = ISA_Print_Type_Create
	("print_triple_ftz", "%s%s.%s \t%s, %s, %s, %s;");
  Name();
  Operand(0);		// ftz
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Operand(3);		// rc
  Instruction_Print_Group(print_triple_ftz,
	TOP_slct_ftz_s8_f32, TOP_slct_ftz_u8_f32,
	TOP_slct_ftz_s16_f32, TOP_slct_ftz_u16_f32,
	TOP_slct_ftz_s32_f32, TOP_slct_ftz_u32_f32,
	TOP_slct_ftz_s64_f32, TOP_slct_ftz_u64_f32,
	TOP_slct_ftz_f32_f32, TOP_slct_ftz_f64_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_fma32 = ISA_Print_Type_Create(
	"print_fma32", "%s%s%s%s.%s \t%s, %s, %s, %s;");
  Name();		// fma
  Operand(0);		// fround 
  Operand(1);		// ftz 
  Operand(2);		// sat 
  Type();		// f32, etc.
  Result(0);		// r1
  Operand(3);		// r2
  Operand(4);		// r3
  Operand(5);		// r4
  Instruction_Print_Group(print_fma32,
	TOP_fma_fround_ftz_sat_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_fma64 = ISA_Print_Type_Create(
	"print_fma64", "%s%s.%s \t%s, %s, %s, %s;");
  Name();		// fma
  Operand(0);		// fround 
  Type();		// f64
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Operand(3);		// r4
  Instruction_Print_Group(print_fma64,
	TOP_fma_fround_f64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_setb = ISA_Print_Type_Create(
	"print_setb", "%s.%s.%s.%s \t%s, %s, %s, %s;");
  Name();		// set, setp
  Operand(0);		// cmp
  Operand(1);		// boolop
  Type();		// u32.u16, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Operand(3);		// r3
  Operand(4);		// r4
  Instruction_Print_Group(print_setb,
	TOP_set_cmp_boolop_u16_s16_lit, TOP_set_cmp_boolop_u16_u16_lit,
	TOP_set_cmp_boolop_u16_s32_lit, TOP_set_cmp_boolop_u16_u32_lit,
	TOP_set_cmp_boolop_u32_s16_lit, TOP_set_cmp_boolop_u32_u16_lit,
	TOP_set_cmp_boolop_u32_s32_lit, TOP_set_cmp_boolop_u32_u32_lit,
	TOP_setp_cmp_boolop_u32_lit,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_mov = ISA_Print_Type_Create
	("print_mov", "%s \t%s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Instruction_Print_Group(print_mov,
	TOP_mov_s8, TOP_mov_s16, TOP_mov_s32, TOP_mov_s64,
	TOP_mov_u8, TOP_mov_u16, TOP_mov_u32, TOP_mov_u64,
	TOP_mov_f32, TOP_mov_f64,
	TOP_mov_b32_i2f, TOP_mov_b32_f2i, TOP_mov_b64_i2f, TOP_mov_b64_f2i,
	TOP_mov_s8_lit, TOP_mov_s16_lit, TOP_mov_s32_lit, TOP_mov_s64_lit,
	TOP_mov_u8_lit, TOP_mov_u16_lit, TOP_mov_u32_lit, TOP_mov_u64_lit,
	TOP_mov_f32_lit, TOP_mov_f64_lit,
	TOP_mov_pred,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_mov12 = ISA_Print_Type_Create
	("print_mov", "%s \t%s, {%s,%s};");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// r3
  Instruction_Print_Group(print_mov12,
	TOP_mov_b64_i22f,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_mov21 = ISA_Print_Type_Create
	("print_mov", "%s \t{%s,%s}, %s;");
  Name();
  Result(0);		// r1
  Result(1);		// r2
  Operand(0);		// r3
  Instruction_Print_Group(print_mov21,
	TOP_mov_b64_f2i2,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ld = ISA_Print_Type_Create
	("print_ld", "%s%s%s.%s \t%s, [%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);		// space 
  Type();		// s32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Instruction_Print_Group(print_ld,
	TOP_ld_qualifier_space_s8, TOP_ld_qualifier_space_s16, 
	TOP_ld_qualifier_space_s32, TOP_ld_qualifier_space_s64,
	TOP_ld_qualifier_space_u8, TOP_ld_qualifier_space_u16, 
	TOP_ld_qualifier_space_u32, TOP_ld_qualifier_space_u64,
	TOP_ld_qualifier_space_f32, TOP_ld_qualifier_space_f64,
	TOP_ld_qualifier_space_s8_b32, TOP_ld_qualifier_space_s16_b32, 
	TOP_ld_qualifier_space_u8_b32, TOP_ld_qualifier_space_u16_b32, 
	TOP_ld_qualifier_space_s8_a64, TOP_ld_qualifier_space_s16_a64, 
	TOP_ld_qualifier_space_s32_a64, TOP_ld_qualifier_space_s64_a64,
	TOP_ld_qualifier_space_u8_a64, TOP_ld_qualifier_space_u16_a64, 
	TOP_ld_qualifier_space_u32_a64, TOP_ld_qualifier_space_u64_a64,
	TOP_ld_qualifier_space_f32_a64, TOP_ld_qualifier_space_f64_a64,
	TOP_ld_qualifier_space_s8_b32_a64, TOP_ld_qualifier_space_s16_b32_a64, 
	TOP_ld_qualifier_space_u8_b32_a64, TOP_ld_qualifier_space_u16_b32_a64, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldo = ISA_Print_Type_Create
	("print_ldo", "%s%s%s.%s \t%s, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);		// space 
  Type();		// s32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Operand(3);		// ofst
  Instruction_Print_Group(print_ldo,
	TOP_ld_qualifier_space_s8_o, TOP_ld_qualifier_space_s16_o, 
	TOP_ld_qualifier_space_s32_o, TOP_ld_qualifier_space_s64_o,
	TOP_ld_qualifier_space_u8_o, TOP_ld_qualifier_space_u16_o, 
	TOP_ld_qualifier_space_u32_o, TOP_ld_qualifier_space_u64_o,
	TOP_ld_qualifier_space_f32_o, TOP_ld_qualifier_space_f64_o,
	TOP_ld_qualifier_space_s8_b32_o, TOP_ld_qualifier_space_s16_b32_o, 
	TOP_ld_qualifier_space_u8_b32_o, TOP_ld_qualifier_space_u16_b32_o, 
	TOP_ld_qualifier_space_s8_a64_o, TOP_ld_qualifier_space_s16_a64_o, 
	TOP_ld_qualifier_space_s32_a64_o, TOP_ld_qualifier_space_s64_a64_o,
	TOP_ld_qualifier_space_u8_a64_o, TOP_ld_qualifier_space_u16_a64_o, 
	TOP_ld_qualifier_space_u32_a64_o, TOP_ld_qualifier_space_u64_a64_o,
	TOP_ld_qualifier_space_f32_a64_o, TOP_ld_qualifier_space_f64_a64_o,
	TOP_ld_qualifier_space_s8_b32_a64_o, TOP_ld_qualifier_space_s16_b32_a64_o, 
	TOP_ld_qualifier_space_u8_b32_a64_o, TOP_ld_qualifier_space_u16_b32_a64_o, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldr = ISA_Print_Type_Create
	("print_ldr", "%s%s%s.%s \t%s, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);		// space 
  Type();		// s32, etc.
  Result(0);		// r1
  Operand(2);		// r2
  Operand(3);		// o
  Instruction_Print_Group(print_ldr,
	TOP_ld_qualifier_space_s8_r, TOP_ld_qualifier_space_s16_r, 
	TOP_ld_qualifier_space_s32_r, TOP_ld_qualifier_space_s64_r,
	TOP_ld_qualifier_space_u8_r, TOP_ld_qualifier_space_u16_r, 
	TOP_ld_qualifier_space_u32_r, TOP_ld_qualifier_space_u64_r,
	TOP_ld_qualifier_space_f32_r, TOP_ld_qualifier_space_f64_r,
	TOP_ld_qualifier_space_s8_b32_r, TOP_ld_qualifier_space_s16_b32_r, 
	TOP_ld_qualifier_space_u8_b32_r, TOP_ld_qualifier_space_u16_b32_r, 
	TOP_ld_qualifier_space_s8_a64_r, TOP_ld_qualifier_space_s16_a64_r, 
	TOP_ld_qualifier_space_s32_a64_r, TOP_ld_qualifier_space_s64_a64_r,
	TOP_ld_qualifier_space_u8_a64_r, TOP_ld_qualifier_space_u16_a64_r, 
	TOP_ld_qualifier_space_u32_a64_r, TOP_ld_qualifier_space_u64_a64_r,
	TOP_ld_qualifier_space_f32_a64_r, TOP_ld_qualifier_space_f64_a64_r,
	TOP_ld_qualifier_space_s8_b32_a64_r, TOP_ld_qualifier_space_s16_b32_a64_r, 
	TOP_ld_qualifier_space_u8_b32_a64_r, TOP_ld_qualifier_space_u16_b32_a64_r, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_lda = ISA_Print_Type_Create
	("print_lda", "%s \t%s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Instruction_Print_Group(print_lda,
	TOP_mov_u32_a, TOP_mov_u64_a,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldao = ISA_Print_Type_Create
	("print_ldao", "%s \t%s, %s+%s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Operand(1);		// ofst
  Instruction_Print_Group(print_ldao,
	TOP_mov_u32_ao, TOP_mov_u64_ao,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldv2r = ISA_Print_Type_Create
	("print_ldv2r", "%s%s%s.%s \t{%s,%s}, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// v2.s16, etc.
  Result(0);		// r1
  Result(1);		// r2
  Operand(2);		// r 
  Operand(3);		// o
  Instruction_Print_Group(print_ldv2r,
	TOP_ld_qualifier_space_v2_s8_r, TOP_ld_qualifier_space_v2_u8_r, 
	TOP_ld_qualifier_space_v2_s16_r, TOP_ld_qualifier_space_v2_u16_r, 
	TOP_ld_qualifier_space_v2_s8_b32_r, TOP_ld_qualifier_space_v2_u8_b32_r, 
	TOP_ld_qualifier_space_v2_s16_b32_r, TOP_ld_qualifier_space_v2_u16_b32_r, 
	TOP_ld_qualifier_space_v2_s32_r, TOP_ld_qualifier_space_v2_u32_r, 
	TOP_ld_qualifier_space_v2_s64_r, TOP_ld_qualifier_space_v2_u64_r, 
	TOP_ld_qualifier_space_v2_f32_r, TOP_ld_qualifier_space_v2_f64_r, 
	TOP_ld_qualifier_space_v2_s8_a64_r, TOP_ld_qualifier_space_v2_u8_a64_r, 
	TOP_ld_qualifier_space_v2_s16_a64_r, TOP_ld_qualifier_space_v2_u16_a64_r, 
	TOP_ld_qualifier_space_v2_s8_b32_a64_r, TOP_ld_qualifier_space_v2_u8_b32_a64_r, 
	TOP_ld_qualifier_space_v2_s16_b32_a64_r, TOP_ld_qualifier_space_v2_u16_b32_a64_r, 
	TOP_ld_qualifier_space_v2_s32_a64_r, TOP_ld_qualifier_space_v2_u32_a64_r, 
	TOP_ld_qualifier_space_v2_s64_a64_r, TOP_ld_qualifier_space_v2_u64_a64_r, 
	TOP_ld_qualifier_space_v2_f32_a64_r, TOP_ld_qualifier_space_v2_f64_a64_r, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldv4r = ISA_Print_Type_Create
	("print_ldv4r", "%s%s%s.%s \t{%s,%s,%s,%s}, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// v4.s16, etc.
  Result(0);		// r1
  Result(1);		// r2
  Result(2);		// r3
  Result(3);		// r4
  Operand(2);		// r 
  Operand(3);		// o
  Instruction_Print_Group(print_ldv4r,
	TOP_ld_qualifier_space_v4_s8_r, TOP_ld_qualifier_space_v4_u8_r, 
	TOP_ld_qualifier_space_v4_s16_r, TOP_ld_qualifier_space_v4_u16_r, 
	TOP_ld_qualifier_space_v4_s8_b32_r, TOP_ld_qualifier_space_v4_u8_b32_r, 
	TOP_ld_qualifier_space_v4_s16_b32_r, TOP_ld_qualifier_space_v4_u16_b32_r, 
	TOP_ld_qualifier_space_v4_s32_r, TOP_ld_qualifier_space_v4_u32_r, 
	TOP_ld_qualifier_space_v4_f32_r,
	TOP_ld_qualifier_space_v4_s8_a64_r, TOP_ld_qualifier_space_v4_u8_a64_r, 
	TOP_ld_qualifier_space_v4_s16_a64_r, TOP_ld_qualifier_space_v4_u16_a64_r, 
	TOP_ld_qualifier_space_v4_s8_b32_a64_r, TOP_ld_qualifier_space_v4_u8_b32_a64_r, 
	TOP_ld_qualifier_space_v4_s16_b32_a64_r, TOP_ld_qualifier_space_v4_u16_b32_a64_r, 
	TOP_ld_qualifier_space_v4_s32_a64_r, TOP_ld_qualifier_space_v4_u32_a64_r, 
	TOP_ld_qualifier_space_v4_f32_a64_r,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldv2o = ISA_Print_Type_Create
	("print_ldv2o", "%s%s%s.%s \t{%s,%s}, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// v2.u16, etc.
  Result(0);		// r1
  Result(1);		// r2
  Operand(2);		// r 
  Operand(3);		// o
  Instruction_Print_Group(print_ldv2o,
	TOP_ld_qualifier_space_v2_s8_o, TOP_ld_qualifier_space_v2_u8_o, 
	TOP_ld_qualifier_space_v2_s16_o, TOP_ld_qualifier_space_v2_u16_o, 
	TOP_ld_qualifier_space_v2_s8_b32_o, TOP_ld_qualifier_space_v2_u8_b32_o, 
	TOP_ld_qualifier_space_v2_s16_b32_o, TOP_ld_qualifier_space_v2_u16_b32_o, 
	TOP_ld_qualifier_space_v2_s32_o, TOP_ld_qualifier_space_v2_u32_o, 
	TOP_ld_qualifier_space_v2_s64_o, TOP_ld_qualifier_space_v2_u64_o, 
	TOP_ld_qualifier_space_v2_f32_o, TOP_ld_qualifier_space_v2_f64_o, 
	TOP_ld_qualifier_space_v2_s8_a64_o, TOP_ld_qualifier_space_v2_u8_a64_o, 
	TOP_ld_qualifier_space_v2_s16_a64_o, TOP_ld_qualifier_space_v2_u16_a64_o, 
	TOP_ld_qualifier_space_v2_s8_b32_a64_o, TOP_ld_qualifier_space_v2_u8_b32_a64_o, 
	TOP_ld_qualifier_space_v2_s16_b32_a64_o, TOP_ld_qualifier_space_v2_u16_b32_a64_o, 
	TOP_ld_qualifier_space_v2_s32_a64_o, TOP_ld_qualifier_space_v2_u32_a64_o, 
	TOP_ld_qualifier_space_v2_s64_a64_o, TOP_ld_qualifier_space_v2_u64_a64_o, 
	TOP_ld_qualifier_space_v2_f32_a64_o, TOP_ld_qualifier_space_v2_f64_a64_o, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_ldv4o = ISA_Print_Type_Create
	("print_ldv4o", "%s%s%s.%s \t{%s,%s,%s,%s}, [%s+%s];");
  Name();		// ld
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// v4.u16, etc.
  Result(0);		// r1
  Result(1);		// r2
  Result(2);		// r3
  Result(3);		// r4
  Operand(2);		// r 
  Operand(3);		// o
  Instruction_Print_Group(print_ldv4o,
	TOP_ld_qualifier_space_v4_s8_o, TOP_ld_qualifier_space_v4_u8_o, 
	TOP_ld_qualifier_space_v4_s16_o, TOP_ld_qualifier_space_v4_u16_o, 
	TOP_ld_qualifier_space_v4_s8_b32_o, TOP_ld_qualifier_space_v4_u8_b32_o, 
	TOP_ld_qualifier_space_v4_s16_b32_o, TOP_ld_qualifier_space_v4_u16_b32_o, 
	TOP_ld_qualifier_space_v4_s32_o, TOP_ld_qualifier_space_v4_u32_o, 
	TOP_ld_qualifier_space_v4_f32_o,
	TOP_ld_qualifier_space_v4_s8_a64_o, TOP_ld_qualifier_space_v4_u8_a64_o, 
	TOP_ld_qualifier_space_v4_s16_a64_o, TOP_ld_qualifier_space_v4_u16_a64_o, 
	TOP_ld_qualifier_space_v4_s8_b32_a64_o, TOP_ld_qualifier_space_v4_u8_b32_a64_o, 
	TOP_ld_qualifier_space_v4_s16_b32_a64_o, TOP_ld_qualifier_space_v4_u16_b32_a64_o, 
	TOP_ld_qualifier_space_v4_s32_a64_o, TOP_ld_qualifier_space_v4_u32_a64_o, 
	TOP_ld_qualifier_space_v4_f32_a64_o,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_st = ISA_Print_Type_Create
	("print_st", "%s%s%s.%s \t[%s], %s;");
  Name();		// st
  Operand(0);		// qualifier 
  Operand(1);		// space 
  Type();		// f32, etc.
  Operand(2);		// r1
  Operand(3);		// r2
  Instruction_Print_Group(print_st,
	TOP_st_qualifier_space_s8, TOP_st_qualifier_space_s16, 
	TOP_st_qualifier_space_s32, TOP_st_qualifier_space_s64,
	TOP_st_qualifier_space_u8, TOP_st_qualifier_space_u16, 
	TOP_st_qualifier_space_u32, TOP_st_qualifier_space_u64,
	TOP_st_qualifier_space_f32, TOP_st_qualifier_space_f64,
	TOP_st_qualifier_space_s8_b32, TOP_st_qualifier_space_s16_b32, 
	TOP_st_qualifier_space_u8_b32, TOP_st_qualifier_space_u16_b32, 
	TOP_st_qualifier_space_s8_a64, TOP_st_qualifier_space_s16_a64, 
	TOP_st_qualifier_space_s32_a64, TOP_st_qualifier_space_s64_a64,
	TOP_st_qualifier_space_u8_a64, TOP_st_qualifier_space_u16_a64, 
	TOP_st_qualifier_space_u32_a64, TOP_st_qualifier_space_u64_a64,
	TOP_st_qualifier_space_f32_a64, TOP_st_qualifier_space_f64_a64,
	TOP_st_qualifier_space_s8_b32_a64, TOP_st_qualifier_space_s16_b32_a64, 
	TOP_st_qualifier_space_u8_b32_a64, TOP_st_qualifier_space_u16_b32_a64, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_sto = ISA_Print_Type_Create
	("print_sto", "%s%s%s.%s \t[%s+%s], %s;");
  Name();		// st
  Operand(0);		// qualifier
  Operand(1);		// space 
  Type();		// f32, etc.
  Operand(2);		// r1
  Operand(3);		// ofst
  Operand(4);		// r2
  Instruction_Print_Group(print_sto,
	TOP_st_qualifier_space_s8_o, TOP_st_qualifier_space_s16_o, 
	TOP_st_qualifier_space_s32_o, TOP_st_qualifier_space_s64_o,
	TOP_st_qualifier_space_u8_o, TOP_st_qualifier_space_u16_o, 
	TOP_st_qualifier_space_u32_o, TOP_st_qualifier_space_u64_o,
	TOP_st_qualifier_space_f32_o, TOP_st_qualifier_space_f64_o,
	TOP_st_qualifier_space_s8_b32_o, TOP_st_qualifier_space_s16_b32_o, 
	TOP_st_qualifier_space_u8_b32_o, TOP_st_qualifier_space_u16_b32_o, 
	TOP_st_qualifier_space_s8_a64_o, TOP_st_qualifier_space_s16_a64_o, 
	TOP_st_qualifier_space_s32_a64_o, TOP_st_qualifier_space_s64_a64_o,
	TOP_st_qualifier_space_u8_a64_o, TOP_st_qualifier_space_u16_a64_o, 
	TOP_st_qualifier_space_u32_a64_o, TOP_st_qualifier_space_u64_a64_o,
	TOP_st_qualifier_space_f32_a64_o, TOP_st_qualifier_space_f64_a64_o,
	TOP_st_qualifier_space_s8_b32_a64_o, TOP_st_qualifier_space_s16_b32_a64_o, 
	TOP_st_qualifier_space_u8_b32_a64_o, TOP_st_qualifier_space_u16_b32_a64_o, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_str = ISA_Print_Type_Create
	("print_str", "%s%s%s.%s \t[%s+%s], %s;");
  Name();		// st
  Operand(0);		// qualifier
  Operand(1);		// space 
  Type();		// f32, etc.
  Operand(2);		// r1
  Operand(3);		// o
  Operand(4);		// r2
  Instruction_Print_Group(print_str,
	TOP_st_qualifier_space_s8_r, TOP_st_qualifier_space_s16_r, 
	TOP_st_qualifier_space_s32_r, TOP_st_qualifier_space_s64_r,
	TOP_st_qualifier_space_u8_r, TOP_st_qualifier_space_u16_r, 
	TOP_st_qualifier_space_u32_r, TOP_st_qualifier_space_u64_r,
	TOP_st_qualifier_space_f32_r, TOP_st_qualifier_space_f64_r,
	TOP_st_qualifier_space_s8_b32_r, TOP_st_qualifier_space_s16_b32_r, 
	TOP_st_qualifier_space_u8_b32_r, TOP_st_qualifier_space_u16_b32_r, 
	TOP_st_qualifier_space_s8_a64_r, TOP_st_qualifier_space_s16_a64_r, 
	TOP_st_qualifier_space_s32_a64_r, TOP_st_qualifier_space_s64_a64_r,
	TOP_st_qualifier_space_u8_a64_r, TOP_st_qualifier_space_u16_a64_r, 
	TOP_st_qualifier_space_u32_a64_r, TOP_st_qualifier_space_u64_a64_r,
	TOP_st_qualifier_space_f32_a64_r, TOP_st_qualifier_space_f64_a64_r,
	TOP_st_qualifier_space_s8_b32_a64_r, TOP_st_qualifier_space_s16_b32_a64_r, 
	TOP_st_qualifier_space_u8_b32_a64_r, TOP_st_qualifier_space_u16_b32_a64_r, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_stv2r = ISA_Print_Type_Create
	("print_stv2r", "%s%s%s.%s \t[%s+%s], {%s,%s};");
  Name();		// st
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// f32, etc.
  Operand(2);		// r 
  Operand(3);		// o
  Operand(4);		// r1
  Operand(5);		// r2
  Instruction_Print_Group(print_stv2r,
	TOP_st_qualifier_space_v2_s8_r, TOP_st_qualifier_space_v2_u8_r, 
	TOP_st_qualifier_space_v2_s16_r, TOP_st_qualifier_space_v2_u16_r, 
	TOP_st_qualifier_space_v2_s8_b32_r, TOP_st_qualifier_space_v2_u8_b32_r, 
	TOP_st_qualifier_space_v2_s16_b32_r, TOP_st_qualifier_space_v2_u16_b32_r, 
	TOP_st_qualifier_space_v2_s32_r, TOP_st_qualifier_space_v2_u32_r, 
	TOP_st_qualifier_space_v2_s64_r, TOP_st_qualifier_space_v2_u64_r, 
	TOP_st_qualifier_space_v2_f32_r, TOP_st_qualifier_space_v2_f64_r, 
	TOP_st_qualifier_space_v2_s8_a64_r, TOP_st_qualifier_space_v2_u8_a64_r, 
	TOP_st_qualifier_space_v2_s16_a64_r, TOP_st_qualifier_space_v2_u16_a64_r, 
	TOP_st_qualifier_space_v2_s8_b32_a64_r, TOP_st_qualifier_space_v2_u8_b32_a64_r, 
	TOP_st_qualifier_space_v2_s16_b32_a64_r, TOP_st_qualifier_space_v2_u16_b32_a64_r, 
	TOP_st_qualifier_space_v2_s32_a64_r, TOP_st_qualifier_space_v2_u32_a64_r, 
	TOP_st_qualifier_space_v2_s64_a64_r, TOP_st_qualifier_space_v2_u64_a64_r, 
	TOP_st_qualifier_space_v2_f32_a64_r, TOP_st_qualifier_space_v2_f64_a64_r, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_stv4r = ISA_Print_Type_Create
	("print_stv4r", "%s%s%s.%s \t[%s+%s], {%s,%s,%s,%s};");
  Name();		// st
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// f32, etc.
  Operand(2);		// r 
  Operand(3);		// o
  Operand(4);		// r1
  Operand(5);		// r2
  Operand(6);		// r3
  Operand(7);		// r4
  Instruction_Print_Group(print_stv4r,
	TOP_st_qualifier_space_v4_s8_r, TOP_st_qualifier_space_v4_u8_r, 
	TOP_st_qualifier_space_v4_s16_r, TOP_st_qualifier_space_v4_u16_r, 
	TOP_st_qualifier_space_v4_s8_b32_r, TOP_st_qualifier_space_v4_u8_b32_r, 
	TOP_st_qualifier_space_v4_s16_b32_r, TOP_st_qualifier_space_v4_u16_b32_r, 
	TOP_st_qualifier_space_v4_s32_r, TOP_st_qualifier_space_v4_u32_r, 
	TOP_st_qualifier_space_v4_f32_r,
	TOP_st_qualifier_space_v4_s8_a64_r, TOP_st_qualifier_space_v4_u8_a64_r, 
	TOP_st_qualifier_space_v4_s16_a64_r, TOP_st_qualifier_space_v4_u16_a64_r, 
	TOP_st_qualifier_space_v4_s8_b32_a64_r, TOP_st_qualifier_space_v4_u8_b32_a64_r, 
	TOP_st_qualifier_space_v4_s16_b32_a64_r, TOP_st_qualifier_space_v4_u16_b32_a64_r, 
	TOP_st_qualifier_space_v4_s32_a64_r, TOP_st_qualifier_space_v4_u32_a64_r, 
	TOP_st_qualifier_space_v4_f32_a64_r,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_stv2o = ISA_Print_Type_Create
	("print_stv2o", "%s%s%s.%s \t[%s+%s], {%s,%s};");
  Name();		// st
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// f32, etc.
  Operand(2);		// r 
  Operand(3);		// o
  Operand(4);		// r1
  Operand(5);		// r2
  Instruction_Print_Group(print_stv2o,
	TOP_st_qualifier_space_v2_s8_o, TOP_st_qualifier_space_v2_u8_o, 
	TOP_st_qualifier_space_v2_s16_o, TOP_st_qualifier_space_v2_u16_o, 
	TOP_st_qualifier_space_v2_s8_b32_o, TOP_st_qualifier_space_v2_u8_b32_o, 
	TOP_st_qualifier_space_v2_s16_b32_o, TOP_st_qualifier_space_v2_u16_b32_o, 
	TOP_st_qualifier_space_v2_s32_o, TOP_st_qualifier_space_v2_u32_o, 
	TOP_st_qualifier_space_v2_s64_o, TOP_st_qualifier_space_v2_u64_o, 
	TOP_st_qualifier_space_v2_f32_o, TOP_st_qualifier_space_v2_f64_o, 
	TOP_st_qualifier_space_v2_s8_a64_o, TOP_st_qualifier_space_v2_u8_a64_o, 
	TOP_st_qualifier_space_v2_s16_a64_o, TOP_st_qualifier_space_v2_u16_a64_o, 
	TOP_st_qualifier_space_v2_s8_b32_a64_o, TOP_st_qualifier_space_v2_u8_b32_a64_o, 
	TOP_st_qualifier_space_v2_s16_b32_a64_o, TOP_st_qualifier_space_v2_u16_b32_a64_o, 
	TOP_st_qualifier_space_v2_s32_a64_o, TOP_st_qualifier_space_v2_u32_a64_o, 
	TOP_st_qualifier_space_v2_s64_a64_o, TOP_st_qualifier_space_v2_u64_a64_o, 
	TOP_st_qualifier_space_v2_f32_a64_o, TOP_st_qualifier_space_v2_f64_a64_o, 
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_stv4o = ISA_Print_Type_Create
	("print_stv4o", "%s%s%s.%s \t[%s+%s], {%s,%s,%s,%s};");
  Name();		// st
  Operand(0);		// qualifier 
  Operand(1);           // space
  Type();		// f32, etc.
  Operand(2);		// r 
  Operand(3);		// o
  Operand(4);		// r1
  Operand(5);		// r2
  Operand(6);		// r3
  Operand(7);		// r4
  Instruction_Print_Group(print_stv4o,
	TOP_st_qualifier_space_v4_s8_o, TOP_st_qualifier_space_v4_u8_o, 
	TOP_st_qualifier_space_v4_s16_o, TOP_st_qualifier_space_v4_u16_o, 
	TOP_st_qualifier_space_v4_s8_b32_o, TOP_st_qualifier_space_v4_u8_b32_o, 
	TOP_st_qualifier_space_v4_s16_b32_o, TOP_st_qualifier_space_v4_u16_b32_o, 
	TOP_st_qualifier_space_v4_s32_o, TOP_st_qualifier_space_v4_u32_o, 
	TOP_st_qualifier_space_v4_f32_o,
	TOP_st_qualifier_space_v4_s8_a64_o, TOP_st_qualifier_space_v4_u8_a64_o, 
	TOP_st_qualifier_space_v4_s16_a64_o, TOP_st_qualifier_space_v4_u16_a64_o, 
	TOP_st_qualifier_space_v4_s8_b32_a64_o, TOP_st_qualifier_space_v4_u8_b32_a64_o, 
	TOP_st_qualifier_space_v4_s16_b32_a64_o, TOP_st_qualifier_space_v4_u16_b32_a64_o, 
	TOP_st_qualifier_space_v4_s32_a64_o, TOP_st_qualifier_space_v4_u32_a64_o, 
	TOP_st_qualifier_space_v4_f32_a64_o,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvt = ISA_Print_Type_Create
	("print_cvt", "%s \t%s, %s;");
  Name();
  Result(0);		// r1
  Operand(0);		// r2
  Instruction_Print_Group(print_cvt,
	TOP_cvt_s16_s8,
	TOP_cvt_s16_u8,
	TOP_cvt_u16_u8,
	TOP_cvt_s32_s8, TOP_cvt_s32_s16,
	TOP_cvt_s32_u8, TOP_cvt_s32_u16,
	TOP_cvt_u32_u8, TOP_cvt_u32_u16,
	TOP_cvt_s64_s8, TOP_cvt_s64_s16, TOP_cvt_s64_s32,
	TOP_cvt_s64_u8, TOP_cvt_s64_u16, TOP_cvt_s64_u32,
	TOP_cvt_u64_u8, TOP_cvt_u64_u16, TOP_cvt_u64_u32,
	TOP_cvt_s16_s8_b32,
	TOP_cvt_s16_u8_b32,
	TOP_cvt_u16_u8_b32,
	TOP_cvt_s16_s8_b64,
	TOP_cvt_s16_u8_b64,
	TOP_cvt_u16_u8_b64,
	TOP_cvt_s32_s8_b64, TOP_cvt_s32_u8_b64, 
	TOP_cvt_s32_s16_b64, TOP_cvt_s32_u16_b64, 
	TOP_cvt_u32_u8_b64, 
	TOP_cvt_u32_u16_b64, 
	TOP_clz_b32, TOP_clz_b64, TOP_popc_b32, TOP_popc_b64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvt_sat = ISA_Print_Type_Create
	("print_cvt_sat", "%s%s.%s \t%s, %s;");
  Name();
  Operand(0);		// sat
  Type();
  Result(0);		// r1
  Operand(1);		// r2
  Instruction_Print_Group(print_cvt_sat,
	TOP_cvt_sat_s8_s16, TOP_cvt_sat_s8_s32, TOP_cvt_sat_s8_s64,
	TOP_cvt_sat_s8_u8, TOP_cvt_sat_s8_u16, TOP_cvt_sat_s8_u32, TOP_cvt_sat_s8_u64,
	TOP_cvt_sat_u8_s8, TOP_cvt_sat_u8_s16, TOP_cvt_sat_u8_s32, TOP_cvt_sat_u8_s64,
	TOP_cvt_sat_u8_u16, TOP_cvt_sat_u8_u32, TOP_cvt_sat_u8_u64,
	TOP_cvt_sat_s16_s32, TOP_cvt_sat_s16_s64,
	TOP_cvt_sat_s16_u16, TOP_cvt_sat_s16_u32, TOP_cvt_sat_s16_u64,
	TOP_cvt_sat_u16_s8, TOP_cvt_sat_u16_s16, TOP_cvt_sat_u16_s32, TOP_cvt_sat_u16_s64,
	TOP_cvt_sat_u16_u32, TOP_cvt_sat_u16_u64,
	TOP_cvt_sat_s32_s64,
	TOP_cvt_sat_s32_u32, TOP_cvt_sat_s32_u64,
	TOP_cvt_sat_u32_s8, TOP_cvt_sat_u32_s16, TOP_cvt_sat_u32_s32, TOP_cvt_sat_u32_s64,
	TOP_cvt_sat_u32_u64,
	TOP_cvt_sat_s64_u64,
	TOP_cvt_sat_u64_s8, TOP_cvt_sat_u64_s16, TOP_cvt_sat_u64_s32, TOP_cvt_sat_u64_s64,
	TOP_cvt_sat_s8_s16_b32, TOP_cvt_sat_s8_s32_b32,
	TOP_cvt_sat_s8_u8_b32, TOP_cvt_sat_s8_u16_b32, TOP_cvt_sat_s8_u32_b32,
	TOP_cvt_sat_u8_s8_b32, TOP_cvt_sat_u8_s16_b32, TOP_cvt_sat_u8_s32_b32,
	TOP_cvt_sat_u8_u16_b32, TOP_cvt_sat_u8_u32_b32,
	TOP_cvt_sat_s16_s32_b32,
	TOP_cvt_sat_s16_u16_b32, TOP_cvt_sat_s16_u32_b32,
	TOP_cvt_sat_u16_s8_b32, TOP_cvt_sat_u16_s16_b32, TOP_cvt_sat_u16_s32_b32,
	TOP_cvt_sat_u16_u32_b32,
	TOP_cvt_sat_s8_s16_b64, TOP_cvt_sat_s8_s32_b64, TOP_cvt_sat_s8_s64_b64,
	TOP_cvt_sat_s8_u8_b64, TOP_cvt_sat_s8_u16_b64, 
	TOP_cvt_sat_s8_u32_b64, TOP_cvt_sat_s8_u64_b64,
	TOP_cvt_sat_u8_s8_b64, TOP_cvt_sat_u8_s16_b64, 
	TOP_cvt_sat_u8_s32_b64, TOP_cvt_sat_u8_s64_b64,
	TOP_cvt_sat_u8_u16_b64, TOP_cvt_sat_u8_u32_b64, TOP_cvt_sat_u8_u64_b64,
	TOP_cvt_sat_s16_s32_b64, TOP_cvt_sat_s16_s64_b64,
	TOP_cvt_sat_s16_u16_b64, 
	TOP_cvt_sat_s16_u32_b64, TOP_cvt_sat_s16_u64_b64,
	TOP_cvt_sat_u16_s8_b64, TOP_cvt_sat_u16_s16_b64, 
	TOP_cvt_sat_u16_s32_b64, TOP_cvt_sat_u16_s64_b64,
	TOP_cvt_sat_u16_u32_b64, TOP_cvt_sat_u16_u64_b64,
	TOP_cvt_sat_s32_u32_b64, TOP_cvt_sat_u32_s8_b64, TOP_cvt_sat_u32_s16_b64, TOP_cvt_sat_u32_s32_b64, 
	TOP_cvt_sat_s32_s64_b64, TOP_cvt_sat_s32_u64_b64, 
	TOP_cvt_sat_u32_s64_b64, TOP_cvt_sat_u32_u64_b64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvt_fround_ftz_sat = ISA_Print_Type_Create
	("print_cvt_fround_ftz_sat", "%s%s%s%s.%s \t%s, %s;");
  Name();
  Operand(0);		// fround
  Operand(1);		// ftz
  Operand(2);		// sat
  Type();
  Result(0);		// r1
  Operand(3);		// r2
  Instruction_Print_Group(print_cvt_fround_ftz_sat,
	TOP_cvt_fround_ftz_sat_s8_f32,
	TOP_cvt_fround_ftz_sat_u8_f32,
	TOP_cvt_fround_ftz_sat_s16_f32,
	TOP_cvt_fround_ftz_sat_u16_f32,
	TOP_cvt_fround_ftz_sat_s32_f32,
	TOP_cvt_fround_ftz_sat_u32_f32,
	TOP_cvt_fround_ftz_sat_s64_f32,
	TOP_cvt_fround_ftz_sat_u64_f32,
	TOP_cvt_fround_ftz_sat_f32_f32,
	TOP_cvt_fround_ftz_sat_f64_f32, 
	TOP_cvt_fround_ftz_sat_f32_f64,
	TOP_cvt_fround_ftz_sat_s8_f32_b32,
	TOP_cvt_fround_ftz_sat_u8_f32_b32,
	TOP_cvt_fround_ftz_sat_s16_f32_b32,
	TOP_cvt_fround_ftz_sat_u16_f32_b32,
	TOP_cvt_fround_ftz_sat_s8_f32_b64,
	TOP_cvt_fround_ftz_sat_u8_f32_b64,
	TOP_cvt_fround_ftz_sat_s16_f32_b64,
	TOP_cvt_fround_ftz_sat_u16_f32_b64,
	TOP_cvt_fround_ftz_sat_s32_f32_b64,
	TOP_cvt_fround_ftz_sat_u32_f32_b64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvt_fround_sat = ISA_Print_Type_Create
	("print_cvt_fround_sat", "%s%s%s.%s \t%s, %s;");
  Name();
  Operand(0);		// fround
  Operand(1);		// sat
  Type();
  Result(0);		// r1
  Operand(2);		// r2
  Instruction_Print_Group(print_cvt_fround_sat,
	TOP_cvt_fround_sat_s8_f64,
	TOP_cvt_fround_sat_u8_f64,
	TOP_cvt_fround_sat_s16_f64,
	TOP_cvt_fround_sat_u16_f64,
	TOP_cvt_fround_sat_s32_f64,
	TOP_cvt_fround_sat_u32_f64,
	TOP_cvt_fround_sat_s64_f64,
	TOP_cvt_fround_sat_u64_f64,
	TOP_cvt_fround_sat_s8_f64_b32,
	TOP_cvt_fround_sat_u8_f64_b32,
	TOP_cvt_fround_sat_s16_f64_b32,
	TOP_cvt_fround_sat_u16_f64_b32,
	TOP_cvt_fround_sat_s8_f64_b64,
	TOP_cvt_fround_sat_u8_f64_b64,
	TOP_cvt_fround_sat_s16_f64_b64,
	TOP_cvt_fround_sat_u16_f64_b64,
	TOP_cvt_fround_sat_s32_f64_b64,
	TOP_cvt_fround_sat_u32_f64_b64,
	TOP_cvt_fround_sat_f32_s8, TOP_cvt_fround_sat_f32_s16,
	TOP_cvt_fround_sat_f32_s32, TOP_cvt_fround_sat_f32_s64,
	TOP_cvt_fround_sat_f32_u8, TOP_cvt_fround_sat_f32_u16,
	TOP_cvt_fround_sat_f32_u32, TOP_cvt_fround_sat_f32_u64,
	TOP_cvt_fround_sat_f64_s8, TOP_cvt_fround_sat_f64_s16,
	TOP_cvt_fround_sat_f64_s32, TOP_cvt_fround_sat_f64_s64,
	TOP_cvt_fround_sat_f64_u8, TOP_cvt_fround_sat_f64_u16,
	TOP_cvt_fround_sat_f64_u32, TOP_cvt_fround_sat_f64_u64,
	TOP_cvt_fround_sat_f64_f64,
	TOP_UNDEFINED);

  /* When converting to and from 16-bit floats ("halves"), we must access them as 32-bit integers, since Open64
     does not handle 16-bit loads.  Also, since the PTX cvt instructions reqquire a .b32 instead of .u32, we
     need to introduce an additional "mov.b32" instruction, along with a temporary .b32 register.  */

  ISA_PRINT_TYPE print_cvt_f32_f16 = ISA_Print_Type_Create
	("print_cvt_f32_f16", "{ .reg .b32 %%b1;\\n\tmov.b32\t\t%%b1, %s;\\n\t%s%s%s%s.%s\t%s, %%b1; }");
  Operand(3);		// r2
  Name();
  Operand(0);		// fround
  Operand(1);		// ftz
  Operand(2);		// sat
  Type();
  Result(0);		// r1
  Instruction_Print_Group(print_cvt_f32_f16,
	TOP_cvt_fround_ftz_sat_f32_f16,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvt_f16_f32 = ISA_Print_Type_Create
	("print_cvt_f16_f32", "{ .reg .b32 %%b1;\\n\t%s%s%s%s.%s\t%%b1, %s;\\n\tmov.b32\t\t%s, %%b1; }");
  Name();
  Operand(0);		// fround
  Operand(1);		// ftz
  Operand(2);		// sat
  Type();
  Operand(3);		// r2
  Result(0);		// r1
  Instruction_Print_Group(print_cvt_f16_f32,
	TOP_cvt_fround_ftz_sat_f16_f32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvta = ISA_Print_Type_Create
	("print_cvta", "%s%s.%s \t%s, %s;");
  Name();
  Operand(0);		// space
  Type();
  Result(0);		// r1
  Operand(1);		// r2
  Instruction_Print_Group(print_cvta,
	TOP_cvta_space_u32, TOP_cvta_space_u64,
	TOP_cvta_space_u32_a, TOP_cvta_space_u64_a,
	TOP_cvta_to_space_u32, TOP_cvta_to_space_u64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_cvtao = ISA_Print_Type_Create
	("print_cvtao", "%s%s.%s \t%s, %s+%s;");
  Name();
  Operand(0);		// space
  Type();
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// ofst
  Instruction_Print_Group(print_cvtao,
	TOP_cvta_space_u32_ao, TOP_cvta_space_u64_ao,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_atomic = ISA_Print_Type_Create(
	"print_atomic", "%s%s.%s \t%s, [%s], %s;");
  Name();
  Operand(0);		// space
  Type();
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Instruction_Print_Group(print_atomic,
	TOP_atom_space_add_s32, TOP_atom_space_add_u32, 
	TOP_atom_space_add_f32, 
	TOP_atom_space_add_s64, TOP_atom_space_add_u64,
	TOP_atom_space_add_f64,
	TOP_atom_space_min_s32, TOP_atom_space_min_u32, 
	TOP_atom_space_min_f32, 
	TOP_atom_space_max_s32, TOP_atom_space_max_u32, 
	TOP_atom_space_max_f32, 
	TOP_atom_space_exch_b32,
	TOP_atom_space_exch_b32_f,
	TOP_atom_space_exch_b64,
	TOP_atom_space_exch_b64_f,
	TOP_atom_space_and_b32,
	TOP_atom_space_or_b32,
	TOP_atom_space_xor_b32,
	TOP_atom_space_inc_u32,
	TOP_atom_space_dec_u32,
	TOP_atom_space_add_s32_a64, TOP_atom_space_add_u32_a64, 
	TOP_atom_space_add_f32_a64, 
	TOP_atom_space_add_s64_a64, TOP_atom_space_add_u64_a64,
	TOP_atom_space_add_f64_a64,
	TOP_atom_space_min_s32_a64, TOP_atom_space_min_u32_a64, 
	TOP_atom_space_min_f32_a64, 
	TOP_atom_space_max_s32_a64, TOP_atom_space_max_u32_a64, 
	TOP_atom_space_max_f32_a64, 
	TOP_atom_space_exch_b32_a64,
	TOP_atom_space_exch_b32_a64_f,
	TOP_atom_space_exch_b64_a64,
	TOP_atom_space_exch_b64_a64_f,
	TOP_atom_space_and_b32_a64,
	TOP_atom_space_or_b32_a64,
	TOP_atom_space_xor_b32_a64,
	TOP_atom_space_inc_u32_a64,
	TOP_atom_space_dec_u32_a64,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_atomic3 = ISA_Print_Type_Create(
	"print_atomic3", "%s%s.%s \t%s, [%s], %s, %s;");
  Name();
  Operand(0);		// space
  Type();
  Result(0);		// r1
  Operand(1);		// r2
  Operand(2);		// r3
  Operand(3);		// r4
  Instruction_Print_Group(print_atomic3,
	TOP_atom_space_cas_b32,
	TOP_atom_space_cas_b32_a64,
	TOP_atom_space_cas_b32_f,
	TOP_atom_space_cas_b32_a64_f,
	TOP_atom_space_cas_b64,
	TOP_atom_space_cas_b64_a64,
	TOP_atom_space_cas_b64_f,
	TOP_atom_space_cas_b64_a64_f,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_vote = ISA_Print_Type_Create(
	"print_vote", "%s \t%s, %s;");
  Name();
  Result(0);		// vote
  Operand(0);		// cond
  Instruction_Print_Group(print_vote,
	TOP_vote_all_pred,
	TOP_vote_any_pred,
	TOP_vote_uni_pred,
	TOP_vote_ballot_b32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_vote_not = ISA_Print_Type_Create(
	"print_vote_not", "%s \t%s, !%s;");
  Name();
  Result(0);		// vote
  Operand(0);		// cond
  Instruction_Print_Group(print_vote_not,
	TOP_vote_all_pred_not,
	TOP_vote_any_pred_not,
	TOP_vote_ballot_b32_not,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_bar_red = ISA_Print_Type_Create(
	"print_bar_red", "%s \t%s, %s, %s;");
  Name();
  Result(0);		// dst
  Operand(0);		// usually zero
  Operand(1);       // pred
  Instruction_Print_Group(print_vote,
	TOP_bar_red_and_pred,
	TOP_bar_red_or_pred,
	TOP_bar_red_popc_u32,
	TOP_UNDEFINED);

  ISA_PRINT_TYPE print_bar_red_not = ISA_Print_Type_Create(
	"print_bar_red_not", "%s \t%s, %s, !%s;");
  Name();
  Result(0);		// dst
  Operand(0);		// usually zero
  Operand(1);       // pred
  Instruction_Print_Group(print_vote,
	TOP_bar_red_and_pred_not,
	TOP_bar_red_or_pred_not,
	TOP_bar_red_popc_u32_not,
	TOP_UNDEFINED);

  ISA_Print_End();
  return 0;
}
