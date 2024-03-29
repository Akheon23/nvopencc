/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/*
 * Copyright 2003, 2004, 2005 PathScale, Inc.  All Rights Reserved.
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


/* ====================================================================
 * ====================================================================
 *
 *  Target Specific Miscellany Declarations which include target
 *  dependencies.
 *
 *  THIS FILE IS ONLY TO BE INCLUDE BY ../cgtarget.h!!!!
 *
 * ====================================================================
 * ====================================================================
 */
#ifdef KEY
#include "config_opt.h"
#endif

#define INST_BYTES 4
#define DEFAULT_LONG_BRANCH_LIMIT (0x1ffffff * INST_BYTES)
#define DEFAULT_BRP_BRANCH_LIMIT DEFAULT_LONG_BRANCH_LIMIT
#define MIN_BRANCH_DISP (2097151 * INST_BYTES)


inline INT32 CGTARG_Branch_Taken_Penalty(void)
{
  return CGTARG_branch_taken_penalty_overridden ?
    CGTARG_branch_taken_penalty : 1;
}

inline INT32 CGTARG_Issue_Width(void)
{
  return 9;
}

inline BOOL 
CGTARG_Use_Brlikely(float branch_taken_probability)
{
  return FALSE;
}

inline BOOL
CGTARG_Can_Predicate_Calls()
{
  return FALSE;
}


inline BOOL
CGTARG_Can_Predicate_Returns()
{
  return FALSE;
}

inline BOOL
CGTARG_Can_Predicate_Branches()
{
  return FALSE;
}

inline INT
CGTARG_Text_Alignment (void)
{
  return 4;
}

inline TOP
CGTARG_Noop_Top(void)
{
  return TOP_nop;	// return simulated nop
}

inline TOP
CGTARG_Noop_Top(ISA_EXEC_UNIT_PROPERTY unit)
{
  return CGTARG_Noop_Top();
}

inline TOP
CGTARG_Simulated_Top(OP *op, ISA_EXEC_UNIT_PROPERTY unit)
{

  TOP top = OP_code(op);
 
  /* Prune the obvious cases. */
  if (!TOP_is_simulated(top)) return top;

  /* Placeholder for itemizing specific simulated ops */
  return top;
}

/*
 *     TN *CGTARG_Return_Enum_TN(OP *op, ISA_ENUM_CLASS class, INT *pos)
 *     if 'op' has an enum operand which matches the ISA_ENUM_CLASS 
 *     <class> type, return the enum TN and its operand position.
 */
inline TN* 
CGTARG_Return_Enum_TN(OP *op, ISA_ENUM_CLASS_VALUE enum_class, INT *pos)
{
  INT i;

  for (i = 0; i < OP_opnds(op); ++i) {
    TN *opnd_tn = OP_opnd(op, i);
    if (TN_is_enum(opnd_tn) && (enum_class == TN_enum(opnd_tn))) {
      *pos = i;
      return opnd_tn;
    }
  }

  return NULL;
}

inline BOOL 
CGTARG_Is_OP_Barrier(OP *op)
{
  if (OP_call(op)) {
    return TRUE;
  }
  switch (OP_code(op)) {
  case TOP_asm:
    {
    extern OP_MAP OP_Asm_Map;
    ASM_OP_ANNOT* asm_info = (ASM_OP_ANNOT*) OP_MAP_Get(OP_Asm_Map, op);
#ifdef KEY
    return (WN_Asm_Clobbers_Mem(ASM_OP_wn(asm_info)) || Asm_Memory);
#else
    return (WN_Asm_Clobbers_Mem(ASM_OP_wn(asm_info)));
#endif
    }
  case TOP_bar_sync:
  case TOP_membar_gl:
  case TOP_membar_cta:
  case TOP_pmevent:
    return TRUE;
  default:
    return FALSE;
  }
}

inline BOOL
CGTARG_Is_OP_Intrinsic(OP *op)
{
  return OP_code(op) == TOP_intrncall;
}

inline BOOL
CGTARG_Is_Bad_Shift_Op (OP *op)
{
  return FALSE;
}

inline BOOL
CGTARG_Is_Right_Shift_Op (OP *op)
{
  switch (OP_code(op)) {
  TOP_shr:
    return TRUE;
  }
  return FALSE;
}
    
inline BOOL 
CGTARG_Is_OP_Addr_Incr(OP *op)
{
  const TOP top = OP_code(op);
  return FALSE;
}

inline BOOL
CGTARG_Can_daddu_Be_Folded(OP *op1, OP *op2)
{
  return FALSE;
}

inline TOP
CGTARG_Copy_Op(UINT8 size, BOOL is_float)
{
  if (is_float) {
    return size == 4 ? TOP_mov_f32 : TOP_mov_f64;
  }
  switch (size) {
  case 1: return TOP_mov_s8;
  case 2: return TOP_mov_s16;
  case 4: return TOP_mov_s32;
  case 8: return TOP_mov_s64;
  }
}

inline BOOL
CGTARG_Have_Indexed_Mem_Insts(void)
{
  return FALSE;
}

inline BOOL
Has_Single_FP_Condition_Code (void)
{
  return FALSE;
}

inline BOOL
CGTARG_Is_OP_daddu(OP *op)
{
  return FALSE;
}

inline BOOL
CGTARG_Can_Predicate()
{
  return FALSE;
}

extern TOP CGTARG_Get_unc_Variant(TOP top);
extern void Make_Branch_Conditional(BB *bb);

inline BOOL
CGTARG_Has_Branch_Predict(void)
{
  return TRUE;
}

// get qualifier from memory op
inline TN*
OP_Qualifier_TN (OP *op)
{
        return OP_opnd(op,0);
}
// get space from memory op
inline TN*
OP_Space_TN (OP *op)
{
        return OP_opnd(op,1);
}

// iterate thru code, assigning virtual registers
extern void Assign_Virtual_Registers (void);
extern void Assign_Virtual_Register (TN *tn);

// the last register allocated tells you how many were used
extern REGISTER Last_Reg_Allocated (ISA_REGISTER_CLASS);

// return appropriate literal class given a mtype
extern ISA_LIT_CLASS Lit_Class_For_Mtype (TYPE_ID mtype);

// can use constant value of tn;
// value is returned in *val, 
// return boolean is whether constant value exists.
extern BOOL TN_Can_Use_Constant_Value (TN *tn, TYPE_ID mtype, INT64 *val);

// try to create unique def for tns,
// catching cases where multiple independent defs.
extern void Create_Unique_Defs_For_TNs (void);

// find unique definition of tn that reaches use_op
extern OP* Find_Reaching_Def (TN *tn, OP *use_op);

// check if the operation touches shared memory
extern BOOL OP_is_shared_memory (OP *op);

// iterate thru code, rematerializing grf loads where appropriate
extern void Rematerialize_GRF (void);
// iterate thru code, rematerializing address immediates where appropriate
extern void Rematerialize_Addr (void);

// Remove unnecessary type conversion instructions
extern void Remove_Typeconv (void);

// optimize away copies by removing later usage of src opnd
extern void Optimize_Copy_Usage (void);

// replace 32bit ops with 16bit ops when safe to do so
extern void Use_16bit_Ops (void);

// whether two memory ops alias each other
extern BOOL Memory_Ops_Alias (OP *op1, OP *op2);

// whether is safe to move load across first_op..last_op range
extern BOOL Safe_To_Move_Load (OP *load_op, OP *first_op, OP *last_op);

// return set of bbs between start and end
extern BB_SET* Find_Intermediate_BBs (BB *start, BB *end, MEM_POOL *pool);

// return fatpoint for bb
extern INT Calculate_BB_Fatpoint (BB *bb, MEM_POOL *pool);
// return fatpoint for whole pu 
extern INT Calculate_Fatpoint (MEM_POOL *local_pool);
// return max #livein tns
extern INT Calculate_Maxlivein (void);
