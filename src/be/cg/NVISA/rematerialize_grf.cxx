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
#include <map>
#include "defs.h"
#include "tracing.h"
#include "errors.h"
#include "wn.h"
#include "bb.h"
#include "bb_set.h"
#include "op.h"
#include "tn.h"
#include "cg.h"
#include "cgtarget.h"
#include "whirl2ops.h"
#include "mempool.h"

static BOOL tracing = FALSE;
#define Trace(msg)	if (tracing) fprintf(TFile, msg "\n");

static MEM_POOL remat_pool;

static std::map<TOP, TOP> StLdInstMap;

static void
Create_Store_Load_Instruction_Map (void)
{
  StLdInstMap[TOP_st_qualifier_space_s8_r] = TOP_ld_qualifier_space_s8_r;
  StLdInstMap[TOP_st_qualifier_space_s8_o] = TOP_ld_qualifier_space_s8_o;
  StLdInstMap[TOP_st_qualifier_space_s8_a64_r] = TOP_ld_qualifier_space_s8_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s8_a64_o] = TOP_ld_qualifier_space_s8_a64_o;
  StLdInstMap[TOP_st_qualifier_space_s8_b32_r] = TOP_ld_qualifier_space_s8_b32_r;
  StLdInstMap[TOP_st_qualifier_space_s8_b32_o] = TOP_ld_qualifier_space_s8_b32_o;
  StLdInstMap[TOP_st_qualifier_space_s8_b32_a64_r] = TOP_ld_qualifier_space_s8_b32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s8_b32_a64_o] = TOP_ld_qualifier_space_s8_b32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u8_r] = TOP_ld_qualifier_space_u8_r;
  StLdInstMap[TOP_st_qualifier_space_u8_o] = TOP_ld_qualifier_space_u8_o;
  StLdInstMap[TOP_st_qualifier_space_u8_a64_r] = TOP_ld_qualifier_space_u8_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u8_a64_o] = TOP_ld_qualifier_space_u8_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u8_b32_r] = TOP_ld_qualifier_space_u8_b32_r;
  StLdInstMap[TOP_st_qualifier_space_u8_b32_o] = TOP_ld_qualifier_space_u8_b32_o;
  StLdInstMap[TOP_st_qualifier_space_u8_b32_a64_r] = TOP_ld_qualifier_space_u8_b32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u8_b32_a64_o] = TOP_ld_qualifier_space_u8_b32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_s16_r] = TOP_ld_qualifier_space_s16_r;
  StLdInstMap[TOP_st_qualifier_space_s16_o] = TOP_ld_qualifier_space_s16_o;
  StLdInstMap[TOP_st_qualifier_space_s16_a64_r] = TOP_ld_qualifier_space_s16_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s16_a64_o] = TOP_ld_qualifier_space_s16_a64_o;
  StLdInstMap[TOP_st_qualifier_space_s16_b32_r] = TOP_ld_qualifier_space_s16_b32_r;
  StLdInstMap[TOP_st_qualifier_space_s16_b32_o] = TOP_ld_qualifier_space_s16_b32_o;
  StLdInstMap[TOP_st_qualifier_space_s16_b32_a64_r] = TOP_ld_qualifier_space_s16_b32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s16_b32_a64_o] = TOP_ld_qualifier_space_s16_b32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u16_r] = TOP_ld_qualifier_space_u16_r;
  StLdInstMap[TOP_st_qualifier_space_u16_o] = TOP_ld_qualifier_space_u16_o;
  StLdInstMap[TOP_st_qualifier_space_u16_a64_r] = TOP_ld_qualifier_space_u16_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u16_a64_o] = TOP_ld_qualifier_space_u16_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u16_b32_r] = TOP_ld_qualifier_space_u16_b32_r;
  StLdInstMap[TOP_st_qualifier_space_u16_b32_o] = TOP_ld_qualifier_space_u16_b32_o;
  StLdInstMap[TOP_st_qualifier_space_u16_b32_a64_r] = TOP_ld_qualifier_space_u16_b32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u16_b32_a64_o] = TOP_ld_qualifier_space_u16_b32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_s32_r] = TOP_ld_qualifier_space_s32_r;
  StLdInstMap[TOP_st_qualifier_space_s32_o] = TOP_ld_qualifier_space_s32_o;
  StLdInstMap[TOP_st_qualifier_space_s32_a64_r] = TOP_ld_qualifier_space_s32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s32_a64_o] = TOP_ld_qualifier_space_s32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u32_r] = TOP_ld_qualifier_space_u32_r;
  StLdInstMap[TOP_st_qualifier_space_u32_o] = TOP_ld_qualifier_space_u32_o;
  StLdInstMap[TOP_st_qualifier_space_u32_a64_r] = TOP_ld_qualifier_space_u32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u32_a64_o] = TOP_ld_qualifier_space_u32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_s64_r] = TOP_ld_qualifier_space_s64_r;
  StLdInstMap[TOP_st_qualifier_space_s64_o] = TOP_ld_qualifier_space_s64_o;
  StLdInstMap[TOP_st_qualifier_space_s64_a64_r] = TOP_ld_qualifier_space_s64_a64_r;
  StLdInstMap[TOP_st_qualifier_space_s64_a64_o] = TOP_ld_qualifier_space_s64_a64_o;
  StLdInstMap[TOP_st_qualifier_space_u64_r] = TOP_ld_qualifier_space_u64_r;
  StLdInstMap[TOP_st_qualifier_space_u64_o] = TOP_ld_qualifier_space_u64_o;
  StLdInstMap[TOP_st_qualifier_space_u64_a64_r] = TOP_ld_qualifier_space_u64_a64_r;
  StLdInstMap[TOP_st_qualifier_space_u64_a64_o] = TOP_ld_qualifier_space_u64_a64_o;
  StLdInstMap[TOP_st_qualifier_space_f32_r] = TOP_ld_qualifier_space_f32_r;
  StLdInstMap[TOP_st_qualifier_space_f32_o] = TOP_ld_qualifier_space_f32_o;
  StLdInstMap[TOP_st_qualifier_space_f32_a64_r] = TOP_ld_qualifier_space_f32_a64_r;
  StLdInstMap[TOP_st_qualifier_space_f32_a64_o] = TOP_ld_qualifier_space_f32_a64_o;
  StLdInstMap[TOP_st_qualifier_space_f64_r] = TOP_ld_qualifier_space_f64_r;
  StLdInstMap[TOP_st_qualifier_space_f64_o] = TOP_ld_qualifier_space_f64_o;
  StLdInstMap[TOP_st_qualifier_space_f64_a64_r] = TOP_ld_qualifier_space_f64_a64_r;
  StLdInstMap[TOP_st_qualifier_space_f64_a64_o] = TOP_ld_qualifier_space_f64_a64_o;
}

static OP*
Create_Load_From_Store (OP *store_op)
{
  FmtAssert(OP_store(store_op), ("not a store"));
  Trace("create load from store");
  TOP opc = StLdInstMap[OP_code(store_op)];
  FmtAssert(opc, ("no map for store"));
  OP *load_op = Mk_OP(opc, OP_opnd(store_op,4), 
      OP_opnd(store_op,0), OP_opnd(store_op,1), 
      OP_opnd(store_op,2), OP_opnd(store_op,3));
  Copy_WN_For_Memory_OP (load_op, store_op);
  return load_op;
}

BOOL
OP_is_shared_memory (OP *op)
{
  if (OP_store(op) || OP_load(op)) {
     ISA_ENUM_CLASS_VALUE v = TN_enum(OP_opnd(op,1));
     return ((v == ECV_space_shared) || (v == ECV_space_param));
  } else {
    return FALSE;
  }
}

// Look for st.shared of tn after def of tn, return NULL if not found.
// For now, just look in same bb; else would need to have
// similar Find_Reaching_Def that would look for st.shared.
static OP*
Find_Shared_Store (TN *tn, OP *defop)
{
  OP *op = OP_next(defop);
  while (op != NULL) {
    if (OP_store(op) && OP_is_shared_memory(op) && OP_opnd(op,4) == tn) {
      return op;
    }
    op = OP_next(op);
  }
  return NULL;
}

static BOOL
Safe_To_Rematerialize_Load_In_BB (OP *load_op, BB *bb)
{
  // search whole bb
  return Safe_To_Move_Load (load_op, BB_first_op(bb), BB_last_op(bb));
}

// Check if safe to rematerialize the load,
// by checking each bb between load_bb and use_bb
static BOOL
Safe_To_Rematerialize_Load (OP *load_op, OP *def_op, OP *use_op)
{
  BB *load_bb = OP_bb(def_op);
  BB *use_bb = OP_bb(use_op);
  Is_True(load_bb != use_bb, ("bbs are same?"));
  // search from use to begin of bb
  if ( ! Safe_To_Move_Load (load_op, BB_first_op(use_bb), use_op))
	return FALSE;
  // search from def to end of bb
  if ( ! Safe_To_Move_Load (load_op, OP_next(def_op), BB_last_op(load_bb)))
	return FALSE;
  // now search intermediate bbs
  BB_SET *interim_bbs = Find_Intermediate_BBs (load_bb, use_bb, &remat_pool);
  if (tracing) { 
    fprintf(TFile, "intermediate bbs:  ");
    BB_SET_Print(interim_bbs, TFile);
    fprintf(TFile, "\n");
  }
  BB *bb;
  FOR_ALL_BB_SET_members(interim_bbs, bb) {
    // search whole intermediate bb
    if ( ! Safe_To_Rematerialize_Load_In_BB (load_op, bb))
      return FALSE;
  }
  return TRUE;
}

void
Rematerialize_GRF (void)
{
  BB *bb;
  OP *op;
  TN *tn;
  INT i;

  tracing = Get_Trace(TP_EBO, 0x100);
  MEM_POOL_Initialize (&remat_pool, "remat_pool", FALSE);
  MEM_POOL_Push(&remat_pool);
  Create_Store_Load_Instruction_Map ();

  for (bb = REGION_First_BB; bb != NULL; bb = BB_next(bb)) {
  if (tracing) fprintf(TFile, "rematerialize_grf: bb %d\n", BB_id(bb));
    FOR_ALL_BB_OPs (bb, op) {
      for (i = 0; i < OP_opnds(op); i++) {
	tn = OP_opnd(op,i);
        if (TN_from_shared_load(tn) || TN_from_param_load(tn)) {
	  OP *load_op = NULL;
	  OP *def_op = NULL;
	  INT j;
	  // search for unique reaching def
	  load_op = Find_Reaching_Def (tn, op);
	  if (load_op == NULL) 
		continue;	// no unique def	
	  if (OP_bb(load_op) == bb) 
		continue; 	// leave alone if already in same bb
	  // make sure the reaching def was the shared load
	  // (might be multiple defs)
	  if (!OP_load(load_op)) {
            // if not a shared load, look for shared store after this,
            // from which we can construct a shared load for rematerialization.
            def_op = Find_Shared_Store (tn, load_op);
            if (def_op) {
                Trace("found shared store");
                load_op = Create_Load_From_Store (def_op);
            } else {
		continue;
            }
          }
          else {
            def_op = load_op;
          }
	  if (!OP_is_shared_memory(load_op))
		continue;

	  BOOL one_def = TRUE;
          TN *base_tn = OP_opnd(load_op, OP_find_opnd_use(load_op,OU_base));
	  if (TN_is_register(base_tn)) {
	    // check that base is not redefined before use
	    if ( ! TN_has_one_def(base_tn)) {
		OP *base1_op = Find_Reaching_Def (base_tn, def_op);
		OP *base2_op = Find_Reaching_Def (base_tn, op);
		if (base1_op != NULL && base1_op == base2_op)
			;	// def doesn't change
		else {
			Trace("base defs don't match");
			one_def = FALSE;
		}
	    }
	  }
	  if (one_def) {
            // check if no intervening kill or barrier.
            // ld.param do not have aliasing issues (no st.param),
            // but can still overwrite the register (e.g. increment)
            if ((TN_from_param_load(tn) || TN_from_shared_load(tn)) 
               && Safe_To_Rematerialize_Load (load_op, def_op, op))
            {
                if (tracing) fprintf(TFile, "safe to insert shared load of TN%d at use\n", TN_number(tn));
		OP *new_op = Dup_OP(load_op);
		BB_Insert_Op_Before (bb, op, new_op);
	    }
	  }
	}
        else if (CG_rematerialize_sreg && TN_from_sreg_cvt(tn)) {
          // sregs can also be moved next to use, constant so should not alias
          OP *cvt_op = Find_Reaching_Def(tn, op);
          if (cvt_op == NULL)
            continue;	// no unique def
          if (!TOP_is_cvt_16to32(OP_code(cvt_op)))
            continue;	// would this happen?
          if (OP_bb(cvt_op) == bb)
            continue;	// leave alone if already in same bb

          Trace("safe to insert sreg cvt at use");
          OP *new_op = Dup_OP(cvt_op);
          BB_Insert_Op_Before (bb, op, new_op);
        }
      }
    }
  }
  StLdInstMap.clear();
  MEM_POOL_Pop(&remat_pool);
  MEM_POOL_Delete (&remat_pool);
}
