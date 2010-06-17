/*
 *  Copyright 2005-2008 NVIDIA Corporation.  All rights reserved.
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
#include "defs.h"
#include "tracing.h"
#include "errors.h"
#include "wn.h"
#include "bb.h"
#include "op.h"
#include "tn.h"
#include "cg.h"
#include "cgtarget.h"

static BOOL tracing = FALSE;
#define Trace(msg)	if (tracing) fprintf(TFile, msg "\n");

static BOOL
OP_is_add_immediate (OP *op)
{
  switch (OP_code(op)) {
  case TOP_add_s8_lit:
  case TOP_add_s16_lit:
  case TOP_add_s32_lit:
  case TOP_add_s64_lit:
  case TOP_add_u8_lit:
  case TOP_add_u16_lit:
  case TOP_add_u32_lit:
  case TOP_add_u64_lit:
    return TRUE;
  default:
    return FALSE;
  }
}

// Fermi allows immediate offsets in addresses, 
// so if reaching_def of address is an add immediate,
// can rematerialize the immediate into the address at no cost,
// thus potentially reducing register pressure
// (if have multiple base+offset pointers that have been hoisted).
void
Rematerialize_Addr (void)
{
  BB *bb;
  OP *op;

  tracing = Get_Trace(TP_EBO, 0x2000);

  for (bb = REGION_First_BB; bb != NULL; bb = BB_next(bb)) {
    if (tracing) fprintf(TFile, "rematerialize_addr: bb %d\n", BB_id(bb));
    FOR_ALL_BB_OPs (bb, op) {
      INT base_num = OP_find_opnd_use(op,OU_base);
      INT offset_num = OP_find_opnd_use(op,OU_offset);
      if (base_num >= 0 && offset_num >= 0
          && TN_is_register(OP_opnd(op,base_num)) )
      {
        // Check that only one def; this is conservative, as we could take
        // any def and then search all intermediate blocks for a redefine,
        // like we do in rematerialize_grf.  But so far the common important
        // cases all have a single def.  May want to redo this later if needed.
        if (!TN_has_one_def(OP_opnd(op,base_num)))
          continue;     // not a unique def
	// search for unique reaching def
        OP *def_op = Find_Reaching_Def (OP_opnd(op,base_num), op);
	if (def_op == NULL) 
	  continue;	// no unique def	
	if (OP_bb(def_op) == bb) 
	  continue; 	// leave alone if already in same bb
        if (OP_is_add_immediate(def_op)) {
          TN *base_tn = OP_opnd(def_op, OP_find_opnd_use(def_op,OU_opnd1));
          TN *immed_tn = OP_opnd(def_op, OP_find_opnd_use(def_op,OU_opnd2));
          FmtAssert(TN_has_value(immed_tn), ("no immediate"));
          FmtAssert(TN_has_value(OP_opnd(op,offset_num)), ("no immediate"));
          // add immediates together
          INT64 val = TN_value(immed_tn) + TN_value(OP_opnd(op,offset_num));
          TN *new_tn = Gen_Literal_TN_Ex (val);
          if (tracing) {
	    fprintf(TFile, "rematerialize add into address\n");
            Print_OP_No_SrcLine(def_op);
            Print_OP_No_SrcLine(op);
          }
          Set_OP_opnd(op, base_num, base_tn); 
          Set_OP_opnd(op, offset_num, new_tn); 
	}
      }
    }
  }
}
