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

#include "defs.h"
#include "wn.h"
#include "cg.h"
#include "cg_internal.h"
#include "cg_flags.h"
#include "config.h"
#include "config_list.h"
#include "gtn_universe.h"
#include "tn_set.h"
#include "gtn_tn_set.h"
#include "region_util.h"
#include "gra.h"
#include "gra_live.h"

// Calculate the maximum number of TNs that are livein to a
// basic block.
INT Calculate_Maxlivein (void)
{
  // Compute liveness info
  INT max_livein = 0;
  BB *t_bb;
  
  FOR_ALL_BBLIST_ITEMS(REGION_First_BB, t_bb) {
    GTN_SET *lin_set = BB_live_in(t_bb);
    TN *temp_tn;
    INT sz = 0;
    INT sz16 = 0;
    INT sz64 = 0;
    // Go through the live TN set and remove non-registers
    // and predicates.
    for(temp_tn=GTN_SET_Choose(lin_set);
        temp_tn != GTN_SET_CHOOSE_FAILURE;
        temp_tn=GTN_SET_Choose_Next(lin_set, temp_tn)) {
      if(!TN_is_register(temp_tn))
        continue;
      if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_predicate)
        continue;
      if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_integer64)
        sz64++;
      else if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_float64)
        sz64++;
      else if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_integer16)
        sz16++;
      else 
        sz++;
    }
    sz += sz64*2;
    if(Target_ISA < TARGET_ISA_compute_20) {
      sz += sz16/2;
      sz += sz16%2;
    }
    else
      sz += sz16;
    if(max_livein < sz) {
      max_livein = sz;
    }
  }
  return max_livein;
}

INT NVISA_TN_SET_Size(TN_SET *s)
{
  TN *temp_tn;
  INT sz = 0;
  INT sz16 = 0;
  INT sz64 = 0;
  // Go through the live TN set and remove non-registers
  // and predicates.

  if(TN_SET_EmptyP(s))
    return 0;
    
  for(temp_tn=TN_SET_Choose(s);
      temp_tn != TN_SET_CHOOSE_FAILURE && temp_tn != NULL;
      temp_tn=TN_SET_Choose_Next(s, temp_tn)) {
    if(!TN_is_register(temp_tn))
      continue;
    if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_predicate)
      continue;
    if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_integer64)
      sz64++;
    else if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_float64)
      sz64++;
    else if(TN_register_class(temp_tn) == ISA_REGISTER_CLASS_integer16)
      sz16++;
    else 
      sz++;
  }
  sz += sz64*2;
  if(Target_ISA < TARGET_ISA_compute_20) {
    sz += sz16/2;
    sz += sz16%2;
  }
  else
    sz += sz16;
  return sz;
}

// For debugging purposes
void cg_print_gtn_set(GTN_SET *s)
{
  GTN_TN_SET_Print(s, stderr);
  fprintf(stderr, "\n");
}

void cg_print_tn_set(TN_SET *s)
{
  TN_SET_Print(s, stderr);
  fprintf(stderr, "\n");
}

INT Calculate_BB_Fatpoint(BB *bb, MEM_POOL *pool)
{
  FmtAssert(GRA_LIVE_Phase_Invoked, ("gra_live not available"));
  GTN_SET *lout_set_orig = BB_live_out(bb);
  TN_SET *lout_set = TN_SET_Create_Empty(Last_TN+1, pool);
  if(!GTN_SET_EmptyP(lout_set_orig)) {
    TN *temp_tn;
    for(temp_tn=GTN_SET_Choose(lout_set_orig);
        temp_tn != GTN_SET_CHOOSE_FAILURE && temp_tn != NULL;
        temp_tn=GTN_SET_Choose_Next(lout_set_orig, temp_tn)) {
      TN_SET_Union1D(lout_set, temp_tn, pool);
    }
  }

  INT local_max_size = 0;
  OP *t_op;
  FOR_ALL_BB_OPs_REV(bb, t_op) {
    for(unsigned i=0, e=OP_results(t_op); i!=e; ++i)
      TN_SET_Difference1D(lout_set, OP_result(t_op, i));
    for(unsigned i=0, e=OP_opnds(t_op); i!=e; ++i) {
      if(!TN_is_register(OP_opnd(t_op, i))) continue;
      TN_SET_Union1D(lout_set, OP_opnd(t_op, i), pool);
    }
    INT size_here = NVISA_TN_SET_Size(lout_set);
    if(size_here > local_max_size)
      local_max_size = size_here;
  }

  return local_max_size;
}

// Calculate the maximum number of TNs that are live at 
// a point in a basic block.
INT Calculate_Fatpoint(MEM_POOL *local_pool)
{
  // Compute liveness info
  INT max_fat = 0;
  BB *t_bb;

  int numbbs = 0;
  FOR_ALL_BBLIST_ITEMS(REGION_First_BB, t_bb) {
    ++numbbs;
  }

  FOR_ALL_BBLIST_ITEMS(REGION_First_BB, t_bb) {
    MEM_POOL_Popper mp(local_pool);

    INT local_max_size = Calculate_BB_Fatpoint(t_bb, mp.Pool());
    if(local_max_size > max_fat)
      max_fat = local_max_size;
  }
  return max_fat;
}
