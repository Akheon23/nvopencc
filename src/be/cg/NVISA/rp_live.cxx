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

#include "tn.h"
#include "bb.h"
#include "bitset.h"
#include "rp_live.h"
#include "errors.h"
#include "tn_set.h"

using std::max;

//==========================================================
// BB_Live_Info Members
//==========================================================
//==========================================================

//==========================================================
//  Initialize the class. Allocate most bit-vectors
//==========================================================

void
Bb_Linfo::Init(MEM_POOL *pool)
{ 
  _local_pool   = pool;
  _pass         = 0;
  _kill         = TN_SET_Create_Empty(Last_TN+1, _local_pool);
  _gen          = TN_SET_Create_Empty(Last_TN+1, _local_pool);
  _live_in      = TN_SET_Create_Empty(Last_TN+1, _local_pool);
  _live_out     = TN_SET_Create_Empty(Last_TN+1, _local_pool);
  _pass_through = TN_SET_Create_Empty(Last_TN+1, _local_pool);
}

//==========================================================
//  Get the set for given kind
//==========================================================

TN_SET * 
Bb_Linfo::Set(BB_LIVE_KIND kind)
{
  switch(kind) {
  case BB_LIVE_KILL:
    return _kill;
  case BB_LIVE_GEN:
    return _gen;
  case BB_LIVE_IN:
    return _live_in;
  case BB_LIVE_OUT:
    return _live_out;
  case BB_LIVE_PASS_THROUGH:
    return _pass_through;   
  case BB_LIVE_TEMP:
    if (!Is_Flag(LINFO_TEMP_ALLOCATED))
      _temp = TN_SET_Create_Empty(Last_TN+1, _local_pool);
    return _temp;
  default:
    FmtAssert(FALSE, ("Illegal set kind in rp_live"));
  }
  return NULL;
}

//==========================================================
//  Update the set of given kind
//==========================================================

void
Bb_Linfo::Update_Set(BB_LIVE_KIND kind, TN_SET* tns)
{
  switch(kind) {
  case BB_LIVE_KILL:
    _kill = tns; 
    break;
  case BB_LIVE_GEN:
    _gen = tns; 
    break;
  case BB_LIVE_IN:
    _live_in = tns;
    break;
  case BB_LIVE_OUT:
    _live_out = tns;
    break;
  case BB_LIVE_PASS_THROUGH:
    _pass_through = tns;
    break;
  case BB_LIVE_TEMP:
    Set_Flag(LINFO_TEMP_ALLOCATED);
    _temp = tns;
    break;
  default:
    FmtAssert(FALSE, ("Illegal set kind in rp_live"));
  }
  return;
}


//==========================================================
// Rp_Liveness  Members
//==========================================================
//==========================================================


//==========================================================
//  Initialize the class
//==========================================================

void 
Rp_Liveness::Init(MEM_POOL* pool)
{
  _local_pool = pool;

  // Allocate a vector for liveness information
  Bb_Linfo info;
  linfo = Bb_Linfo_VECTOR(PU_BB_Count + 2, info,
                              Bb_Linfo_VECTOR::allocator_type(Pool()));

  // Initialize live info for each block

  // do this separately since cannot call constructor
  // with arguments above
  for (int i = 0; i < PU_BB_Count + 2; i++)
    linfo[i].Init(Pool());

  Set_Flag(LIVENESS_INITIALIZED);
}

//==========================================================
//  Set that a block's local information is stale
//==========================================================

void 
Rp_Liveness::Set_Local_Info_Stale(BB *bb) 
{ 
  linfo[BB_id(bb)].Clear_Flag(LINFO_LOCAL_VALID); 
  Set_Flag(LIVENESS_STALE);
}

//==========================================================
//  Print the liveness information of all blocks in PU
//==========================================================

void
Rp_Liveness::Print(FILE *file)
{
  BB *bb;

  fprintf(file, "LIVE-INFO ==%d=========START==============\n", Pass());

  for (bb = REGION_First_BB; bb != NULL; bb = BB_next(bb)) {

    if (Is_Flag(LIVENESS_BUILT)) {
      fprintf(file,"\n\n--- BB(%d) -----------------\n",BB_id(bb));
      Bb_Linfo *bb_info = Live_Info(bb);

      if (Is_Flag(LIVENESS_MAX_LIVES)) {
        INT32 max_live = bb_info->Max_Live();
        INT32 pass_through = TN_SET_Size(bb_info->Set(BB_LIVE_TEMP));
        fprintf(file,"\nRegister pressure: %d", max_live + pass_through);
        fprintf(file,"\nmax_live         : %d", max_live);
        fprintf(file,"\npass_through     : %d\n", pass_through);
      }

      fprintf(file,"\nlive_in     :");
      TN_SET_Print(bb_info->Set(BB_LIVE_IN), file);
      
      fprintf(file,"\ngen         :");
      TN_SET_Print(bb_info->Set(BB_LIVE_GEN), file);
      
      fprintf(file,"\nkill        :");
      TN_SET_Print(bb_info->Set(BB_LIVE_KILL), file);
      
      fprintf(file,"\nlive_out    :");
      TN_SET_Print(bb_info->Set(BB_LIVE_OUT), file);
      
      fprintf(file,"\n");
    }
    Print_BB(bb);
  }

  fprintf(file, "\n\nLIVE-INFO =============STOP===============\n");
}

//==========================================================
//  Update live out set for block from successors
//  Return true if changed
//==========================================================

BOOL
Rp_Liveness::Update_Live_Out(BB *bb)
{
  Bb_Linfo *bb_info = Live_Info(bb);
  BOOL          changed = FALSE;

  BBLIST *list;
  BB     *succ;
  FOR_ALL_BB_SUCCS (bb, list) {

    succ = BBLIST_item(list);
    Bb_Linfo *succ_info = Live_Info(succ);

    if ( (Pass() == 1) || 
         !TN_SET_ContainsP(bb_info->Set(BB_LIVE_OUT), 
                           succ_info->Set(BB_LIVE_IN))) {
      changed = TRUE;
      // {OUT} = {OUT} U {SUCC_IN}
      bb_info->Update_Set(BB_LIVE_OUT, 
                          TN_SET_UnionD(bb_info->Set(BB_LIVE_OUT),
                                        succ_info->Set(BB_LIVE_IN),
                                        Pool()));
    }
  }
  return changed;
}

//==========================================================
//  Update live-in from live out + local info
//==========================================================

void
Rp_Liveness::Update_Live_In(BB *bb)
{
  Bb_Linfo *bb_info = Live_Info(bb);

  // {IN} = {OUT}
  bb_info->Update_Set(BB_LIVE_IN, 
                      TN_SET_CopyD(bb_info->Set(BB_LIVE_IN),
                                   bb_info->Set(BB_LIVE_OUT),
                                   Pool()));
  // {IN} = { {IN}-{KILL} } U {GEN}
  bb_info->Update_Set(BB_LIVE_IN, 
                      TN_SET_UnionD(TN_SET_DifferenceD(bb_info->Set(BB_LIVE_IN),
                                                       bb_info->Set(BB_LIVE_KILL)),
                                    bb_info->Set(BB_LIVE_GEN),
                                    Pool()));
}

//==========================================================
//  Do liveness computation for 1 instruction
//==========================================================

void 
Rp_Liveness::Transfer_Instruction(BB *bb, OP* op)
{
  INT32 i;
  Bb_Linfo *info = Live_Info(bb);
  // Mark results as kills. gen members might be killed
  for (i = 0; i < OP_results(op); i++) {
    TN *tn = OP_result(op, i);
    if (TN_is_register(tn)) {
      info->Add_To_Set(BB_LIVE_KILL, tn);
      info->Remove_From_Set(BB_LIVE_GEN, tn);
    }
  }

  // Mark sources as gen
  for (i = 0; i < OP_opnds(op); i++) {
    TN *tn = OP_opnd(op, i);
    if (TN_is_register(tn))
      info->Add_To_Set(BB_LIVE_GEN, tn);
  }
}

//==========================================================
//  Do liveness computation for 1 block
//==========================================================

void 
Rp_Liveness::Transfer_Block(BB *bb)
{
  // Recompute local Kill/Gen sets
  if (!Local_Info_Current(bb)) {
    OP *op;
    FOR_ALL_BB_OPs_REV(bb, op) {
      Transfer_Instruction(bb, op);
    }
  }

  // Changed live-out => another pass & changed live-in
  if (Update_Live_Out(bb)) {
    Set_Flag(LIVENESS_CHANGED);
    Update_Live_In(bb);
  }
}

//==========================================================
//  Visit the block in post order
//==========================================================

void 
Rp_Liveness::Dfs_Visit_Block(BB *bb)
{
  Set_Visited(bb);
  
  // Visit all unvisited successors
  BBLIST  *list;
  BB      *succ;
  FOR_ALL_BB_SUCCS (bb, list) {

    succ = BBLIST_item(list);
    if (!Visited(succ))
      Dfs_Visit_Block(succ);
  }

  // Visit current block
  Transfer_Block(bb);
}


//==========================================================
//  Build liveness information
//==========================================================

void 
Rp_Liveness::Build()
{
  FmtAssert(Is_Flag(LIVENESS_INITIALIZED), 
            ("Rp_Liveness: Use without initialization"));

  BB *root = REGION_First_BB;

  // Compute liveness
  Start_Pass();
  do {
    Next_Pass();
    Dfs_Visit_Block(root);
  }while(Is_Flag(LIVENESS_CHANGED));

  Set_Flag(LIVENESS_BUILT);
  Clear_Flag(LIVENESS_STALE);
}

//==========================================================
// Update liveness information, some blocks should be
// marked to have stale information, otherwise do nothing
//==========================================================

void 
Rp_Liveness::Update()
{
  FmtAssert(Is_Flag(LIVENESS_BUILT), 
            ("Rp_Liveness: Update without build live"));

  if (Is_Flag(LIVENESS_STALE))
    Build();
}

//==========================================================
//  Compute max live in a block indicative of register use
//==========================================================

void 
Rp_Liveness::Compute_Max_Lives()
{
  FmtAssert(Is_Flag(LIVENESS_BUILT), 
            ("Rp_Liveness: Max lives without build live"));

  BB *bb;
  for (bb = REGION_First_BB; bb != NULL; bb = BB_next(bb)) {

    Bb_Linfo *info = Live_Info(bb);

    // Compute the pass-through set
    Compute_Pass_Through(info);
    
    // Comptue Max lives in this block
    Compute_Block_Max_Live(bb);
  }
  Set_Flag(LIVENESS_MAX_LIVES);
}

//==========================================================
//  Compute pass through set.
//  {PT} = { OUT INTERSECT IN } - { KILL U GEN }
//==========================================================

void
Rp_Liveness::Compute_Pass_Through(Bb_Linfo *info)
{
  // {PT} <- {OUT}
  info->Update_Set(BB_LIVE_PASS_THROUGH,
                   TN_SET_CopyD(info->Set(BB_LIVE_PASS_THROUGH),
                                info->Set(BB_LIVE_OUT),
                                Pool()));

  // {PT} INTERSECT {IN}
  info->Update_Set(BB_LIVE_PASS_THROUGH,
                   TN_SET_IntersectionD(info->Set(BB_LIVE_PASS_THROUGH),
                                        info->Set(BB_LIVE_IN)));
    
  // {TEMP} = {KILL} U {GEN}
  info->Update_Set(BB_LIVE_TEMP, TN_SET_Union(info->Set(BB_LIVE_KILL),
                                              info->Set(BB_LIVE_GEN),
                                              Pool()));
  // {PT} - {TEMP}
  info->Update_Set(BB_LIVE_PASS_THROUGH,
                   TN_SET_DifferenceD(info->Set(BB_LIVE_PASS_THROUGH),
                                      info->Set(BB_LIVE_TEMP)));

}

//==========================================================
//  Compute the max live values in a block
//==========================================================

void 
Rp_Liveness::Compute_Block_Max_Live(BB *bb)
{
  Bb_Linfo *info = Live_Info(bb);

  // Temp will hold the lives after each instruction

  // {TEMP} <- {OUT}
  info->Update_Set(BB_LIVE_TEMP,
                   TN_SET_CopyD(info->Set(BB_LIVE_TEMP),
                                info->Set(BB_LIVE_OUT),
                                Pool()));
  // {TEMP} - {PASS_THROUGH}
  info->Update_Set(BB_LIVE_TEMP,
                   TN_SET_DifferenceD(info->Set(BB_LIVE_TEMP),
                                      info->Set(BB_LIVE_PASS_THROUGH)));
    
  // Walk block backward to find max
  INT32 i, max_live = 0;
  OP *op;               
  FOR_ALL_BB_OPs_REV(bb, op) {

    // Mark results as kills. gen members might be killed
    for (i = 0; i < OP_results(op); i++) {
      TN *tn = OP_result(op, i);
      if (TN_is_register(tn)) {
        info->Remove_From_Set(BB_LIVE_TEMP, tn);
      }
    }

    // Mark sources as gen
    for (i = 0; i < OP_opnds(op); i++) {
      TN *tn = OP_opnd(op, i);
      if (TN_is_register(tn))
        info->Add_To_Set(BB_LIVE_TEMP, tn);
    }
   
    max_live = max(max_live, TN_SET_Size(info->Set(BB_LIVE_TEMP)));
  }

  info->Set_Max_Live(max_live);
}
