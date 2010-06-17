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

#ifndef RP_LIVE_INCLUDED
#define RP_LIVE_INCLUDED

#include "mempool_allocator.h"
#include "defs.h"
#include "cxx_memory.h"
#include "tn.h"
#include "tn_set.h"
#include "bb.h"
#include <vector>
using std::vector;


typedef enum 
{
  BB_LIVE_KILL  = 0,      // kill set
  BB_LIVE_GEN,            // gen set, upward exposed gens
  BB_LIVE_IN,             // live-in into the block
  BB_LIVE_OUT,            // live-out from the block
  BB_LIVE_PASS_THROUGH,   // pass through, see computation
  BB_LIVE_TEMP            // temporary set for computing fat points
} BB_LIVE_KIND;

enum LINFO_FLAG {
  LINFO_UNINITIALIZED  = 0x00, // uninitialized
  LINFO_INITIALIZED    = 0x01, // initialized, bitvectors allocated
  LINFO_TEMP_ALLOCATED = 0x02, // temporary bitvector allocated
  LINFO_LOCAL_VALID    = 0x04  // local information is valied
};

enum LIVENESS_FLAG {
  LIVENESS_UNINITIALIZED  = 0x00, // uninitialized
  LIVENESS_INITIALIZED    = 0x01, // initialized, bitvectors allocated
  LIVENESS_BUILT          = 0x02, // local liveness information is built
  LIVENESS_MAX_LIVES      = 0x04, // max lives have been calculated
  LIVENESS_STALE          = 0x08, // If any of the blocks is marked with stale info
  LIVENESS_CHANGED        = 0x10  // there was a live information change in
                                  // this dataflow iteration
};

//==========================================================
//  Holds the liveness info associated with 1 block
//==========================================================

class Bb_Linfo 
{
 private:
  LINFO_FLAG      _flags;
  INT32           _pass, _max_live;
  MEM_POOL       *_local_pool;
  TN_SET         *_kill, *_gen;
  TN_SET         *_live_in, *_live_out;
  TN_SET         *_pass_through, *_temp;

 public:
  Bb_Linfo() { _flags = LINFO_UNINITIALIZED; }
  void Init(MEM_POOL *pool);

  void     Clear_Flag(LINFO_FLAG f){ _flags =(LINFO_FLAG) (_flags & ~f); }
  void     Set_Flag(LINFO_FLAG f)  { _flags =(LINFO_FLAG) (_flags |  f); }
  BOOL     Is_Flag(LINFO_FLAG f)   { return _flags & f; }
  INT32    Max_Live()              { return _max_live; }
  INT32    Visit()                 { return _pass; }
  void     Set_Max_Live(INT32 l)   { _max_live = l; }
  void     Set_Visit(INT32 pass)   { _pass = pass; }

  TN_SET * Set(BB_LIVE_KIND kind);
  void     Update_Set(BB_LIVE_KIND kind, TN_SET* tns);
  void     Add_To_Set(BB_LIVE_KIND kind, TN* tn) {
    Update_Set(kind, TN_SET_Union1D(Set(kind), tn, _local_pool)); }
  void     Remove_From_Set(BB_LIVE_KIND kind, TN* tn) {
    Update_Set(kind, TN_SET_Difference1D(Set(kind), tn)); }
};

//==========================================================
//  The liveness class
//==========================================================

class Rp_Liveness 
{
 public:

  typedef mempool_allocator<Bb_Linfo> Bb_Linfo_ALLOCATOR;
  typedef vector<Bb_Linfo, Bb_Linfo_ALLOCATOR> Bb_Linfo_VECTOR;

 private:
  LIVENESS_FLAG  _flags;
  MEM_POOL      *_local_pool;
  UINT32         _pass;

  Bb_Linfo_VECTOR linfo; 

  void       Clear_Flag(LIVENESS_FLAG f) { _flags =(LIVENESS_FLAG) (_flags & ~f); }
  void       Set_Flag(LIVENESS_FLAG f)   { _flags =(LIVENESS_FLAG) (_flags |  f); }
  BOOL       Is_Flag(LIVENESS_FLAG f)    { return _flags & f; }

  // Data/Control helpers
  MEM_POOL*  Pool()              { return _local_pool; }
  INT32      Pass()              { return _pass; }
  BOOL       Visited(BB *bb)     { return linfo[BB_id(bb)].Visit() == Pass(); }
  void       Set_Visited(BB *bb) { linfo[BB_id(bb)].Set_Visit(Pass()); }
  void       Start_Pass()        { _pass = 0; }
  void       Next_Pass()         { _pass++; Clear_Flag(LIVENESS_CHANGED); }
  BOOL       Local_Info_Current(BB *bb) { return linfo[BB_id(bb)].Is_Flag(LINFO_LOCAL_VALID);}
  // Algorithms helpers
  BOOL       Update_Live_Out(BB *bb);
  void       Update_Live_In(BB *bb);
  void       Transfer_Instruction(BB *bb, OP* op);
  void       Transfer_Block(BB *bb);
  void       Dfs_Visit_Block(BB *bb);
  void       Compute_Pass_Through(Bb_Linfo *info);
  void       Compute_Block_Max_Live(BB *bb);

 public:
  Rp_Liveness() { _flags = LIVENESS_UNINITIALIZED; }
  void       Init(MEM_POOL* pool);
  void       Print(FILE *file);
  // Interface
  Bb_Linfo*  Live_Info(BB *bb)   { return  &(linfo[BB_id(bb)]); }
  void       Build();
  void       Compute_Max_Lives();
  void       Update();
  void       Set_Local_Info_Stale(BB *bb);
};

#endif // RP_LIVE_INCLUDED

