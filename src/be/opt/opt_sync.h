/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

// ===================================================================
// ===================================================================
//
// Module: opt_sync.cxx
// Date: 2008/06/12
// Author: John A. Stratton
//
// ===================================================================
//
//
//
// ===================================================================
//
// Description: Revome redundant synchronization points from an SPMD
// function.  Initially targeted at the NVIDIA CUDA language, this 
// process models barrier synchronization points and shared memory 
// uses in an analysis similar to dead store elimination.  Hence, we 
// call the class "dead sync elimination" 
//
// ===================================================================
// ===================================================================

#define BIT(n) (0x1 << (n))

#define Sync_Live BIT(0)
#define Sync_Dead BIT(1)
#define Sync_Unkwn BIT(2)

#include "wn.h"
#include "erglob.h"
#include "mempool.h"
#include "wn_util.h"
#include "opt_cfg.h"
#include "opt_bb.h"
#include "opt_sym.h"
#include "opt_base.h"
#include <map>

typedef char LIVE_STATE;

class DSyncE{
private:
  CFG *_cfg; 			// pointer to control flow graph
  OPT_STAB *_opt_stab; 	// Symbol table pointer
  MEM_POOL _loc_pool; 	// Local memory pool
  WN_MAP _live_wns; 		// Map to sync liveness bits
  MAP* _live_bbs; 		// Map to bb liveness bits
  OPT_STAB* opt_stab;		// optimizer symbol table
  // Access functions
  CFG* Cfg() {return _cfg;}

  void SetState(WN* stmt, LIVE_STATE val);
  LIVE_STATE GetState(WN* stmt);
  void SetBBState(BB_NODE* bb, LIVE_STATE val);
  LIVE_STATE GetBBState(BB_NODE* bb);

  LIVE_STATE Propogate_Liveness(WN* sync, BB_NODE* bb);
  BOOL Has_Shared_Or_Global_Mem_Access(WN* root); 
  OPT_STAB* Opt_stab() { return _opt_stab;}

  // typedefs for mempool allocators
  typedef mempool_allocator<std::pair<BB_NODE * const, BOOL> > bbnode_bool_alloc_t;
  typedef mempool_allocator<std::pair<WN * const, BOOL> > wn_bool_alloc_t;
  
  // BB local bit vectors. Does this BB have shared
  // memory read/write
  std::map<BB_NODE *, BOOL, std::less<BB_NODE *>, bbnode_bool_alloc_t> shmem_r, shmem_w;

  // Forward analysis bit vectors.
  std::map<BB_NODE *, BOOL, std::less<BB_NODE *>, bbnode_bool_alloc_t> f_nr, f_nw, f_xr, f_xw;

  // backward analysis bit vectors.
  std::map<BB_NODE *, BOOL, std::less<BB_NODE *>, bbnode_bool_alloc_t> b_nr, b_nw, b_xr, b_xw;

  std::map<WN *, BOOL, std::less<WN *>, wn_bool_alloc_t> nr, nw, xr, xw;

  void Set_local_sets(BOOL=TRUE);
  void Forward_analysis();
  void Backward_analysis();
  BOOL Check_deadness();
  void Local_deadness(BB_NODE *);
public:
  DSyncE(CFG* cfg, OPT_STAB* opt_stab);//Constructor
  ~DSyncE();			//Cleanup
  void Do_dead_sync_elim();	//Run analysis and transformation
  void Do_dead_sync_elim_new(); //Improved version
  BOOL Tracing() { return FALSE; }
};


