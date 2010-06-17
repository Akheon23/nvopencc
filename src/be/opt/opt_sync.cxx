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
// process models barrier synchronization points and global/shared memory 
// uses in an analysis similar to dead store elimination.  Hence, we 
// call the class "dead sync elimination" 
//
// ===================================================================
// ===================================================================

//
// Modified by Manjunath Kudlur.
//
// Description : The improved method performs a forward data flow analysis
// followed by a backward data flow analysis. The bit vectors (really one bit,
// to indicate shared/global memory reads/writes) f_* are used in the forward analysis
// and b_* used in the backward analysis. f_n* store what shared/global memory accesses
// reach the entry of a basic block and f_x* store what shared/global memory accesses
// leave the basic block. b_x* and b_n* play the symmetrically opposite
// roles. After the data flow analyses, we have what shared/global memory accesses
// reach a syncthread from above, and what reach it from below. Let's denote
// reads by R, writes by W, and none by X. In the following cases, a syncthread
// can be removed :
// 
// X  sync X  <--
// X  sync R  <--
// X  sync W  <--
// X  sync RW <--
// R  sync X  <--
// R  sync R  <--
// R  sync W
// R  sync RW
// W  sync X  <--
// W  sync R
// W  sync W
// W  sync RW
// RW sync X  <--
// RW sync R
// RW sync W
// RW sync RW
// 
// Do_dead_sync_elim_new performs the analysis, finds one the cases above where
// a syncthread can be removed, removes it, then rinses and repeats.
// TBD : We should have a better way to choose which syncthread to remove from a
// set of removeable syncthreads. Right now, the choice is arbitrary.
//

#include "opt_sync.h"
#include "errors.h"
#include "erbe.h"
#include "erglob.h"
#include "glob.h"
#include "intrn_info.h"

// ==================================================================
// Check to see if the intrinsic reads or writes memory
void Check_intrn_effects(INTRINSIC id, BOOL& read, BOOL &write)
{
  if(INTRN_is_pure(id))
    return;
  if(id == INTRN_GETSHAREDMEM) {
    read = TRUE;
    write = TRUE;
    return;
  }
  if(id == INTRN_SYNCTHREADS) {
    return;
  }
  if(INTRN_has_no_side_effects(id)) {
    // Conservatively assume that the intrinsic can read shared state
    read = TRUE;
    return;
  }
  // Conservatively assume read AND write to shared state
  read = TRUE;
  write = TRUE;
}

// ==================================================================
// Get the current summary state of sync liveness at a whirl node

void 
DSyncE::SetState(WN* stmt, LIVE_STATE val)
{
  if(stmt == NULL)
    return;

  LIVE_STATE* state = (LIVE_STATE*)WN_MAP_Get(_live_wns, stmt);
        
  if(state == NULL)
  {
    state = CXX_NEW(LIVE_STATE[1], &_loc_pool);
    WN_MAP_Set(_live_wns, stmt, state);
  }

  *state = val;
}

// ==================================================================
// Get the current summary state of sync liveness at a whirl node

LIVE_STATE 
DSyncE::GetState(WN* stmt)
{
  if(stmt == NULL)
    return Sync_Unkwn;

  LIVE_STATE* state = (LIVE_STATE*)WN_MAP_Get(_live_wns, stmt);

  if(state != NULL)
    return *state;
        
  SetState(stmt, Sync_Unkwn);
  return Sync_Unkwn;
}


// ==================================================================
// Set the current summary state of sync liveness at the exit of a bb

void
DSyncE::SetBBState(BB_NODE* bb, LIVE_STATE val)
{
  LIVE_STATE * old_state =
    (LIVE_STATE*) _live_bbs->Get_val((POINTER)(INTPTR)bb);

  if(old_state == NULL)
  {
    old_state = CXX_NEW(LIVE_STATE, &_loc_pool);
    _live_bbs->Add_map(((POINTER)(INTPTR)bb), old_state);
  }

  *old_state = val;
}


// ==================================================================
// Get the current summary state of sync liveness at the exit of a bb

LIVE_STATE
DSyncE::GetBBState(BB_NODE* bb)
{
  LIVE_STATE * state = (LIVE_STATE*)_live_bbs->Get_val((POINTER)(INTPTR)bb);

  if(state == NULL)
    return Sync_Unkwn;

  return *state;
}

// ===================================================================
//
// Description:
//
// TRUE if the whirl node is a __syncthread() intrinsic, 
// FALSE otherwise
//
// ===================================================================
static BOOL 
Is_syncthreads(WN * wn)
{
  FmtAssert(wn != NULL, ("null node tested for __synchthreads()"));

  if (WN_operator(wn) == OPR_INTRINSIC_CALL) {
    return (WN_intrinsic(wn) == INTRN_SYNCTHREADS);
  }
  return FALSE;
}

BOOL 
DSyncE::Has_Shared_Or_Global_Mem_Access(WN * root) 
{
  for (WN_ITER* wni = WN_WALK_TreeIter(root);
       wni != NULL;
       wni = WN_WALK_TreeNext(wni)) 
  {
    WN* addr = WN_ITER_wn(wni);
    OPERATOR opr = WN_operator(addr);
    if ((OPERATOR_is_load(opr) == FALSE) &&
	(OPERATOR_is_store(opr)== FALSE) &&
	(OPERATOR_is_call(opr) == FALSE)) {
      return FALSE;
    }
    switch(opr)
    {
      case OPR_MSTORE:
      case OPR_ISTORE:
      case OPR_MLOAD:
      case OPR_ILOAD:
        //Assume worst case
        return TRUE;
      case OPR_LDID:
      case OPR_STID:
      case OPR_LDBITS:
      case OPR_STBITS:
        //Check for shared memory access
        ST* real_sym;
        //Check the auxillary symbol table if necessary
        if(WN_has_aux(addr))
          real_sym = Opt_stab()->St_ptr(addr);
        else
          real_sym = WN_st(addr);

        if(real_sym == NULL)
          return TRUE;
        if(ST_in_shared_mem(real_sym))
          return TRUE;
        // parameter memory is currently implemented in shared memory
        if (ST_in_param_mem(real_sym))
          return TRUE;
        if (ST_in_global_mem(real_sym))
          return TRUE;
        break;
      //Cover the COMIC case, where smem is represented by intrinsics
      case OPR_INTRINSIC_OP:
      case OPR_INTRINSIC_CALL:
      {
        BOOL r = FALSE, w = FALSE;
        Check_intrn_effects(WN_intrinsic(addr), r, w);
        if(r || w)
          return TRUE;
      }
        break;
      // Conservatively assume the following OPRs as black boxes that can touch shared memory
      // TBD : Remove conservative assumption for special cases that we know about
      case OPR_CALL:
      case OPR_VFCALL:
      case OPR_ICALL:
      case OPR_PICCALL:
      case OPR_IO:
      case OPR_ASM_STMT:
        return TRUE;
      default:
        FmtAssert((!OPERATOR_is_load(opr)) && (!OPERATOR_is_store(opr)), ("Not expecting load or store here."));
        break;
    }
  }

  return FALSE;
}

// Removes and deletes a wn from a doubly linked list of wns in a bb
void
Remove_WN_from_BB(WN* wn, BB_NODE* bb)
{
  if(wn == NULL)
    return;
  if(bb == NULL) {
    WN_Delete(wn);
    return;
  }

  if(WN_next(wn) != NULL)
    WN_prev(WN_next(wn)) = WN_prev(wn);
  else
    bb->Set_laststmt(WN_prev(wn));

  if(WN_prev(wn) != NULL)
    WN_next(WN_prev(wn)) = WN_next(wn);
  else
    bb->Set_firststmt(WN_next(wn));

  WN_Delete(wn);
}


DSyncE::DSyncE(CFG* cfg, OPT_STAB* opt_stab):
  _cfg(cfg), _opt_stab(opt_stab)
{
  MEM_POOL_Initialize(&_loc_pool, "dead_sync_pool", FALSE);
  MEM_POOL_Push(&_loc_pool);
  _live_wns = WN_MAP_Create(&_loc_pool);
  _live_bbs = CXX_NEW(MAP(101, &_loc_pool), &_loc_pool);
}

DSyncE::~DSyncE()
{
  WN_MAP_Delete(_live_wns);
  CXX_DELETE(_live_bbs, &_loc_pool);
  MEM_POOL_Pop(&_loc_pool);
  MEM_POOL_Delete(&_loc_pool);
}


// ===================================================================
// This propogates the liveness information of sync nodes through the 
// reverse cfg.  It is assumed that each sync stmt and the terminal 
// statement of each bb is already marked with the summary liveness 
// state at that point. 
// ===================================================================

LIVE_STATE
DSyncE::Propogate_Liveness(WN* sync, BB_NODE* bb)
{
  FmtAssert(bb != NULL, ("traversal bb is null"));
  LIVE_STATE initial_state;

  //For the first level call, check the sync node's state 
  //rather than the basic block's state
  if(sync != NULL)
    initial_state = GetState(sync);
  else
    initial_state = GetBBState(bb);

  // If we already know the current state, return it
  if(initial_state != Sync_Unkwn)
    return initial_state;

  // Before traversing, mark the current node as dead, so we don't 
  // get infinite recursion on cycles, and get the correct information.
  // For the first level of recursion, the bb -must- have a non-unknown 
  // state, since it contains a sync point.
  if(sync == NULL)
    SetBBState(bb, Sync_Dead);

  BB_NODE* pred;
  BB_LIST_ITER pred_iter;
  LIVE_STATE path_state = Sync_Dead;

  FOR_ALL_ELEM( pred, pred_iter, Init(bb->Pred()))
  {
    // All deeper levels get a NULL sync pointer so they examine 
    // the end-block state
    path_state = Propogate_Liveness(NULL, pred);

    if(path_state == Sync_Live)
      break;
  }

  // If the for loop exited without finding a live path, the 
  // path_state will fall through with a dead value.
  // If this is the first level, the liveness leaving this bb 
  // is already defined, and independent of the liveness entering
  if(sync == NULL)
    SetBBState(bb, path_state);

  return path_state;
}


// ===================================================================
// Perform dead sync elimination
// ===================================================================

void
DSyncE::Do_dead_sync_elim()
{
  STACK<WN*> worklist_wn(&_loc_pool);
  STACK<BB_NODE*> worklist_bb(&_loc_pool);

  //Compute summary liveness information for each BB while searching 
  //for sync statements. 
  CFG_ITER cfgi(Cfg());
  BB_NODE* node;


  FOR_ALL_NODE(node, cfgi, Init())
  {
    WN* wn;
    WN* current_sync = NULL;

    LIVE_STATE current_state = Sync_Unkwn;

    for(wn = node->Laststmt(); wn != NULL; wn = WN_prev(wn)) 
    {
      // Synchronization kills, and with no intervening statements 
      // to redefine liveness, the last stmt in this region is sync-dead.
      if(Is_syncthreads(wn))
      {
        if(current_state == Sync_Unkwn)
          current_state = Sync_Dead;
        
        if(current_sync != NULL)
          SetState(current_sync, current_state);
        else
          SetBBState(node, current_state);
        current_sync = wn;

        // Save the sync point to propogate in the future
        worklist_wn.Push(wn);
        worklist_bb.Push(node);
        current_state = Sync_Unkwn;
      }
      // We define all shared memory accesses as defining sync liveness.
      else if(Has_Shared_Or_Global_Mem_Access(wn)) 
        current_state = Sync_Live;
    }

    // Save whatever the final state for the last explored stmt of 
    // this bb was.
    if(current_sync == NULL)
      SetBBState(node, current_state);
    else
      SetState(current_sync, current_state);
  }

  //For each synchronization point, propogate its deadness/liveness.
  while(worklist_wn.Elements() != 0)
  {
    WN* sync = worklist_wn.Pop();
    BB_NODE* start_block = worklist_bb.Pop();

    LIVE_STATE final_state = Propogate_Liveness(sync, start_block);

    // If the final state is dead, take out the sync.  
    if(final_state == Sync_Dead)
    {
      if (WOPT_Enable_Optinfo > 0) {
        if ((sync != NULL) && OPERATOR_is_stmt(WN_operator(sync)))
          ErrMsgSrcpos(EC_Dead_Sync, WN_Get_Linenum(sync), Cur_PU_Name);
        else
          ErrMsg(EC_Dead_Sync, Cur_PU_Name);
      }
      //start_block->Remove_stmtrep(sync_rep);
      Remove_WN_from_BB(sync, start_block);
    }
  }
}

static 
void has_shared_global_mem_access(WN * root, OPT_STAB *stab, BOOL& read, BOOL& write) 
{
  read = FALSE;
  write = FALSE;
  for (WN_ITER* wni = WN_WALK_TreeIter(root);
       wni != NULL;
       wni = WN_WALK_TreeNext(wni)) 
  {
    WN* addr = WN_ITER_wn(wni);
    OPERATOR opr = WN_operator(addr);
    if ((OPERATOR_is_load(opr) == FALSE) &&
	(OPERATOR_is_store(opr)== FALSE) &&
	(OPERATOR_is_call(opr) == FALSE)) {
      continue;
    }
    switch(opr)
    {
      case OPR_MSTORE:
      case OPR_ISTORE:
      case OPR_ISTBITS:
        write = TRUE;
        break;

      case OPR_MLOAD:
      case OPR_ILOAD:
      case OPR_ILDBITS:
        read = TRUE;
        break;

      case OPR_LDID:
      case OPR_STID:
      case OPR_LDBITS:
      case OPR_STBITS:
        //Check for shared memory access
        ST* real_sym;
        //Check the auxillary symbol table if necessary
        if(WN_has_aux(addr))
          real_sym = stab->St_ptr(addr);
        else
          real_sym = WN_st(addr);

        if(real_sym == NULL) {
          if(opr == OPR_LDID || opr == OPR_LDBITS)
            read = TRUE;
          else
            write = TRUE;
        }
        if(ST_in_shared_mem(real_sym) || ST_in_param_mem(real_sym) || ST_in_global_mem(real_sym)) {
          if(opr == OPR_LDID || opr == OPR_LDBITS)
            read = TRUE;
          else
            write = TRUE;
        }
        break;
      // Cover the COMIC case, where smem is represented by intrinsics
      // TBD : Can any other INTRINSIC_OP touch shared memory
      case OPR_INTRINSIC_OP:
      case OPR_INTRINSIC_CALL:
        Check_intrn_effects(WN_intrinsic(addr), read, write);
        break;

      // Conservatively assume the following OPRs as black boxes that can touch shared memory
      // TBD : Remove conservative assumption for special cases that we know about
      case OPR_CALL:
      case OPR_VFCALL:
      case OPR_ICALL:
      case OPR_PICCALL:
      case OPR_IO:
      case OPR_ASM_STMT:
        read = TRUE;
        write = TRUE;
        break;
      default:
        FmtAssert((!OPERATOR_is_load(opr)) && (!OPERATOR_is_store(opr)), ("Not expecting load or store here."));
        break;
    }
  }
}

static 
void has_shared_global_mem_access(BB_NODE *bb, OPT_STAB *stab, BOOL& read, BOOL& write)
{
  read = FALSE;
  write = FALSE;

  STMT_ITER stmt_iter;
  WN *stmt;

  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
    BOOL r=FALSE, w=FALSE;
    has_shared_global_mem_access(stmt, stab, r, w);
    read = read | r;
    write = write | w;
  }
}

/* Go backward until the last sync statement and check for
 * shared/global mem access from there.
 */
static 
void has_shared_global_mem_access_forward(BB_NODE *bb, OPT_STAB *stab, BOOL& read, BOOL& write)
{
  read = FALSE;
  write = FALSE;

  STMT_ITER stmt_iter;
  WN *stmt;

  BOOL found_sync = FALSE;
  WN *begin;

  FOR_ALL_ELEM_REVERSE( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
    if(Is_syncthreads(stmt)) {
      found_sync = TRUE;
      begin = stmt;
      break;
    }
  }

  if(found_sync == FALSE) {
    has_shared_global_mem_access(bb, stab, read, write);
    return;
  }

  FOR_ALL_ELEM( stmt, stmt_iter, Init(begin, bb->Laststmt()) ) {
    BOOL r=FALSE, w=FALSE;
    has_shared_global_mem_access(stmt, stab, r, w);
    read = read | r;
    write = write | w;
  }
}

/* Go forward until the first sync statement and check for
 * shared mem access until there.
 */
static 
void has_shared_global_mem_access_backward(BB_NODE *bb, OPT_STAB *stab, BOOL& read, BOOL& write)
{
  read = FALSE;
  write = FALSE;

  STMT_ITER stmt_iter;
  WN *stmt;

  BOOL found_sync = FALSE;
  WN *end;

  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
    if(Is_syncthreads(stmt)) {
      found_sync = TRUE;
      end = stmt;
      break;
    }
  }

  if(found_sync == FALSE) {
    has_shared_global_mem_access(bb, stab, read, write);
    return;
  }

  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(), end) ) {
    BOOL r=FALSE, w=FALSE;
    has_shared_global_mem_access(stmt, stab, r, w);
    read = read | r;
    write = write | w;
  }
}

void DSyncE::Set_local_sets(BOOL forward)
{
  shmem_r.clear();
  shmem_w.clear();

  CFG_ITER cfg_iter(Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE(bb, cfg_iter, Init()) {
    BOOL r=FALSE, w=FALSE;
    if(forward)
      has_shared_global_mem_access_forward(bb, Opt_stab(), r, w);
    else
      has_shared_global_mem_access_backward(bb, Opt_stab(), r, w);
    shmem_r[bb] = r;
    shmem_w[bb] = w;
  }
  if(Tracing()) {
    FOR_ALL_NODE(bb, cfg_iter, Init()) {

      printf("bb id %d - ", bb->Id());
      if(shmem_r[bb])
        printf("READ ");
      if(shmem_w[bb])
        printf("WRITE");
      printf("\n");
    }
  }
}

static
int num_syncthreads(BB_NODE *bb)
{
  STMT_ITER stmt_iter;
  WN *stmt;

  int retval = 0;

  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
    if(Is_syncthreads(stmt))
      retval++;
  }
  return retval;
}

static
BOOL has_syncthread(BB_NODE *bb)
{
  return (num_syncthreads(bb) > 0);
}


void DSyncE::Forward_analysis()
{
  f_nr.clear();
  f_nw.clear();
  f_xr.clear();
  f_xw.clear();

  CFG_ITER cfg_iter(Cfg());
  BB_NODE *bb;
  FOR_ALL_NODE(bb, cfg_iter, Init()) {
    f_nr[bb] = FALSE;
    f_nw[bb] = FALSE;
    f_xr[bb] = FALSE;
    f_xw[bb] = FALSE;
  }

  BOOL changed = FALSE;
  do {
    changed = FALSE;
    FOR_ALL_NODE(bb, cfg_iter, Init()) {
      BB_NODE *pred;
      BB_LIST_ITER pred_iter(bb->Pred());

      BOOL cur_nr = f_nr[bb],
        cur_nw = f_nw[bb],
        cur_xr = f_xr[bb],
        cur_xw = f_xw[bb];

      FOR_ALL_ELEM(pred, pred_iter, Init()) {
        f_nr[bb] |= f_xr[pred];
        f_nw[bb] |= f_xw[pred];
      }

      if(has_syncthread(bb)) {
        f_xr[bb] = shmem_r[bb];
        f_xw[bb] = shmem_w[bb];
      }
      else {
        f_xr[bb] = f_nr[bb] | shmem_r[bb];
        f_xw[bb] = f_nw[bb] | shmem_w[bb];
      }

      if(cur_nr != f_nr[bb]) changed = TRUE;
      if(cur_nw != f_nw[bb]) changed = TRUE;
      if(cur_xr != f_xr[bb]) changed = TRUE;
      if(cur_xw != f_xw[bb]) changed = TRUE;
    }
  } while(changed);

}

void DSyncE::Backward_analysis()
{
  b_nr.clear();
  b_nw.clear();
  b_xr.clear();
  b_xw.clear();

  CFG_ITER cfg_iter(Cfg());
  BB_NODE *bb;
  FOR_ALL_NODE(bb, cfg_iter, Init()) {
    b_nr[bb] = FALSE;
    b_nw[bb] = FALSE;
    b_xr[bb] = FALSE;
    b_xw[bb] = FALSE;
  }

  BOOL changed = FALSE;
  do {
    changed = FALSE;
    FOR_ALL_NODE_REVERSE(bb, cfg_iter, Init()) {
      BB_NODE *succ;
      BB_LIST_ITER succ_iter(bb->Succ());

      BOOL cur_nr = b_nr[bb],
        cur_nw = b_nw[bb],
        cur_xr = b_xr[bb],
        cur_xw = b_xw[bb];

      FOR_ALL_ELEM(succ, succ_iter, Init()) {
        b_xr[bb] |= b_nr[succ];
        b_xw[bb] |= b_nw[succ];
      }

      if(has_syncthread(bb)) {
        b_nr[bb] = shmem_r[bb];
        b_nw[bb] = shmem_w[bb];
      }
      else {
        b_nr[bb] = b_xr[bb] | shmem_r[bb];
        b_nw[bb] = b_xw[bb] | shmem_w[bb];
      }

      if(cur_nr != b_nr[bb]) changed = TRUE;
      if(cur_nw != b_nw[bb]) changed = TRUE;
      if(cur_xr != b_xr[bb]) changed = TRUE;
      if(cur_xw != b_xw[bb]) changed = TRUE;
    }
  } while(changed);
}

static
INT32 bb_linenum(BB_NODE *bb)
{
  STMT_ITER stmt_iter;
  WN *stmt;
  BOOL found = FALSE;
  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
    if(OPCODE_has_next_prev(WN_opcode(stmt))) {
      found = TRUE;
      break;
    }
  }
  if(found)
    return Srcpos_To_Line(WN_Get_Linenum(stmt));
  return 0LL;
}

void DSyncE::Local_deadness(BB_NODE *bb)
{
  STMT_ITER stmt_iter;
  WN *stmt;

  BOOL cr = f_nr[bb], cw = f_nw[bb];

  FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {

    if(Is_syncthreads(stmt)) {
      nr[stmt] = cr;
      nw[stmt] = cw;
      cr = FALSE;
      cw = FALSE;
      continue;
    }

    BOOL tempr = FALSE, tempw = FALSE;

    has_shared_global_mem_access(stmt, Opt_stab(), tempr, tempw);
    cr |= tempr;
    cw |= tempw;
  }

  cr = b_xr[bb];
  cw = b_xw[bb];

  FOR_ALL_ELEM_REVERSE( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {

    if(Is_syncthreads(stmt)) {
      xr[stmt] = cr;
      xw[stmt] = cw;
      cr = FALSE;
      cw = FALSE;
      continue;
    }

    BOOL tempr = FALSE, tempw = FALSE;

    has_shared_global_mem_access(stmt, Opt_stab(), tempr, tempw);
    cr |= tempr;
    cw |= tempw;
  }

}

BOOL DSyncE::Check_deadness()
{
  CFG_ITER cfg_iter(Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE(bb, cfg_iter, Init()) {
    if(!has_syncthread(bb))
      continue;

    Local_deadness(bb);

    STMT_ITER stmt_iter;
    WN *stmt;

    FOR_ALL_ELEM( stmt, stmt_iter, Init(bb->Firststmt(),bb->Laststmt()) ) {
      if(!Is_syncthreads(stmt))
        continue;

      BOOL dead_sync = FALSE;
      if(nr[stmt] == FALSE && nw[stmt] == FALSE) {
        dead_sync = TRUE;
      }
      else if(nr[stmt] == TRUE && nw[stmt] == FALSE) {
        if(xw[stmt] == FALSE)
          dead_sync = TRUE;
      }
      else if(nw[stmt] == TRUE) {
        if(xr[stmt] == FALSE && xw[stmt] == FALSE)
          dead_sync = TRUE;
      }
      if(dead_sync) {
        if(Tracing()) {
          printf("Dead Sync in bb %d Line %d ([%d %d], [%d %d])\n", bb->Id(),
                 Srcpos_To_Line(WN_Get_Linenum(stmt)),
                 nr[stmt], nw[stmt], xr[stmt], xw[stmt]);
        }
        if (WOPT_Enable_Optinfo > 0) {
          if ((stmt != NULL) && OPERATOR_is_stmt(WN_operator(stmt)))
            ErrMsgSrcpos(EC_Dead_Sync, WN_Get_Linenum(stmt), Cur_PU_Name);
          else
            ErrMsg(EC_Dead_Sync, Cur_PU_Name);
        }
        Remove_WN_from_BB(stmt, bb);
        return TRUE;
      }
    }
  }
  return FALSE;
}

void DSyncE::Do_dead_sync_elim_new()
{
  BOOL changed = FALSE;
  do {
    Set_local_sets(TRUE);
    Forward_analysis();
    Set_local_sets(FALSE);
    Backward_analysis();

    if(Tracing()) {
      CFG_ITER cfg_iter(Cfg());
      BB_NODE *bb;

      FOR_ALL_NODE(bb, cfg_iter, Init()) {

        if(!has_syncthread(bb))
          continue;
        printf("Entry info for bb id %d - ", bb->Id());
        if(f_nr[bb])
          printf("READ ");
        if(f_nw[bb])
          printf("WRITE");
        printf("\n");

        printf("Exit info for bb id %d - ", bb->Id());
        if(b_xr[bb])
          printf("READ ");
        if(b_xw[bb])
          printf("WRITE");
        printf("\n");

      }
    }

    changed = Check_deadness();
  } while(changed);

}
