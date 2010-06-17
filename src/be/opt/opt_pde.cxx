/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//  Options to PDE :
//
//  PDE uses one variable WOPT_PDE_Options, which is controlled by the
//  command line option -WOPT:pde_options to control difference settings.
//  WOPT_PDE_Options is treated as a bit-vector, and different bits
//  control different settings. Different bits are described below :
//  Bit 0 is the LSB, Bit 1 is the next LSB and so on.
//
//  Bit 0    : Enables tracing
//  Bit 1-3  : Control delayability and floatability options. When
//             treated as a 3-bit number, Bit 1-3 controls the following :
//             0 - run delayability only
//             1 - run floatability only
//             2 - run delayability followed by floatability
//             3 - run floatability followed by delayability 
//  Bit 4    : Enables liveness sensitive PDE     
//
//  Default is pde_options=0
//  
//  -WOPT:pde_max_iters=n
//    -controls number of iterations for delayability/floatability.
//    -defaults to 0, which means they are run until convergence.
//  
#include "opt_pde.h"
#include <memory>
#include <algorithm>

//  PDE settings
INT Order_Options;
INT Enable_Liveness_Sensitive;

//  Some typedefs of maps used later.
//  
typedef mempool_allocator<std::pair<WN *, IDX_32_SET *> > wn_idxset_alloc_t;
typedef mempool_allocator<std::pair<WN *, INT> >          wn_int_alloc_t;
typedef mempool_allocator<std::pair<WN *, BOOL> >         wn_bool_alloc_t;
typedef mempool_allocator<std::pair<WN *, set<WN *> > >   wn_set_alloc_t;

typedef hash_map<WN *, IDX_32_SET *, hash<WN *>, eqWN, wn_idxset_alloc_t> wn_idxset_map_t;
typedef hash_map<WN *, INT, hash<WN *>, eqWN, wn_int_alloc_t> wn_int_map_t;
typedef hash_map<WN *, BOOL, hash<WN *>, eqWN, wn_bool_alloc_t> wn_bool_map_t;
typedef hash_map<WN *, set<WN *>, hash<WN *>, eqWN, wn_set_alloc_t> wn_set_map_t;


//  Number_stmts : This function gives a unique number to the statements
//  in the function. This enables the use of IDX_32_SET to represent set
//  of statements. The static global variables num_stmts, stmt_map and stmt_rev_map
//  are reset by this function.
//  
static INT num_stmts;
static wn_int_map_t stmt_map;
static hash_map<INT, WN *, hash<INT>, __gnu_cxx::equal_to<INT> > stmt_rev_map;
static map<WN *, WN *> parent_map;

static void Find_parents(WN *wn)
{
  if (!OPCODE_is_leaf (WN_opcode (wn))) { 
    if (WN_operator(wn) == OPR_BLOCK) {
      WN* kid = WN_first (wn);
      while (kid) {
        parent_map[kid] = wn;
        Find_parents (kid);
        kid = WN_next (kid);
      }
    }
    else {
      INT kidno;
      WN* kid;
      for (kidno=0; kidno<WN_kid_count(wn); kidno++) {
        kid = WN_kid (wn, kidno);
        if (kid) {
          parent_map[kid] = wn;
          Find_parents (kid);
        }
      }
    }
  }
}

static BOOL Number_stmts(WN *wn)
{
  num_stmts = 0;
  stmt_map.clear();
  stmt_rev_map.clear();

  WN_ITER *ti1 = WN_WALK_StmtIter(wn);
  while(ti1) {
    WN *stmt = WN_ITER_wn(ti1);

    if(!(OPERATOR_is_stmt(WN_operator(stmt)))) {
      ti1 = WN_WALK_StmtNext(ti1);
      continue;
    }

    stmt_map[stmt] = num_stmts;
    stmt_rev_map[num_stmts] = stmt;
    num_stmts++;

    ti1 = WN_WALK_StmtNext(ti1);
  }
  parent_map.clear();
  Find_parents(wn);
}

//  Delayability computes the set of statements that can be delayed until
//  a given statement, and stores it in delayed_map. The set of statements
//  is represented as a IDX_32_SET. This assumes Number_stmts has been called,
//  so that num_stmts, stmt_map, and stmt_rev_map are valid.
//  

static wn_idxset_map_t delayed_map;

//  Similar map as above, for Floatability
//  
static wn_idxset_map_t floated_map;

//  Delayability populates this map with the copies of statements
//  that it moved.
//  
static wn_bool_map_t moved_by_delayability;
static wn_bool_map_t moved_by_floatability;
static wn_bool_map_t immoveable_stmts;

//  Has_dedicated_preg_num : This function finds out if any 
//  of the operands of wn is a dedicated preg. This is used by
//  ASM statements and call statements to represent arguments.
//  These statements should not be moved.
//  So this function is eventually used by is_immoveable.
//  
static BOOL Has_dedicated_preg_num(WN *wn)
{
  WN_ITER *ti1 = WN_WALK_TreeIter(wn);
  while(ti1) {
    WN *node = WN_ITER_wn(ti1);

    if((WN_operator(node) != OPR_LDID) && (WN_operator(node) != OPR_STID)) {
      ti1 = WN_WALK_TreeNext(ti1);
      continue;
    }

    if(WN_class(node) != CLASS_PREG) {
      ti1 = WN_WALK_TreeNext(ti1);
      continue;
    }

    if(Preg_Is_Dedicated(WN_offset(node)))
      return TRUE;
    ti1 = WN_WALK_TreeNext(ti1);
  }
  return FALSE;
}

//  In_same_bb : Finds out if wn1 and wn2 are in the same basic block.
//  Walk forward and backward from wn1 to see if we see a label or a
//  branch before we see wn2. Do the same thing starting from wn2. If we hit wn1
//  (or wn2) before a label or a branch, then they are in the same basic block.
//  This is used eventually to decide if we should move a statement. Statements
//  should NOT be moved within basic blocks.
//  
static BOOL In_same_bb(WN *wn1, WN *wn2)
{
  WN *temp;

  temp = WN_prev(wn1);

  while(temp) {
    if(temp == wn2)
      return TRUE;
    if((WN_operator(temp) == OPR_GOTO) ||
       (WN_operator(temp) == OPR_TRUEBR) ||
       (WN_operator(temp) == OPR_FALSEBR) ||
       (WN_operator(temp) == OPR_LABEL)) {
      break;
    }
    temp = WN_prev(temp);
  }

  temp = WN_next(wn1);

  while(temp) {
    if(temp == wn2)
      return TRUE;
    if((WN_operator(temp) == OPR_GOTO) ||
       (WN_operator(temp) == OPR_TRUEBR) ||
       (WN_operator(temp) == OPR_FALSEBR) ||
       (WN_operator(temp) == OPR_LABEL))
      return FALSE;
    temp = WN_next(temp);
  }
  return FALSE;
}

//  is_[rw]a[rw]_hazard : The following 3 functions decide data dependency
//  between the arguments src and dest. 
//  
//  raw -> read after write.
//  war -> write after read.
//  waw -> write after write.
//  
//  They use the global variable 'alias_mgr' that has to be set before
//  they are called.
//  
//  First, all the reads (or writes) is collected from the statements by walking
//  the tree, and stored in src_[loads/stores] and dest_[loads/stores] vectors.
//  Let's look at Is_raw_hazard. src_stores and dest_loads are populated with
//  stores in src statement and loads in dest statement. There is a RAW hazard if
//  some load in dest Aliases with some store in src. That is checked by using the
//  Aliased function. WAR and WAW cases are handled similarly.
//   
static ALIAS_MANAGER *alias_mgr;

static BOOL Is_raw_hazard(WN *src, WN *dest)
{
  vector<WN *> src_stores;
  vector<WN *> dest_loads;

  WN_ITER *ti1 = WN_WALK_TreeIter(src);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);
    if(OPERATOR_is_store(WN_operator(temp)))
      src_stores.push_back(temp);
    ti1 = WN_WALK_TreeNext(ti1);
  }

  WN_ITER *ti2 = WN_WALK_TreeIter(dest);
  while(ti2) {
    WN *temp = WN_ITER_wn(ti2);
    if(OPERATOR_is_load(WN_operator(temp)))
      dest_loads.push_back(temp);
    ti2 = WN_WALK_TreeNext(ti2);
  }

  for(INT i=0; i<src_stores.size(); i++) {
    for(INT j=0; j<dest_loads.size(); j++) {
      if(Aliased(alias_mgr, src_stores[i], dest_loads[j], 
                 FALSE) != NOT_ALIASED) {
        return TRUE;
      }
    }
  }

  return FALSE;
}

//  Check if there is a waw hazard from src to dest, src -> dest
static BOOL Is_waw_hazard(WN *src, WN *dest)
{
  vector<WN *> src_stores;
  vector<WN *> dest_stores;

  WN_ITER *ti1 = WN_WALK_TreeIter(src);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);
    if(OPERATOR_is_store(WN_operator(temp)))
      src_stores.push_back(temp);
    ti1 = WN_WALK_TreeNext(ti1);
  }

  WN_ITER *ti2 = WN_WALK_TreeIter(dest);
  while(ti2) {
    WN *temp = WN_ITER_wn(ti2);
    if(OPERATOR_is_store(WN_operator(temp)))
      dest_stores.push_back(temp);
    ti2 = WN_WALK_TreeNext(ti2);
  }

  for(INT i=0; i<src_stores.size(); i++) {
    for(INT j=0; j<dest_stores.size(); j++) {
      if(Aliased(alias_mgr, src_stores[i], dest_stores[j], 
                 FALSE) != NOT_ALIASED) {
        return TRUE;
      }
    }
  }

  return FALSE;
}

//  Check if there is a war hazard from src to dest, src -> dest
static BOOL Is_war_hazard(WN *src, WN *dest)
{
  vector<WN *> src_loads;
  vector<WN *> dest_stores;

  WN_ITER *ti1 = WN_WALK_TreeIter(src);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);
    if(OPERATOR_is_load(WN_operator(temp)))
      src_loads.push_back(temp);
    ti1 = WN_WALK_TreeNext(ti1);
  }

  WN_ITER *ti2 = WN_WALK_TreeIter(dest);
  while(ti2) {
    WN *temp = WN_ITER_wn(ti2);
    if(OPERATOR_is_store(WN_operator(temp)))
      dest_stores.push_back(temp);
    ti2 = WN_WALK_TreeNext(ti2);
  }

  for(INT i=0; i<src_loads.size(); i++) {
    for(INT j=0; j<dest_stores.size(); j++) {
      if(Aliased(alias_mgr, src_loads[i], dest_stores[j], 
                 FALSE) != NOT_ALIASED) {
        return TRUE;
      }
    }
  }

  return FALSE;
}

//  Find out if wn is a barrier. Check for INTRN_SYNCHRONIZE
//  operator.
//  
static BOOL Is_barrier(WN *wn)
{
  FmtAssert(wn != NULL, ("null node tested for __synchthreads()"));

  if (WN_operator(wn) == OPR_INTRINSIC_CALL) {
    return (WN_intrinsic(wn) == INTRN_SYNCHRONIZE);
  }

  return FALSE;
}

//  Always_blocks: Checks if wn always blocks every statement.
//  The following type statements always block statements from above.
//  
//  1. Barriers.
//  2. Calls.
//  3. Returns.
//  4. ASM statements.
//  5. Pragmas
//  
static BOOL Always_blocks(WN *wn)
{
  if(Is_barrier(wn))
    return TRUE;
  if(WN_operator(wn) == OPR_FORWARD_BARRIER)
    return TRUE;
  if(WN_operator(wn) == OPR_BACKWARD_BARRIER)
    return TRUE;
  if(OPERATOR_is_call(WN_operator(wn)))
    return TRUE;
  if(WN_operator(wn) == OPR_RETURN)
    return TRUE;
  if(WN_operator(wn) == OPR_RETURN_VAL)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_STMT)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_INPUT)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_EXPR)
    return TRUE;

  return FALSE;
}

//  Is_blocked: Checks to see if dest blocks src.
//  
//  If dest Always_blocks, returns TRUE.
//  Otherwise, checks to see various types of data dependencies. Basically, if
//  any type dependency exists for src -> dest, returns TRUE.
//  
static BOOL Is_blocked(WN *src, WN *dest)
{
  if(Always_blocks(dest))
    return TRUE;
  if(Is_raw_hazard(src, dest))
    return TRUE;
  if(Is_waw_hazard(src, dest))
    return TRUE;
  if(Is_war_hazard(src, dest))
    return TRUE;
  return FALSE;
}

//  Is_blocked_from_above: Checks to see if src blocks dest's movement upwards.
//  
//  If src Always_blocks, returns TRUE.
//  Otherwise, checks to see various types of data dependencies. Basically, if
//  any type dependency exists for src -> dest, returns TRUE.
//  
static BOOL Is_blocked_from_above(WN *src, WN *dest)
{
  if(Always_blocks(src))
    return TRUE;
  if(Is_raw_hazard(src, dest))
    return TRUE;
  if(Is_waw_hazard(src, dest))
    return TRUE;
  if(Is_war_hazard(src, dest))
    return TRUE;
  return FALSE;
}

//  Is_killed: Checks to see if dest kills src.
//  
//  For dest to kill src, all the stores of src must be overwritten
//  by dest. 'Overwritten' is checked with
//  Aliased(src_stores, dest_stores) == SAME_LOCATION.
//  
static BOOL Is_killed(WN *src, WN *dest)
{
  if(src == dest)
    return FALSE;

  vector<WN *> src_stores;
  vector<WN *> dest_stores;

  WN_ITER *ti1 = WN_WALK_TreeIter(src);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);
    if(OPERATOR_is_store(WN_operator(temp)))
      src_stores.push_back(temp);
    ti1 = WN_WALK_TreeNext(ti1);
  }

  WN_ITER *ti2 = WN_WALK_TreeIter(dest);
  while(ti2) {
    WN *temp = WN_ITER_wn(ti2);
    if(OPERATOR_is_store(WN_operator(temp)))
      dest_stores.push_back(temp);
    ti2 = WN_WALK_TreeNext(ti2);
  }

  if(dest_stores.size() < 1)
    return FALSE;

  if(src_stores.size() < 1)
    return FALSE;

  for(INT i=0; i<src_stores.size(); i++) {
    for(INT j=0; j<dest_stores.size(); j++) {
      if(Aliased(alias_mgr, src_stores[i], dest_stores[j], 
                 FALSE) != SAME_LOCATION) {
        return FALSE;
      }
    }
  }

  return TRUE;
}

//  Has_volatile_access : Check if there are volatile accesses in the WHIRL node.
//  
static BOOL Has_volatile_access(WN *wn)
{
  WN_ITER *ti1 = WN_WALK_TreeIter(wn);
  while(ti1) {
    WN *node = WN_ITER_wn(ti1);

    switch(WN_operator(node)) {
    case OPR_LDID:
    case OPR_LDBITS:
    case OPR_MLOAD:
      if(TY_is_volatile(WN_ty(node)))
        return TRUE;
      break;

    case OPR_ILOAD:
    case OPR_ILOADX:
    case OPR_ILDBITS:
      if(TY_is_volatile(WN_ty(node)) ||
         TY_is_volatile(WN_load_addr_ty(node)) ||
         TY_is_volatile(TY_pointed(WN_load_addr_ty(node))))
        return TRUE;
      break;

    case OPR_STID:
    case OPR_STBITS:
    case OPR_MSTORE:
      if(TY_is_volatile(WN_ty(node)))
        return TRUE;
      break;

    case OPR_ISTORE:
    case OPR_ISTOREX:
    case OPR_ISTBITS:
      if(TY_is_volatile(WN_ty(node)) ||
         TY_is_volatile(TY_pointed(WN_ty(node))))
        return TRUE;
      break;

    default:
      break;
    }

    ti1 = WN_WALK_TreeNext(ti1);
  }
  return FALSE;
}

//  Is_immoveable: Checks to see if wn can be moved.
//  The following cannot be moved.
//  
//  1. Barriers.
//  2. Calls.
//  3. Returns.
//  4. Branches, GOTOs, Labels.
//  5. IOs.
//  6. PRAGMAs.
//  7. ASM statements.
//  8. Statements with negative preg nums. These feed ASM statements.
//  9. Has_volatile_access.
//  
static BOOL Is_immoveable(WN *wn)
{
  if(Is_barrier(wn))
    return TRUE;
  if(WN_operator(wn) == OPR_FORWARD_BARRIER)
    return TRUE;
  if(WN_operator(wn) == OPR_BACKWARD_BARRIER)
    return TRUE;
  if(OPERATOR_is_call(WN_operator(wn)))
    return TRUE;
  if(WN_operator(wn) == OPR_RETURN)
    return TRUE;
  if(WN_operator(wn) == OPR_RETURN_VAL)
    return TRUE;
  if(WN_operator(wn) == OPR_LABEL || 
     WN_operator(wn) == OPR_FALSEBR ||
     WN_operator(wn) == OPR_TRUEBR ||
     WN_operator(wn) == OPR_GOTO ||
     WN_operator(wn) == OPR_IO 
     )
    return TRUE;
  if(WN_operator(wn) == OPR_PRAGMA)
    return TRUE;
  if(WN_operator(wn) == OPR_LABEL)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_STMT)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_INPUT)
    return TRUE;
  if(WN_operator(wn) == OPR_ASM_EXPR)
    return TRUE;
  if(Has_dedicated_preg_num(wn))
    return TRUE;
  if(Has_volatile_access(wn))
    return TRUE;
  return FALSE;
}

//  This is the memory pool used by PDE. Right now, it is initialized before
//  PDE starts, and destroyed after all iterations of PDE ends.
//  TBD: Memory can be reclaimed after every iteration of PDE. So maybe the
//  memory pool has to be initialized and destroyed for each iteration?
//  
static MEM_POOL pde_pool;

//  Constructors for Delayability. Creates and empty set.
//  
DelayabilityState::DelayabilityState()
{
  delayed = CXX_NEW(IDX_32_SET(num_stmts, &pde_pool, OPTS_FALSE), &pde_pool);
}

//  Destructor: Can destroy delayed. But destruction is taken care of later
//  by the memory pool manager. So doing nothing here.
//  
DelayabilityState::~DelayabilityState()
{
}

//  CopyFrom: Copy delayed state from other instance
//  
void DelayabilityState::CopyFrom(DelayabilityState *other)
{
  delayed->CopyD(other->delayed);
}

//  Empty: This makes the set to be the universe. This is because
//  the confluence function is intersection.
//  
void DelayabilityState::Empty()
{
  delayed->UniverseD(num_stmts);
}

//  Merge: Intersect with the other instance.
//  Returns TRUE if the intersection results in a new set.
//  This requires a temporary copy to be made on the stack.
//  TBD: Is there a different way that makes no copies?
//  
BOOL DelayabilityState::Merge(DelayabilityState *other)
{
  IDX_32_SET temp(num_stmts, &pde_pool, OPTS_FALSE);
  temp.CopyD(delayed);

  delayed->IntersectionD(other->delayed);

  return !(temp.EqualP(delayed));
}

// Parent_Is_Stmt:
// Walk the parent links to find if the given statement is 
// a child of a statement whirl node. Some children of ASM_STMT
// are themselves statements.
static BOOL Parent_Is_Stmt(WN *wn)
{
  if(parent_map.find(wn) == parent_map.end())
    return FALSE;
  WN *p = parent_map[wn];
  if(OPERATOR_is_stmt(WN_operator(p)))
    return TRUE;
  return Parent_Is_Stmt(p);
}

//  TransferFunction: 
//  Input -> Set of statements coming from above.
//  Output -> Set of statements that can be delayed below.
//  
//  For each statement coming from above, find out if the current statement
//  blocks that statement. If so remove from set.
//  Finally, if current statement can be delayed, add it to the set.
//  
//  Another functionality is piggybacked on to this function, which is
//  filling the label_map. label_map holds the WN* corresponding to a label number.
//  Used later to find out what is the delayed set for a label, given a label
//  number.
//  
static hash_map<INT32, WN *, hash<INT32>, __gnu_cxx::equal_to<INT32> > label_map;

void DelayabilityClient::TransferFunction(WN *node, DelayabilityState *state)
{
  if(WN_operator(node) == OPR_GOTO)
    return;
  if(WN_operator(node) == OPR_BLOCK)
    return;
  if(WN_operator(node) == OPR_LABEL) {
    label_map[WN_label_number(node)] = node;
    return;
  }
  if(!OPERATOR_is_stmt(WN_operator(node)))
    return;
  if(Parent_Is_Stmt(node))
    return;

  FmtAssert(OPERATOR_is_stmt(WN_operator(node)), ("Expecting a stmt here."));

  IDX_32_SET *delayed = state->delayed;
  IDX_32 x;
  for(x=delayed->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
      x=delayed->Choose_Next(x)) {
    WN *si = stmt_rev_map[x];
    if(Is_blocked(si, node)) {
      delayed->Difference1D(stmt_map[si]);
    }
  }

//   if((!Is_raw_hazard(node, node)) &&
//      (!Is_war_hazard(node, node)) &&
  if((!Is_immoveable(node)) &&
     (immoveable_stmts.find(node) == immoveable_stmts.end()) &&
     (moved_by_floatability.find(node) == moved_by_floatability.end())) {
    delayed->Union1D(stmt_map[node]);
    FmtAssert(!(Is_immoveable(node)), ("Cannot move an immoveable node."));
  }
}

static void Print_Delayed_Stmts(IDX_32_SET *dset)
{
  IDX_32 x;
  INT c = 0;
  for(x=dset->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
      x=dset->Choose_Next(x)) {
    printf("stmt %d:\n", c); dump_tree(stmt_rev_map[x]);
    c++;
  }
}

//  Apply: Populates delayed_map with the current delayed set.
//   
void DelayabilityClient::Apply(WN *node, DelayabilityState *state)
{
  if(WN_operator(node) == OPR_BLOCK)
    return;

  if(!OPERATOR_is_stmt(WN_operator(node)))
    return;

  if(Parent_Is_Stmt(node))
    return;

  FmtAssert(OPERATOR_is_stmt(WN_operator(node)), ("Expecting a stmt here."));

#if 0
  if(Tracing()) {
    printf("delayed for :\n"); dump_tree(node);
    IDX_32 x;
    INT c = 0;
    for(x=delayed->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
        x=delayed->Choose_Next(x)) {
      printf("stmt %d:\n", c); dump_tree(stmt_rev_map[x]);
      c++;
    }
  }
#endif

  delayed_map[node] = CXX_NEW(IDX_32_SET(num_stmts, &pde_pool, OPTS_FALSE), &pde_pool);
  delayed_map[node]->CopyD(state->delayed);
}

//  Helper functions.
//  Insert_wn_before, Insert_wn_after, remove_wn.
//  Pretty much self explanatory.
//  
static void Insert_wn_before(WN *wn, WN *before_this)
{
  WN_next(wn) = before_this;
  if(WN_prev(before_this) != NULL) {
    WN_prev(wn) = WN_prev(before_this);
    WN_next(WN_prev(wn)) = wn;
  }
  else {
    WN_prev(wn) = NULL;
  }
  WN_prev(before_this) = wn;
}

//  insert wn after "after_this"
static void Insert_wn_after(WN *wn, WN *after_this)
{
  WN_prev(wn) = after_this;
  if(WN_next(after_this) != NULL) {
    WN_next(wn) = WN_next(after_this);
    WN_prev(WN_next(wn)) = wn;
  }
  else {
    WN_next(wn) = NULL;
  }
  WN_next(after_this) = wn;
}

//  remove wn from the tree
static void Remove_wn(WN *wn)
{
  if(wn == NULL)
    return;

  if(WN_next(wn) != NULL)
    WN_prev(WN_next(wn)) = WN_prev(wn);
  if(WN_prev(wn) != NULL)
    WN_next(WN_prev(wn)) = WN_next(wn);

  WN_Delete(wn);
}

// Liveness related variables
static WN_MAP liveness_map;
static MEM_POOL liveness_pool;
static BOOL liveness_valid = FALSE;
static loc_set_t program_liveins;
static loc_set_t written_vars, read_vars;

// Data flow state and client for liveness
LivenessState::LivenessState()
{
  _cur_live_vars.clear();
}

LivenessState::~LivenessState()
{
}

void LivenessState::CopyFrom(LivenessState *other)
{
  _cur_live_vars = other->_cur_live_vars;
}

BOOL LivenessState::Merge(LivenessState *other)
{
  loc_set_t temp;
  set_union(_cur_live_vars.begin(), _cur_live_vars.end(),
            other->_cur_live_vars.begin(), other->_cur_live_vars.end(),
            inserter(temp, temp.begin()),
            loc_t_cmp());

  loc_set_t diff;

  set_difference(temp.begin(), temp.end(),
                 _cur_live_vars.begin(), _cur_live_vars.end(),
                 inserter(diff, diff.begin()),
                 loc_t_cmp());

  BOOL retval = TRUE;
  if(diff.empty())
    retval = FALSE;

  _cur_live_vars = temp;

  return retval;
}

void LivenessState::Empty(void)
{
  loc_set_t temp;
  _cur_live_vars = temp;
}

extern void dump_st(ST *);

static void Loc_set_print(loc_set_t& _cur_live_vars)
{
  printf("Liveness info :\n");

  if(_cur_live_vars.empty()) {
    printf("Empty set\n");
    return;
  }

  for(loc_set_t::iterator si=_cur_live_vars.begin();
      si!=_cur_live_vars.end(); si++) {
    printf("[\n");
    printf("Symbol :\n");
    dump_st((*si).sym);
    printf("Offset : %u\n", (*si).off);
    printf("]\n");
  }
  printf("\n");
}

void LivenessState::Print(void)
{
  printf("Liveness info :\n");

  if(_cur_live_vars.empty()) {
    printf("Empty set\n");
    return;
  }

  for(loc_set_t::iterator si=_cur_live_vars.begin();
      si!=_cur_live_vars.end(); si++) {
    printf("[\n");
    printf("Symbol :\n");
    dump_st((*si).sym);
    printf("Offset : %u\n", (*si).off);
    printf("]\n");
  }
  printf("\n");
}

void LivenessClient::TransferFunction(WN *wn, LivenessState *in)
{
  loc_set_t writes, reads;
  if(!(OPERATOR_is_stmt(WN_operator(wn))))
    return;

  WN_ITER *ti1 = WN_WALK_TreeIter(wn);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);

    switch(WN_operator(temp)) {
    case OPR_LDID:
      reads.insert(loc_t(WN_st(temp), WN_offset(temp)));
      read_vars.insert(loc_t(WN_st(temp), WN_offset(temp)));
      break;
    case OPR_STID:
      writes.insert(loc_t(WN_st(temp), WN_offset(temp)));
      written_vars.insert(loc_t(WN_st(temp), WN_offset(temp)));
      break;
    default:
      break;
    }
    ti1 = WN_WALK_TreeNext(ti1);
  }

  loc_set_t result;

  // Subtract the writes
  set_difference(in->_cur_live_vars.begin(), in->_cur_live_vars.end(),
                 writes.begin(), writes.end(),
                 inserter(result, result.begin()),
                 loc_t_cmp());

  in->_cur_live_vars.clear();

  // Add the reads
  set_union(result.begin(), result.end(),
            reads.begin(), reads.end(),
            inserter(in->_cur_live_vars, in->_cur_live_vars.begin()),
            loc_t_cmp());
}

void LivenessClient::Apply(WN *wn, LivenessState *in)
{
  if(!(OPERATOR_is_stmt(WN_operator(wn))))
    return;
  WN_MAP_Set(liveness_map, wn, new loc_set_t(in->_cur_live_vars));
}

static void Create_liveness_info(WN *wn)
{
  MEM_POOL_Initialize(&liveness_pool, "liveness pool", FALSE);
  MEM_POOL_Push(&liveness_pool);

  liveness_map = WN_MAP_Create(&liveness_pool);

  LivenessClient *lclient = new LivenessClient();
  LivenessState *lstate = new LivenessState();
  lstate->Empty();

  DFSolver<LivenessClient, LivenessState> l(lclient);
  l.Solve(FALSE, wn, lstate);
  program_liveins.clear();

  set_difference(read_vars.begin(), read_vars.end(),
                 written_vars.begin(), written_vars.end(),
                 inserter(program_liveins, program_liveins.begin()),
                 loc_t_cmp());
  liveness_valid = TRUE;
}

static void Destroy_liveness_info()
{
  WN_MAP_Delete(liveness_map);

  MEM_POOL_Pop(&liveness_pool);
  MEM_POOL_Delete(&liveness_pool);
  liveness_valid = FALSE;
}

// Check_liveness_constraint : If any of the LDIDs in stmt are not live
// after where, then return FALSE.
//
static BOOL Check_liveness_constraint(WN *stmt, WN *where)
{
  loc_set_t reads;

  if(WN_prev(where))
    where = WN_prev(where);

  WN_ITER *ti1 = WN_WALK_TreeIter(stmt);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);

    switch(WN_operator(temp)) {
    case OPR_LDID:
    {
      loc_t var(WN_st(temp), WN_offset(temp));
      //if(program_liveins.find(var) == program_liveins.end())
        reads.insert(var);
    }
      break;
    default:
      break;
    }
    ti1 = WN_WALK_TreeNext(ti1);
  }

  loc_set_t *liveset = (loc_set_t *)WN_MAP_Get(liveness_map, where);

  if(liveset == NULL) {
    return TRUE;
  }

  // All elements in reads have to be live after where.
  for(loc_set_t::iterator si=reads.begin(); si!=reads.end(); si++) {
    if(liveset->find(*si) == liveset->end())
      return FALSE;
  }

  return TRUE;
}

//  Insertion_points: Uses delayed_map to find where to insert statements
//  and inserts them. See inline comments for details.
//  
static INT Insertion_points(WN *wn, BOOL move_within_bb=FALSE)
{
  INT num_moved = 0;
  //  Insertion points are kept track of in the points_map. It contains for
  //  each statement, the set of points, viz., the statements before which
  //  the statement should be inserted. since the number of insertion points
  //  will be pretty small, (no more than two in most cases), a set is
  //  used to represent the set.
  //  

  wn_set_map_t points_map;

  //  Go through each statement 's' and look at the delayed set from delayed_map
  //  for that statement. If delayed_map[s] contains statement 'x', but
  //  delayed_map[WN_next(s)] does not contain 'x', then 'x' should be inserted
  //  before 's'.
  //  
  WN_ITER *ti1 = WN_WALK_StmtIter(wn);
  while(ti1) {
    WN *temp = WN_ITER_wn(ti1);
    //  If delayed_map does not have an entry for temp,
    //  then probably this is a PRAGMA. (TBD: Will this ever happen?)
    //  So just continue.
    //  
    if(delayed_map.find(temp) == delayed_map.end()) {
      ti1 = WN_WALK_StmtNext(ti1);
      continue;
    }

    IDX_32_SET *delayed = delayed_map[temp];

    //  If delayed set is empty, nothing to do here, just continue.
    //  
    if(delayed->EmptyP()) {
      ti1 = WN_WALK_StmtNext(ti1);
      continue;
    }

    //  Get the logically next statement. For a statement that is not a GOTO,
    //  the logically next statement is WN_next(s). For a GOTO, it is the statement
    //  after the target label of the goto statement.
    //  
    WN *temp_next = NULL;
    if(WN_operator(temp) == OPR_GOTO) {
      //  label_map should have been populated correctly.
      //  
      FmtAssert((label_map.find(WN_label_number(temp)) != label_map.end()),
                ("Cannot find label WN for label."));
      temp_next = label_map[WN_label_number(temp)];
    }
    else if((WN_operator(temp) == OPR_RETURN) ||
            (WN_operator(temp) == OPR_RETURN_VAL)) {
      temp_next = NULL;
    }
    else {
      temp_next = WN_next(temp);
    }

    //  Temporary vector to hold the set of statements that should be
    //  inserted here.
    //  
    vector<WN *> to_be_inserted;

    //  If temp_next is NULL here, then temp was the last statement. So
    //  all statements in delayed map have to inserted here. Otherwise only
    //  those not in the delayed_map[temp_next] need to be inserted here.
    //  
    if(temp_next != NULL) {
      if(delayed_map.find(temp_next) != delayed_map.end()) {
        IDX_32_SET *next_delayed = delayed_map[temp_next];
        IDX_32 x;
        for(x=delayed->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
            x=delayed->Choose_Next(x)) {
          if(next_delayed->MemberP(x))
            continue;
          to_be_inserted.push_back(stmt_rev_map[x]);
        }
      }
      else {
        //  TBD : Can this situation ever arise?
        //  
        IDX_32 x;
        for(x=delayed->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
            x=delayed->Choose_Next(x)) {
          to_be_inserted.push_back(stmt_rev_map[x]);
        }
      }
    }
    else {
      //  Last statement in the function. So insert everything
      //  delayed here before the statement.
      //  
      IDX_32 x;
      for(x=delayed->Choose(); x!=IDX_32_SET_CHOOSE_FAILURE;
          x=delayed->Choose_Next(x)) {
        to_be_inserted.push_back(stmt_rev_map[x]);
      }
    }

    //  Go through to_be_inserted and populate the points map.
    //   
    for(INT i=0; i<to_be_inserted.size(); i++) {
      WN *stmt = to_be_inserted[i];

      //  If stmt and temp (the current statement are in the
      //  same basic block, then don't bother inserting.
      //  TBD : Is this the right place to do this check?
      //  
      if(!move_within_bb) {
        if(In_same_bb(stmt, temp))
          continue;
      }
      else {
        if(!In_same_bb(stmt, temp))
          continue;
      }
      if(WN_next(stmt) == temp)
        continue;
      if(points_map.find(stmt) != points_map.end()) {
        points_map[stmt].insert(temp);
      }
      else {
        set<WN *> temp_points;
        temp_points.insert(temp);
        points_map[stmt] = temp_points;
      }
    }

    ti1 = WN_WALK_StmtNext(ti1);
  }

  //  Find out all the places a statement has to go from points_map.
  //  Insert them there. Put the original statement in to_remove
  //  vector, so that they can be removed later.
  //  
  vector<WN *> to_remove;

  INT which_stmt = 0;

  if(Enable_Liveness_Sensitive) {
    if(liveness_valid == FALSE)
      Create_liveness_info(wn);
  }

  for(wn_set_map_t::iterator pi=points_map.begin();
      pi!=points_map.end(); pi++) {
    WN *stmt = (*pi).first;
    set<WN *> points = (*pi).second;

    //  If no copies of a statement are allowed (WOPT_PDE_Options & 16),
    //  then don't move statements if it has to be moved to more than one place.
    //  
    if(WOPT_PDE_Options & 16) {
      if(points.size() > 1)
        continue;
    }
    //  For Debugging: Hook to stop after moving certain number of
    //  statements.
    //  
    which_stmt++;

    if(Enable_Liveness_Sensitive) {
      int places_count = 0;
      for(set<WN *>::iterator wi=points.begin(); wi!=points.end(); wi++) {
        WN *where = *wi;

        if(Check_liveness_constraint(stmt, where) == FALSE)
          break;
        places_count++;
      }
      if(places_count < points.size())
        continue;
    }

    // If moved to all locations in the points set, remove the op. 
    // Otherwise, don't remove it.
    BOOL not_moved = FALSE;
    BOOL moved_once = FALSE;

    for(set<WN *>::iterator wi=points.begin(); wi!=points.end(); wi++) {
      WN *where = *wi;

      FmtAssert(!(Is_immoveable(stmt)), ("Cannot move an immoveable stmt."));
      WN *stmt_copy = WN_COPY_Tree(stmt);

      moved_once = TRUE;
      Duplicate_alias_info(alias_mgr, stmt, stmt_copy);
      //  Put stmt_copy in the moved_by_delayability map
      //  
      moved_by_delayability[stmt_copy] = TRUE;
      if(move_within_bb)
        immoveable_stmts[stmt_copy] = TRUE;

      if(WN_operator(where) == OPR_LABEL)
        Insert_wn_after(stmt_copy, where);
      else 
        Insert_wn_before(stmt_copy, where);
    }

    if(not_moved == FALSE)
      to_remove.push_back(stmt);
    else {
      immoveable_stmts[stmt] = TRUE;
    }
    if(moved_once == TRUE)
      num_moved++;
  }

  if(Enable_Liveness_Sensitive) {
    if(liveness_valid == TRUE)
      Destroy_liveness_info();
  }

  for(INT i=0; i<to_remove.size(); i++) {
    Remove_wn(to_remove[i]);
  }
  
  return num_moved;
}

//  Delayability_onepass: One pass of delayability.
//  See inline comments for details.
//  
static INT Delayability_onepass(WN *wn, BOOL move_within_bb=FALSE)
{
  //  Clear the maps used for storing delayed statements' info.
  //  
  label_map.clear();
  delayed_map.clear();

  //  Number the statements, so that set of statements can
  //  be represented using IDX_32_SETs. Global variables num_stmts, stmt_map,
  //  and stmt_rev_map are set by this.
  //  
  Number_stmts(wn);

  //  Variable to keep track of number of statements moved.
  //  
  INT num_moved = 0;

  MEM_POOL_Initialize(&pde_pool, "dce pool", FALSE);
  MEM_POOL_Push(&pde_pool);

  //  Compute the data flow information using Delayability. After this,
  //  delayed_map will have the set of statements that can be delayed
  //  until a given statement.
  //  I am calling MLSolve which works only mid-level WHIRL. Make sure
  //  pde is called on ML WHIRL.
  //  
  DelayabilityClient *dclient = new DelayabilityClient();
  DelayabilityState *dstate = new DelayabilityState();

  DFSolver<DelayabilityClient, DelayabilityState> d(dclient);
  d.Solve(TRUE, wn, dstate);

  //  Use the information in delayed_map to find where to insert 
  //  statements and insert them there.
  //  
  num_moved = Insertion_points(wn, move_within_bb);
  MEM_POOL_Pop(&pde_pool);
  MEM_POOL_Delete(&pde_pool);

  printf("Num moved down : %d\n", num_moved);
  return num_moved;
}

static void Do_delayability(WN *wn)
{
  INT iter = 0;
  while(1) {
    if(Delayability_onepass(wn) < 1)
      break;

    if(WOPT_PDE_Max_Iters > 0)
      iter++;

    if((WOPT_PDE_Max_Iters > 0) && (iter >= WOPT_PDE_Max_Iters))
      break;
  }
}

WN *Perform_PDE(WN *wn, ALIAS_MANAGER *am)
{
  alias_mgr = am;

  INT iter = 0;

  moved_by_delayability.clear();
  moved_by_floatability.clear();

  Order_Options = WOPT_PDE_Options & 14;
  Enable_Liveness_Sensitive = WOPT_PDE_Options & 32;
  
  if(Order_Options == 0) {
    Do_delayability(wn);
  }
#if 0
  else if(Order_Options == 2) {
    do_floatability(wn);
  }
  else if(Order_Options == 4) {
    Do_delayability(wn);
    immoveable_stmts.clear();
    while(Delayability_onepass(wn, TRUE) > 0);
    do_floatability(wn);
    while(floatability_onepass(wn, TRUE) > 0);
  }
  else if(Order_Options == 6) {
    do_floatability(wn);
    Do_delayability(wn);
  }
  else if(Order_Options == 8) {
    immoveable_stmts.clear();
    while(Delayability_onepass(wn, TRUE) > 0);
    floatability_onepass(wn, TRUE);
  }
  else if(Order_Options == 10) {
    immoveable_stmts.clear();
    while(floatability_onepass(wn, TRUE) > 0);
    while(Delayability_onepass(wn, TRUE) > 0);
  }
#endif


  return wn;
}
