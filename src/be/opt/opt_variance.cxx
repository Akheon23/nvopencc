/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifdef TARG_NVISA

// ====================================================================
//
// Module: opt_variance.cxx
//
//    This module is for implementing Variance Analysis on
//    SSA-based IR. The whole analysis is implemented as a
//    class CR_VarianceMap. The result of the analysis is
//    stored in the class, and several interfaces are provided to
//    retrieve the variance information for any
//    CODEREP and STMTREP.
// 
//      ThreadIdx_VAR_STATE CR_is_variant (CODEREP *cr);
//      ThreadIdx_VAR_STATE STMT_is_variant (STMTREP *cr);
//
//    LDU analysis is also done based on the variance analysis and
//    read only analysis. The following interface is provided for
//    retrieving the LDU analysis result,
//
//         BOOL CR_is_uniform (CODEREP *cr);
//
//    This is for checking an IVAR/ILOAD CODEREP is read only and
//    thread invariant. 
//
// ====================================================================
//
#include "cxx_memory.h"
#include "intrn_info.h"
#include "opt_cfg.h"
#include "opt_sym.h"
#include "opt_htable.h"
#include "opt_ssa.h"
#include "opt_mu_chi.h"
#include "bb_node_set.h"
#include "opt_bb.h"

#include "config_wopt.h"
#include "config_debug.h"
#include "opt_htable.h"
#include "opt_variance.h"
#include "opt_memory_space.h"

// for recording the current CODEREP Thread ID variant map
static CR_VarianceMap *_current_CR_var_map = NULL;

// Interface for setting the current CODEREP ThreadID Variant Map
static void 
Set_current_CR_var_map (CR_VarianceMap *var_map) 
{ 
  _current_CR_var_map = var_map; 
}

CR_VarianceMap *
Get_current_CR_var_map (void)
{
  return _current_CR_var_map;
}

// Interface for retrieving the variance state of a CODEREP
ThreadIdx_VAR_STATE
CR_is_thread_variant (CODEREP *cr)
{
  if (_current_CR_var_map == NULL) {
    // if the analysis is not done, fall into conservative
    return ThreadIdx_XYZ_Variant;
  }

  return _current_CR_var_map->CR_get_tid_var_state(cr);
}

// Interface for retrieving the variance state of a STMTREP
ThreadIdx_VAR_STATE
STMT_is_thread_variant (STMTREP *stmt)
{
  if (_current_CR_var_map == NULL) {
    // if the analysis is not done, fall into conservative
    return ThreadIdx_XYZ_Variant;
  }

  return _current_CR_var_map->STMT_get_tid_var_state(stmt);
}

// interface for check the variance state of a BB_NODE's end condition
//
ThreadIdx_VAR_STATE BB_get_tid_var_state (BB_NODE *bb)
{
  // if the block fall through, the condition does affect on the variance state
  if (bb->Succ() == NULL || bb->Succ()->Len() == 1) {
    return ThreadIdx_Invariant;
  }
 
  if (_current_CR_var_map == NULL) {
    // if the analysis is not done, fall into conservative
    return ThreadIdx_XYZ_Variant;
  }

  STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
  const STMTREP *stmt;

  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    OPERATOR stmt_opr = stmt->Opr();

    if (stmt_opr == OPR_TRUEBR || stmt_opr == OPR_FALSEBR) {
      return _current_CR_var_map->CR_get_tid_var_state (stmt->Rhs());
    }
  }

  return ThreadIdx_XYZ_Variant;  // fall into conservative
}

// Traverse all the affected coderep and apply the func
void 
CODEREP_AFFECT_INFO::Walk_all_affected_coderep (CODEREP_DEF_USE_WALK *tr,
                                                BOOL mu_included,
                                                STACK<CODEREP *> *work_list)
{
  if (_cr_affected_list != NULL) {
    for (INT32 i = 0; i < _cr_affected_list->Elements(); i++) {
      tr->Walk_coderep_def_use(_self, _cr_affected_list->Top_nth(i), work_list);
    } // for
  } // if

  if (_stmt_affected_list != NULL) {

    for (INT32 i = 0; i < _stmt_affected_list->Elements(); i++) {

      const STMTREP *parent_stmt = _stmt_affected_list->Top_nth(i);

      CHI_NODE *cnode;
      CHI_LIST_ITER chi_iter;
      FOR_ALL_NODE(cnode, chi_iter, Init(parent_stmt->Chi_list())) {
        if (cnode->Live() == FALSE) {
          continue;
        }
        tr->Walk_coderep_def_use(_self, cnode->RESULT(), work_list);      
      } // FOR_ALL_NODE

    } // for

  } // if

}
    

// Add parent relations between a CR and its kids.
// For IVAR CR, consider the mu_node as its kid.
void 
CODEREP_AFFECT_MAP::Add_coderep_info (CODEREP *cr)
{
  if (cr->Kind() == CK_OP) {
    for (INT32 i = 0; i < cr->Kid_count(); i++) {
      Add_affected_coderep (cr->Get_opnd(i), cr);
      Add_coderep_info (cr->Get_opnd(i));
    }
  } else if (cr->Kind() == CK_IVAR) {
    // Consider the address expression CR
    if (cr->Ilod_base() != NULL) {
      Add_affected_coderep (cr->Ilod_base(), cr);
      Add_coderep_info (cr->Ilod_base());
    } else {
      Add_affected_coderep (cr->Istr_base(), cr);
      Add_coderep_info (cr->Istr_base());
    }

    // Consider the MU node, IVAR node has at most one mu node
    if (cr->Ivar_mu_node() && cr->Ivar_mu_node()->OPND()) {
      Add_affected_coderep (cr, cr->Ivar_mu_node()->OPND());
    }

    if (cr->Opr() == OPR_MLOAD) {
      CODEREP *num_byte = cr->Mload_size();
      Add_affected_coderep (num_byte, cr);
      Add_coderep_info (num_byte);
    }
    if (cr->Opr() == OPR_ILOADX) {
      Add_affected_coderep (cr->Index(), cr);
      Add_coderep_info (cr->Index());
    }
  }
}

// Traverse statements in a BB_NODE, and record the
// parent relations, and condition CODEREP for the BB_NODE.
void
CODEREP_AFFECT_MAP::BB_setup_coderep_affect_info (BB_NODE *bb)
{
  STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
  const STMTREP *stmt;

  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    OPERATOR stmt_opr = stmt->Opr();

    if (stmt_opr == OPR_PRAGMA) {
      // ignore the pragma and entry
      continue;
    }

    if (stmt_opr == OPR_OPT_CHI) {
      if (Current_PU_is_Global()) {
        // For global CUDA function, the definition from
        // entry can be ignored, since those are from
        // the host call-site
        continue;
      }
    }

    if (stmt->Has_mu()) {
      MU_NODE *mnode;
      MU_LIST_ITER mu_iter;
      if (stmt->Mu_list()) {
        FOR_ALL_NODE(mnode, mu_iter, Init(stmt->Mu_list())) {
          if (mnode->OPND()) {  // Check if the MU has been deleted.
            Add_affected_stmtrep (mnode->OPND(), stmt);
          }
        }
      }
    }

    if (stmt->Rhs()) {
      if (OPERATOR_is_scalar_store(stmt_opr)) {
        Add_affected_coderep (stmt->Rhs(), stmt->Lhs());
      } else {
        Add_affected_stmtrep (stmt->Rhs(), stmt);
      }
      Add_coderep_info (stmt->Rhs());
    }

    if (stmt->Lhs()) {
      Add_coderep_info (stmt->Lhs());
      if (stmt->Has_chi() && stmt->Chi_list() != NULL) {
        // avoid adding useless parent
        Add_affected_mu_stmtrep (stmt->Lhs(), stmt);
      }
    }

    if (stmt->Has_chi()) {
      CHI_NODE *cnode;
      CHI_LIST_ITER chi_iter; 
      FOR_ALL_NODE(cnode, chi_iter, Init(stmt->Chi_list())) {
        if (cnode->Live()) { // Check if the chi_node has been deleted.
          // Consider chi's result is parent of chi's opnd, i.e.,
          // if the opnd's changed, the result has to be changed
          Add_affected_coderep (cnode->OPND(), cnode->RESULT());
        }
      }
    }

  }

  // Consider PHI_NODE's RESULT as OPNDs' parent
  PHI_NODE *phi;
  PHI_LIST_ITER phi_iter;
  FOR_ALL_ELEM (phi, phi_iter, Init(bb->Phi_list())) {
    if (phi->Live() == FALSE) {
      continue;
    }
    for (INT32 i = 0; i < phi->Size(); i++) {
      Add_affected_coderep (phi->OPND(i), phi->RESULT());
    }
  } // FOR_ALL_ELEM (phi, phi_iter, Init(bb->Phi_list()))

}

// Setup CODEREP's parent information, and
// condition CR for BB_NODEs with condition branches.
void 
CODEREP_AFFECT_MAP::CFG_setup_coderep_affect_info (void)
{
  CFG_ITER cfg_iter(_codemap->Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE( bb, cfg_iter, Init() ) {
    BB_setup_coderep_affect_info (bb);
  }
}


// Check if a BB_NODE belongs to the stack_list.
static BOOL
Element_of (BB_NODE *node, STACK<BB_NODE *> *stack_list)
{
  if (stack_list == NULL) {
    return FALSE;
  }

  for (INT32 i = 0; i < stack_list->Elements(); i++) {

    BB_NODE *b= stack_list->Top_nth(i);
    if (b == node) {
      return TRUE;
    }
  }

  return FALSE;
}

// Recursively add all control dependences for a BB_NODE.
static STACK<BB_NODE *> *
Add_iter_control_dep (STACK<BB_NODE *> *control_dep_list,
                      BB_NODE *control_dep_bb,
                      MEM_POOL *mem_pool)
{
  if (control_dep_list == NULL) {
    control_dep_list = new STACK<BB_NODE *>(mem_pool);
    control_dep_list->Push(control_dep_bb);
  } else if (Element_of (control_dep_bb, control_dep_list)) {
    return control_dep_list;
  } else { 
    control_dep_list->Push(control_dep_bb);
  }

  BB_NODE *cb;
  BB_NODE_SET_ITER rcfg_iter;
  FOR_ALL_ELEM(cb, rcfg_iter, Init(control_dep_bb->Rcfg_dom_frontier())) {
    control_dep_list = Add_iter_control_dep (control_dep_list, cb, mem_pool);
  }

  return control_dep_list;
} 

// Compute iterative control dependence for a BB_NODE.
static void
BB_compute_iter_control_dep (BB_NODE *bb,
                             BB_CONTROL_INFO *BB_info,
                             MEM_POOL *mem_pool)
{
  BB_NODE *cb;
  BB_NODE_SET_ITER rcfg_iter;
  FOR_ALL_ELEM(cb, rcfg_iter, Init(bb->Rcfg_dom_frontier())) {
    BB_info[bb->Id()].iter_control_dep = 
	Add_iter_control_dep (BB_info[bb->Id()].iter_control_dep, cb, mem_pool);
  }
}

// Record BB with syncthreads calls. 
// information to keep is the unique set of control dependence, so
// if BB with same control dependence set is checked to avoid
// redundancy.
void
CR_VarianceMap::Add_bb_with_sync_threads(BB_NODE *bb)
{
  if (_BB_with_sync_threads == NULL) {
    _BB_with_sync_threads = new STACK<BB_NODE *>(&_mem_pool);
    _BB_with_sync_threads->Push(bb);
    return;
  }

  BB_NODE_SET *bb_ctrl = bb->Rcfg_dom_frontier();

  // go through the current list, and find if the 
  // control dependence SET of BB matches with any
  // one on the list, if yes, return.
  for (INT32 i = 0; i < _BB_with_sync_threads->Elements(); i++) {

    BB_NODE *b= _BB_with_sync_threads->Top_nth(i);
    BB_NODE_SET *in_list_ctrl = b->Rcfg_dom_frontier();
    
    BB_NODE_SET *diff = bb_ctrl->Difference(in_list_ctrl, &_mem_pool);
    if (diff != NULL && diff->EmptyP()) {
      return;
    }
  }

  _BB_with_sync_threads->Push(bb);
}


// Traverse statements in a BB_NODE, and record the
// condition CODEREP for the BB_NODE, and if there is
// an atomic intrinsic call, set the TID variant state
// for all CODEREPs affected to the worst case.
void
CR_VarianceMap::BB_initialize_control_var_info (BB_NODE *bb, 
                                                BB_CONTROL_INFO *bb_info, 
                                                STACK<CODEREP *> *work_list)
{
  STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
  const STMTREP *stmt;

  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    OPERATOR stmt_opr = stmt->Opr();

    if (stmt_opr == OPR_PRAGMA ||
      // entry can be ignored for __global__ routine
      stmt_opr == OPR_OPT_CHI && Current_PU_is_Global()) {
      continue;
    } else if (stmt_opr == OPR_TRUEBR || stmt_opr == OPR_FALSEBR) {
      // Keep the condition CODEREP for a branch BLOCK
      bb_info[bb->Id()].cond_coderep = stmt->Rhs();
      continue;
    } 

    if (stmt_opr == OPR_INTRINSIC_CALL &&
        stmt->Rhs()->Intrinsic() == INTRN_SYNCTHREADS) {
      bb_info[bb->Id()].contain_syncthread_calls = TRUE;
      continue;
    }

    if (!(stmt_opr == OPR_INTRINSIC_CALL && INTRN_is_atomic(stmt->Rhs()->Intrinsic())
          || stmt_opr == OPR_CALL
          || stmt_opr == OPR_ICALL
          || stmt_opr == OPR_VFCALL
          || stmt_opr == OPR_PICCALL)) {
      continue;
    }

    // Need to fall into conservative for Atomic call and other user calls
    if (stmt->Has_chi()) {
      CHI_NODE *cnode;
      CHI_LIST_ITER chi_iter; 
      FOR_ALL_NODE(cnode, chi_iter, Init(stmt->Chi_list())) {
        if (cnode->Live()) {
          if (CR_add_tid_var_state (cnode->RESULT(), ThreadIdx_XYZ_Variant)) {
            // Add to the work_list
            work_list->Push(cnode->RESULT());
          }
        }
      } // FOR_ALL_NODE

    } // if

  } // FOR_ALL_NODE(stmt, stmt_iter, Init())

  if (WOPT_Enable_LDU_Use_Syncthreads) {
    if (bb_info[bb->Id()].contain_syncthread_calls) {
      // if the bb contains any syncthread calls, the bb either never be
      // entered, or always convergent, so its control dep should always 
      // be thread invariant.
      BB_NODE *cb;
      BB_NODE_SET_ITER rcfg_iter;
      INT32 control_dep_size = bb->Rcfg_dom_frontier()->Size();
      if (control_dep_size == 1) {
        // single control dependent itself must be thread invariant
        FOR_ALL_ELEM(cb, rcfg_iter, Init(bb->Rcfg_dom_frontier())) {
          bb_info[cb->Id()].cond_coderep_must_be_tid_inv = TRUE; 
        }
      }
      else if (control_dep_size > 1) {
        // if the size > 1, need to consider the whole set together
        Add_bb_with_sync_threads(bb);
      }
    }
  }

}

// Check if a list of BB_NODEs contains a Set of BB_NODEs
BOOL
BB_List_contain_bb_set(STACK<BB_NODE *> *bb_list, BB_NODE_SET *bb_set)
{
   BB_NODE *cb;
   BB_NODE_SET_ITER rcfg_iter;
   FOR_ALL_ELEM(cb, rcfg_iter, Init(bb_set)) {
     if (Element_of(cb, bb_list) == FALSE) {
       return FALSE;
     }
   }
   return TRUE;  
}

// Compute the BB_NODE list - the BB_NODE set
STACK<BB_NODE *> *
BB_List_minus_bb_set(STACK<BB_NODE *> *bb_list, BB_NODE_SET *bb_set, MEM_POOL *pool)
{
  STACK<BB_NODE *> *result = new STACK<BB_NODE *>(pool);
  for (INT32 i = 0; i < bb_list->Elements(); i++) {
    BB_NODE *b = bb_list->Top_nth(i);
    if (bb_set->MemberP(b)) {
      continue;
    }
    result->Push(b);
  }

  return result;
}

// Use syncthreads semantics to remove some control dependence condtions
// from ierative control dependences of a BB_NODE
void
CR_VarianceMap::BB_refine_iter_control_dep (BB_NODE *bb, 
					    BB_CONTROL_INFO *bb_info, 
					    MEM_POOL *mem_pool)
{
  STACK<BB_NODE *> *control_dep = bb_info[bb->Id()].iter_control_dep;
  for (INT32 i = 0; i < _BB_with_sync_threads->Elements(); i++) {
    BB_NODE *b = _BB_with_sync_threads->Top_nth(i);
    if (BB_List_contain_bb_set(control_dep, b->Rcfg_dom_frontier())) {
      // printf("BB %d - control-dep of (B%d, size %d)\n", bb->Id(), b->Id(),
      //        b->Rcfg_dom_frontier()->Size());
      fflush(stdout);
      control_dep = BB_List_minus_bb_set(control_dep, b->Rcfg_dom_frontier(), mem_pool);
    }
  }

  bb_info[bb->Id()].iter_control_dep = control_dep;
}

// Setup CODEREP's parent information, and
// condition CR for BB_NODEs with condition branches.
void 
CR_VarianceMap::CFG_init_control_var_info (BB_CONTROL_INFO *bb_info,
                                           MEM_POOL *mem_pool,
                                           STACK<CODEREP *> *work_list)
{
  CFG_ITER cfg_iter(_codemap->Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE( bb, cfg_iter, Init() ) {
    BB_compute_iter_control_dep (bb, bb_info, mem_pool);
    BB_initialize_control_var_info (bb, bb_info, work_list);
  }

  if (WOPT_Enable_LDU_Use_Syncthreads && _BB_with_sync_threads != NULL) {
    FOR_ALL_NODE( bb, cfg_iter, Init() ) {
      BB_refine_iter_control_dep (bb, bb_info, mem_pool);
    }
  }
    
}



// Check if the control dependence ThreadIdx variant state changed.
BOOL 
CR_VarianceMap::BB_control_tid_var_state_changed(BB_NODE *bb, BB_CONTROL_INFO *bb_info)
{
  if (bb_info[bb->Id()].iter_control_dep == NULL) {
    return FALSE;
  }

  ThreadIdx_VAR_STATE tmp_var_state = ThreadIdx_Invariant;

  for (INT32 i = 0; i < bb_info[bb->Id()].iter_control_dep->Elements(); i++) {

    BB_NODE *cb= bb_info[bb->Id()].iter_control_dep->Top_nth(i);
    if (bb_info[cb->Id()].cond_coderep_must_be_tid_inv) {
      // ignore control dep which must be thread invariant
      continue;
    }
    CODEREP *cond_cr = bb_info[cb->Id()].cond_coderep;
    if (cond_cr == NULL) {
      // un-conditional go-to
      continue;
    }
    tmp_var_state = (ThreadIdx_VAR_STATE)(tmp_var_state | CR_get_tid_var_state(cond_cr));
  }

  if (tmp_var_state == ThreadIdx_Invariant) {
    return FALSE;
  }

  return BB_add_control_tid_var_state(bb, bb_info, tmp_var_state);
}

// Find all codereps which are one of the ThreadIdx variables.
void 
CR_VarianceMap::Init_tid_var_list (STACK<CODEREP *> *work_list)
{
  CODEREP_ITER cr_iter;
  CODEREP *cr;
  CODEMAP *htable = _codemap;

  work_list->Clear();

  // traverse the aux_stab first, find threadIdx CRs
  AUX_ID i;
  AUX_STAB_ITER aux_stab_iter(_codemap->Opt_stab());
  FOR_ALL_NODE(i, aux_stab_iter, Init()) {
    AUX_STAB_ENTRY *aux = _codemap->Opt_stab()->Aux_stab_entry(i);
    if (strcmp(aux->Base_name(),"threadIdx") == 0) {
      mINT64 offset = aux->St_ofst();
     
      FOR_ALL_NODE(cr, cr_iter, Init(aux->Cr_list())) {
        if (offset == 0) {
          CR_add_tid_var_state(cr, ThreadIdx_X_Variant);
        } else if (offset == 4) {
          CR_add_tid_var_state(cr, ThreadIdx_Y_Variant);
        } else {
          Is_True(offset == 8, ("Illegal Thread ID\n")); 
          CR_add_tid_var_state(cr, ThreadIdx_Z_Variant);
        }
        work_list->Push(cr);
      }
    }
  }
}

// Add the TID variant state of defCR to the TID variant state of useCR, and
// if changed, add the useCR to the work_list.
void
CR_VarianceMap::Walk_coderep_def_use(CODEREP *defCR, CODEREP *useCR, STACK<CODEREP *> *work_list)
{
  if (CR_add_tid_var_state(useCR, CR_get_tid_var_state(defCR))) {
    work_list->Push(useCR);
  }
}


// The main interface routine for setup the ThreadIdx Variance Information.
void 
CR_VarianceMap::Setup_coderep_tid_var_info (void)
{
  MEM_POOL bb_mem_pool;

  MEM_POOL_Initialize(&bb_mem_pool, "CR_variance_tmp_pool", FALSE);
  MEM_POOL_Push(&bb_mem_pool);

  STACK<CODEREP *> *work_list = CXX_NEW(STACK<CODEREP*>(&bb_mem_pool), &bb_mem_pool);

  INT32 total_BBs = _codemap->Cfg()->Total_bb_count() + 1;
    
  BB_CONTROL_INFO *bb_info = (BB_CONTROL_INFO *)MEM_POOL_Alloc (&bb_mem_pool, total_BBs * sizeof(BB_CONTROL_INFO));

  // Initialized bbs' control info
  memset (bb_info, 0, total_BBs * sizeof(BB_CONTROL_INFO));

  // go through all VAR codereps, collect all the original ThreadIdx codereps.
  Init_tid_var_list (work_list);

  // For each BB_NODE, its control ThreadIdx variance information is initialized,
  // and condtion CODEREP is recorded if the BB_NODE has a condition branch at the end.
  //
  // Also check unknown calls and intrinsic atomic calls to collect affected CRs to work_list 
  CFG_init_control_var_info (bb_info, &bb_mem_pool, work_list);

  // Traverse the whole program unit to record relations between CODEREP and
  // CODEREPs/STMTREPs, for efficiently tracking all CODEREPs in the 
  // program unit.
  //
  CODEREP_AFFECT_MAP *CR_affect_map = new CODEREP_AFFECT_MAP (_codemap);
  CR_affect_map->CFG_setup_coderep_affect_info();

  while (work_list->Elements() != 0) {

    do {
      // For each variant CR in the work_list, find its parents, and 
      // add CR's variant state to its parent's state, and if parent's 
      // state is changed, add its parent to the work_list
      //
      CODEREP *cr = work_list->Pop();
      CR_affect_map->Walk_all_affected_coderep (cr, TRUE, this, work_list);

    } while (work_list->Elements() != 0);

    CFG_ITER cfg_iter(_codemap->Cfg());
    BB_NODE *bb;
    FOR_ALL_NODE( bb, cfg_iter, Init() ) {

      if (bb_info[bb->Id()].contain_syncthread_calls) {
        // control variance should not affect on BB_NODEs with syncthreads calls
        continue;
      }

      if (BB_control_tid_var_state_changed (bb, bb_info) == FALSE) {
        continue;
      }

      ThreadIdx_VAR_STATE bb_control_state = bb_info[bb->Id()].control_tid_var_state;

      // Traverse statements in the block, applied bb_control_state to
      // each of CRs
      STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
      const    STMTREP   *stmt;
      FOR_ALL_NODE(stmt, stmt_iter, Init()) {
        if (OPERATOR_is_scalar_store (stmt->Opr())) { 
          if (CR_add_tid_var_state(stmt->Lhs(), bb_control_state)) {
            work_list->Push(stmt->Lhs());
          }
        }

        CHI_NODE *cnode;
        CHI_LIST_ITER chi_iter;
        FOR_ALL_NODE(cnode, chi_iter, Init(stmt->Chi_list())) {
          if (cnode->Live() == FALSE) {
            continue;
          }
          CODEREP *chi_cr = cnode->RESULT();      
          if (CR_add_tid_var_state(chi_cr, bb_control_state)) {
            work_list->Push(chi_cr);
          }
        }
      } // FOR_ALL_NODE(stmt, stmt_iter, Init())

      // No need to consider PHI_NODEs here, since PHI_NODE
      // is eventually ignored in the emitter.

    } // FOR_ALL_NODE( bb, cfg_iter, Init() )        

  } // while (work_list->Elements() != 0)

  // Put the tid_var_state into CODEREP
  // and check and set if a VAR CODEREP is read only
  for (INT32 i = 1; i < _codemap->Coderep_id_cnt(); i++) {
    CODEREP *cr;
    ThreadIdx_VAR_STATE state = CR_get_tid_var_state(i);
    cr = CR_affect_map->CR_get_coderep(i);
    if (cr != NULL
        && cr->Kind() == CK_VAR
        && cr->Get_defstmt() != NULL
        && cr->Get_defstmt()->Opr() == OPR_OPT_CHI
        && cr->Is_flag_set(CF_IS_ZERO_VERSION) == FALSE) {
      if (Current_PU_is_Global()) {
        // for __device__ routine, this is not correct
        CR_set_read_only (cr);
      }
    }
  }

  if (WOPT_Enable_CR_Memory_Space_Check) {
    // build memory state
     _cr_mem_map->Setup_coderep_memory_kind (CR_affect_map);
  }

  // free the forward df information
  delete CR_affect_map;

  // Free the memory for BB info
  MEM_POOL_Pop(&bb_mem_pool);
  MEM_POOL_Delete(&bb_mem_pool);
}

// return the variant state for a STMTREP
// by combining variant states of all modified CODEREPs in the STMTREP
//
ThreadIdx_VAR_STATE
CR_VarianceMap::STMT_get_tid_var_state (STMTREP *stmt)
{
  OPERATOR stmt_opr = stmt->Opr();

  if (stmt_opr == OPR_OPT_CHI) {
    // ignore the entry
    return ThreadIdx_Invariant;
  }

  ThreadIdx_VAR_STATE stmt_state = ThreadIdx_Invariant;

  if (OPERATOR_is_scalar_store(stmt_opr)) {
    stmt_state = (ThreadIdx_VAR_STATE)(stmt_state | CR_get_tid_var_state(stmt->Lhs()));
  }

  if (stmt->Has_chi()) {
    CHI_NODE *cnode;
    CHI_LIST_ITER chi_iter; 
    FOR_ALL_NODE(cnode, chi_iter, Init(stmt->Chi_list())) {
      if (cnode->Live()) { // Check if the chi_node has been deleted.
        stmt_state = (ThreadIdx_VAR_STATE)(stmt_state | CR_get_tid_var_state(cnode->RESULT()));
      }
    }
  }

  return stmt_state;
}


// The constructor, allocate and initialized needed memory
// and set the current CR var map
CR_VarianceMap::CR_VarianceMap (CODEMAP *codemap)
{
  _codemap = codemap;
  MEM_POOL_Initialize(&_mem_pool, "CR_variance_pool", FALSE);
  MEM_POOL_Push(&_mem_pool);
  _total_CRs = codemap->Coderep_id_cnt() + 1; 
  _CR_info = (struct _CODEREP_INFO_ *)MEM_POOL_Alloc (&_mem_pool, _total_CRs * sizeof(struct _CODEREP_INFO_));
  _BB_with_sync_threads = NULL;
  // initialized to ThreadIdx_Invariant and not read_only
  memset (_CR_info, 0, _total_CRs * sizeof(struct _CODEREP_INFO_));

   _cr_mem_map = new CR_MemoryMap(codemap);

  // make this as the current_CR_var_map
  Set_current_CR_var_map (this); 
}

// The destructor, free up the memory space, and 
// reset the current_CR_var_nap
CR_VarianceMap::~CR_VarianceMap(void)
{
  MEM_POOL_Pop(&_mem_pool);
  MEM_POOL_Delete(&_mem_pool);

  delete _cr_mem_map;

  // reset the current_CR_var_map
  Set_current_CR_var_map (NULL); 
}


#endif // ifdef TARG_NVISA

