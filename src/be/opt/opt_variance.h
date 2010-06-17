/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef opt_variance_INCLUDED
#define opt_variance_INCLUDED


#ifdef TARG_NVISA

#include "opt_ldu.h"

class CR_MemoryMap;

// for annotating a variable's ThreadIdx variant state.
// !!! Changes to this enum type will need to consider how
//     states are accumulated or combined. The current declaration
//     requires the bitwise OR operations be used for accumlating
//     or combining two or more states.
//
typedef enum {
  ThreadIdx_Invariant       = 0,
  ThreadIdx_X_Variant       = 1,
  ThreadIdx_Y_Variant       = 2,
  ThreadIdx_XY_Variant      = 3,
  ThreadIdx_Z_Variant       = 4,
  ThreadIdx_XZ_Variant      = 5,
  ThreadIdx_YZ_Variant      = 6,
  ThreadIdx_XYZ_Variant     = 7
} ThreadIdx_VAR_STATE;

// for keeping information about a BB_NODE
typedef struct BB_control_info {
   CODEREP *cond_coderep;
   STACK<BB_NODE *> *iter_control_dep;
   ThreadIdx_VAR_STATE control_tid_var_state : 3;
   BOOL contain_syncthread_calls: 1;
   BOOL cond_coderep_must_be_tid_inv : 1;
} BB_CONTROL_INFO;

// This class serves as an interface for implementing a call-back function 
// to walk through the def-use.
//
class CODEREP_DEF_USE_WALK
{
public:
  virtual void Walk_coderep_def_use (CODEREP *defCR, CODEREP *useCR, STACK<CODEREP *> *work_list) {};
};


// To keep relations between one CODEREP and other CODEREPs or STMTREPs,
// this is essentially to track all references to the CODEREP,
// or forward data-flow edges. The information for CODEREP from zero-version
// is more conservative.
//
class CODEREP_AFFECT_INFO
{    
private:
    STACK<CODEREP *> *_cr_affected_list;   // list of CODEREPs are affected by any changes
                                           // to this CODEREP
    
    STACK<const STMTREP *> *_stmt_affected_list; // list of STMTREPs which are affected 
                                                 // when this CODEREP is changed.

    STACK<const STMTREP *> *_mu_stmt_affected_list; // list of STMTREPs which are affected
                                                    // when this CODEREP appears on the left
                                                    // of the statement, i.e.,  STID, ISTORE,
                                                    // for some analysis, following the implicit
                                                    // data-flow may not make sense.

    CODEREP *_self;          // for convenience

public:
    CODEREP_AFFECT_INFO(CODEREP *cr) 
    { 
      _cr_affected_list = NULL;
      _stmt_affected_list = NULL;
      _mu_stmt_affected_list = NULL;
      _self = cr;
    }

    void Add_affected_stmtrep (const STMTREP *parent, MEM_POOL *mem_pool)
    {
      if (_stmt_affected_list == NULL) {
        _stmt_affected_list = new STACK<const STMTREP *>(mem_pool);
      }
      _stmt_affected_list->Push (parent);
    }

    void Add_affected_coderep (CODEREP *parent, MEM_POOL *mem_pool)
    {
      if (_cr_affected_list == NULL) {
        _cr_affected_list = new STACK<CODEREP *>(mem_pool);
      }
      _cr_affected_list->Push (parent);
    }

    void Add_affected_mu_stmtrep (const STMTREP *parent, MEM_POOL *mem_pool)
    {
      if (_mu_stmt_affected_list == NULL) {
        _mu_stmt_affected_list = new STACK<const STMTREP *>(mem_pool);
      }
      _mu_stmt_affected_list->Push (parent);
    }

    // Traverse all the affected coderep and apply the func
    void Walk_all_affected_coderep (CODEREP_DEF_USE_WALK *tr,
                                    BOOL mu_included,
                                    STACK<CODEREP *> *work_list);

    CODEREP *Get_coderep() { return _self; }
};


// This is a container class serving as interfaces for building and retrieving
// information about each CODEREP's refereces.
// 
// To steps of using this class are,
//                  CODEREP_AFFECT_MAP *ca_map = new CODEREP_AFFECT_MAP(htable);
//                  ca_map->CFG_setup_coderep_affect_info();
//
//  At this point, all CODEREPs' forward data-flow information is available,
//  and the following interface can be used to walk the affected CODEREP list,
//
//                 ca_map->Walk_all_affected_coderep (CODEREP *cr, 
//                                                    CODEREP_DEF_USE_WALK *tr,
//                                                    STACK<CODEREP *> *work_list)
//
//  Here, the class CODEREP_DEF_USE_WALK is a virtual class, a virtual function.
//  Walk_coderep_def_use() should be implemented in the sub-class to
//  apply things to the affected CODEREP. For sample implementation,
//  see CR_VarianceMap->Check_coderep_def_use().
//
//  When it is done, 
//                  delete ca_map;
//
class CODEREP_AFFECT_MAP
{
private:
  CODEMAP *_codemap;
  MEM_POOL _mem_pool;
  INT32 _total_CRs;       // total number of CODEREPs in the program unit
  CODEREP_AFFECT_INFO **_CR_info;

private:

  // Track references for IVAR and OP CODEREP
  void Add_coderep_info (CODEREP *cr);

  // Track CODEREPs in a BB_NODE
  void BB_setup_coderep_affect_info (BB_NODE *bb);

  // Add a STMTREP as an affected one by the child
  void 
  Add_affected_stmtrep (CODEREP *child, const STMTREP *parent)
  {
    if (child->Kind() == CK_CONST || child->Kind() == CK_RCONST) {
      return;
    }

    CODEREP_AFFECT_INFO *info = CR_get_var_info (child);
    info->Add_affected_stmtrep(parent, &_mem_pool);
  }

  // Add a CODEREP as an affected on by the child
  void 
  Add_affected_coderep (CODEREP *child, CODEREP *parent)
  {
    if (child == parent || child->Kind() == CK_CONST || child->Kind() == CK_RCONST) {
      return;
    }

    CODEREP_AFFECT_INFO *info = CR_get_var_info (child);
    info->Add_affected_coderep(parent, &_mem_pool);
  }

  // Add a STMTREP as an affected one by the left child
  void
  Add_affected_mu_stmtrep (CODEREP *child, const STMTREP *parent)
  {
    if (child->Kind() == CK_CONST || child->Kind() == CK_RCONST) {
      return;
    }

    CODEREP_AFFECT_INFO *info = CR_get_var_info (child);
    info->Add_affected_mu_stmtrep(parent, &_mem_pool);
  }

  // Create CODEREP_AFFECT_INFO for the cr if not create yet, and
  // return the CODEREP_AFFECT_INFO for the cr.
  CODEREP_AFFECT_INFO *
  CR_get_var_info (CODEREP *cr)
  {
    if (_CR_info[cr->Coderep_id()] == NULL) {
      _CR_info[cr->Coderep_id()] = new CODEREP_AFFECT_INFO(cr);
    }
    return _CR_info[cr->Coderep_id()];
  }


public:
  CODEREP_AFFECT_MAP(CODEMAP *cmap)
  {
    _codemap = cmap;
    MEM_POOL_Initialize(&_mem_pool, "CODEREP_AFFECT_MAP_pool", FALSE);
    MEM_POOL_Push(&_mem_pool);
    
    _total_CRs = cmap->Coderep_id_cnt() + 1;
    _CR_info = (CODEREP_AFFECT_INFO **)MEM_POOL_Alloc (&_mem_pool, _total_CRs * sizeof(CODEREP_AFFECT_INFO *));
    memset (_CR_info, 0, _total_CRs * sizeof(CODEREP_AFFECT_INFO *));
  }

  ~CODEREP_AFFECT_MAP() 
  {
    MEM_POOL_Pop(&_mem_pool);
    MEM_POOL_Delete(&_mem_pool);
  }

  // The interface for building the information
  void CFG_setup_coderep_affect_info (void);

  // Traverse all the affected coderep and apply the func
  void Walk_all_affected_coderep (CODEREP *cr, 
                                  BOOL mu_included,
                                  CODEREP_DEF_USE_WALK *tr,
                                  STACK<CODEREP *> *work_list)
  {
    CODEREP_AFFECT_INFO *info = _CR_info[cr->Coderep_id()];
    if (info == NULL) {
      return;
    }
    info->Walk_all_affected_coderep(tr, mu_included, work_list);
  }

  // If a CODEREP is considered in the MAP, return it, otherwise NULL
  CODEREP *CR_get_coderep(INT32 cr_id)
  {
    CODEREP_AFFECT_INFO *info = _CR_info[cr_id];
    if (info == NULL) {
      return NULL;
    } else {
      return info->Get_coderep();
    }
  }

};


// This is a container class for implementing ThreadID variance analysis
// on SSA-based IR. The following sequences of calls should be used to
// invoke the analysis,
//
//           CR_VarianceMap *cr_map = new CR_VarianceMap(program_codemap);
//           cr_map->Setup_coderep_tid_var_info();
//
//  At this point, the CODEREP's variance information can be retrieved, and the
//  the following interface can be used,
//              
//           cr_map->CR_get_tid_var_state(CODEREP *);
//
//           cr_map->STMT_get_tid_var_state(STMTREP *);
//
//    or
//
//           ThreadIdx_VAR_STATE CR_is_thread_variant(CODEREP *);
//           ThreadIdx_VAR_STATE STMT_is_thread_variant(STMTREP *);
//
//
//  When it is done, 
//
//           delete cr_map;
//
class CR_VarianceMap : CODEREP_DEF_USE_WALK
{
private:
    CODEMAP *_codemap;          // the SSA-based IR container
    CR_MemoryMap *_cr_mem_map;  // keep information about CODEREP's memory kind
    MEM_POOL _mem_pool;     

    INT32 _total_CRs;           // total number of CODEREPs in the program unit
    struct _CODEREP_INFO_ {
      ThreadIdx_VAR_STATE tid_var_state : 3;
      BOOL read_only : 1;

    } *_CR_info;   // Keep thread ID variant state 
                   // and read_only info for each CODEREP

    STACK<BB_NODE *> *_BB_with_sync_threads;

private:
    // Initialize one BB_NODE's cond_coderep, and collect initial work list items from
    // atomic intrinsic and user calls
    void
    BB_initialize_control_var_info (BB_NODE *bb,
                                    BB_CONTROL_INFO *control_info,
                                    STACK<CODEREP *> *work_list);

    // Initialize all BB_NODE's cond_coderep, and collect initial work list items from
    // atomic intrinsic and user calls
    //
    void
    CFG_init_control_var_info (BB_CONTROL_INFO *control_info,
                               MEM_POOL *mem_pool,
                               STACK<CODEREP *> *work_list);

    // Add a new state to the existing TID var state, and if changed, return TRUE
    //
    BOOL BB_add_control_tid_var_state (BB_NODE *bb, 
                                       BB_CONTROL_INFO *bb_info,
                                       ThreadIdx_VAR_STATE state)
    {
      ThreadIdx_VAR_STATE tmp = (ThreadIdx_VAR_STATE)(bb_info[bb->Id()].control_tid_var_state | state);
      if (tmp != bb_info[bb->Id()].control_tid_var_state) {
        bb_info[bb->Id()].control_tid_var_state = tmp;
        return TRUE;
      } else {
        return FALSE;
      }
    }

    // Check if a BB_NODE's control TID variant state is changed
    // by re-compute the control TID variant state and compare with the existing one
    BOOL
    BB_control_tid_var_state_changed (BB_NODE *bb,
                                      BB_CONTROL_INFO *control_info);

    // Add a new state to the CODEREP's TID var state, and if changed, return TRUE
    //
    BOOL CR_add_tid_var_state (CODEREP *cr, ThreadIdx_VAR_STATE new_state) 
    {
      ThreadIdx_VAR_STATE tmp = (ThreadIdx_VAR_STATE)(CR_get_tid_var_state(cr) | new_state);
      if (tmp != CR_get_tid_var_state(cr)) {
        CR_set_tid_var_state(cr, tmp);
        return TRUE;
      }
      return FALSE;
    }

    // Find all VAR CODEREP which represent ThreadIdx
    void Init_tid_var_list (STACK<CODEREP *> *work_list);

    void Add_bb_with_sync_threads (BB_NODE *bb);

    void BB_refine_iter_control_dep (BB_NODE *bb, 
                                     BB_CONTROL_INFO *bb_info, 
                                     MEM_POOL *mem_pool);

public:
  CR_VarianceMap (CODEMAP *codemap);

  ~CR_VarianceMap(void);

  inline CODEMAP * Get_codemap() const 
  { 
    return _codemap;      // the SSA-based IR container
  }

  CR_MemoryMap *CR_memory_map() { return _cr_mem_map; }

  // Update the TID var state of useCR based on the TID var state of defCR 
  void Walk_coderep_def_use(CODEREP *defCR, CODEREP *useCR, STACK<CODEREP *> *work_list);

  // interfaces for getting the ThreadID variant state for a CODEREP
  inline ThreadIdx_VAR_STATE CR_get_tid_var_state(CODEREP *cr) const 
  { 
    return CR_get_tid_var_state((INT32)cr->Coderep_id());
  }

  // interface for getting TID variant state for a CODEREP ID
  //
  inline ThreadIdx_VAR_STATE CR_get_tid_var_state(INT32 cr_id) const
  {
    return _CR_info[cr_id].tid_var_state;
  }

  // return the TID variant state for a STMTREP
  //
  ThreadIdx_VAR_STATE STMT_get_tid_var_state (STMTREP *stmt);

  // interface for setting the ThreadID variant state
  inline void CR_set_tid_var_state(CODEREP *cr, ThreadIdx_VAR_STATE new_state) 
  { 
    _CR_info[cr->Coderep_id()].tid_var_state = new_state; 
  }

  // return if the CODEREP is read only in the program unit
  inline BOOL CR_get_read_only (CODEREP *cr) const 
  { 
    return _CR_info[cr->Coderep_id()].read_only; 
  }

  // Set the CODEREP read only in the program unit
  inline void CR_set_read_only (CODEREP *cr) const 
  { 
    _CR_info[cr->Coderep_id()].read_only = TRUE; 
  }

  // The interface for invoking the thread-ID variance analysis
  //
  void Setup_coderep_tid_var_info(void);
};


// interface for retrieving a CODEREP's variant state
extern ThreadIdx_VAR_STATE CR_is_thread_variant (CODEREP *cr);

// interface for retrieving a STMTREP's variant state 
extern ThreadIdx_VAR_STATE STMT_is_thread_variant (STMTREP *stmt);

// interface for check the variance state of a BB_NODE's end condition
extern ThreadIdx_VAR_STATE BB_get_tid_var_state (BB_NODE *bb);

// Interface for retrieving the current CR_VarianceMap
extern CR_VarianceMap *Get_current_CR_var_map(void);

#endif // ifdef TARG_NVISA

#endif // opt_variance_INCLUDED

