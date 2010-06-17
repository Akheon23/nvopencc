/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef opt_memory_space_INCLUDED
#define opt_memory_space_INCLUDED

#ifdef TARG_NVISA

#include "opt_variance.h"

// The container class for setup and retrieving memory kind of
// CODEREPs in the current program unit.
//
// The interfaces for using the class are as follows,
//
//          CR_MemoryMap *cr_mem_map = new CR_MemoryMap(codemap);
//          cr_mem_map->Setup_coderep_memory_kind (p_cr_affect_map);
//
//   At this point, the cr_mem_map can be used to retrieve the memory kind for
//     each CODEREP in the program unit, the interface is,
//
//          cr_mem_map->CR_get_memory_kind (CODEREP *);
//
//   Upon finishing, 
//
//          delete cr_mem_map;
//
class CR_MemoryMap : CODEREP_DEF_USE_WALK
{
private:
    CODEMAP *_codemap;       // Container of SSA-based IR
    MEM_POOL _mem_pool;     // memory pool for memory kind array 

    INT32 _total_CRs;       // total number of CODEREPs in the program unit

    struct cr_mem_info {
      MEMORY_KIND mem_kind : 6;  // Keep MEMORY kind information  
                                 // for each CODEREP
      BOOL is_pointer:1;         // keep tracking if a CODEREP has pointer type

      MEMORY_KIND pointee_mem_kind : 6; // if the CR is a pointer, the memory space of the pointee


    } *_CR_mem_info;

private:

  // Add a memory kind to the cr's existing memory kind
  // return if the cr's memory kind is changed, 
  BOOL Add_memory_kind (CODEREP *cr, MEMORY_KIND new_kind)
  {
    MEMORY_KIND tmp_kind = (MEMORY_KIND)(CR_get_memory_kind (cr) | new_kind);
    if (tmp_kind != CR_get_memory_kind (cr)) {
      CR_set_memory_kind (cr, tmp_kind);
      return TRUE;
    }

    return FALSE;
  }

  // Add a memory kind to the cr's existing memory kind
  // return if the cr's memory kind is changed, 
  BOOL Add_pointee_memory_kind (CODEREP *cr, MEMORY_KIND new_kind)
  {
    MEMORY_KIND tmp_kind = (MEMORY_KIND)(CR_get_pointee_memory_kind (cr) | new_kind);
    if (tmp_kind != CR_get_pointee_memory_kind (cr)) {
      CR_set_pointee_memory_kind (cr, tmp_kind);
      return TRUE;
    }

    return FALSE;
  }

  // interfaces for setting the memory kind of a given CODEREP
  inline void CR_set_memory_kind (CODEREP *cr, MEMORY_KIND kind) 
  { 
    _CR_mem_info[cr->Coderep_id()].mem_kind = kind; 
  }

  // interfaces for setting the memory kind of a given CODEREP
  inline void CR_set_pointee_memory_kind (CODEREP *cr, MEMORY_KIND kind) 
  { 
    _CR_mem_info[cr->Coderep_id()].pointee_mem_kind = kind; 
  }

  BOOL CR_get_is_pointer(CODEREP *cr) 
  {
    return _CR_mem_info[cr->Coderep_id()].is_pointer;
  }

  void CR_set_is_pointer(CODEREP *cr, BOOL is_pointer) 
  {
    _CR_mem_info[cr->Coderep_id()].is_pointer = is_pointer;
  }

  // Internal interface for following a CODEREP to initialize memory 
  // kind for each related ones
  void
  CR_init_memory_kind(CODEREP *cr, STACK<CODEREP *> *work_list);

  // Internal interface for tracking memory kind of CODEREPs in a BB_NODE 
  void
  BB_init_memory_kind(BB_NODE *bb, STACK<CODEREP *> *work_list);

  // Internal interface for tracking memory kind of CODEREPs in the current
  // program unit
  void 
  CFG_init_memory_kind(STACK<CODEREP *> *work_list);

  void Dump_conflicts(CODEREP *parent, const STMTREP *stmt);
 
  void BB_dump_conflicts (BB_NODE *bb);
 
  void CR_dump_conflicts (CODEREP *cr, const STMTREP *stmt);

  // Dump the information about iloads with conflict memory space
  void CFG_dump_conflicts ();

public:
  CR_MemoryMap (CODEMAP *codemap);

  ~CR_MemoryMap(void);

  // Update the memory kind following forward data-flow 
  void Walk_coderep_def_use(CODEREP *defCR, CODEREP *useCR, STACK<CODEREP *> *work_list);

  // interfaces for retrieving the memory kind of a given CODEREP
  inline MEMORY_KIND CR_get_memory_kind (CODEREP *cr) const 
  { 
    return CR_get_memory_kind((INT32)cr->Coderep_id());
  }

  // interfaces for retrieving the memory kind of a given CODEREP
  inline MEMORY_KIND CR_get_memory_kind (INT32 cr_id) const
  {
    return _CR_mem_info[cr_id].mem_kind;
  }

    // interfaces for retrieving the memory kind of a given CODEREP
  inline MEMORY_KIND CR_get_pointee_memory_kind (CODEREP *cr) const 
  { 
    return CR_get_pointee_memory_kind((INT32)cr->Coderep_id());
  }

  // interfaces for retrieving the memory kind of a given CODEREP
  inline MEMORY_KIND CR_get_pointee_memory_kind (INT32 cr_id) const
  {
    return _CR_mem_info[cr_id].pointee_mem_kind;
  }

  // The interface for setup memory kind of CODEREPs in the program unit
  //
  void Setup_coderep_memory_kind (CODEREP_AFFECT_MAP *map);
};

extern void
Create_Current_WN_Memory_Space_Map (CODEMAP *codemap, MEM_POOL *pu_pool);

extern void
Delete_Current_WN_Memory_Space_Map(void);

extern void
Delete_Current_CR_Memory_Map (void);

extern MEMORY_KIND 
WN_get_memory_kind (WN *wn);

extern void
WN_set_memory_kind (WN *wn, MEMORY_KIND kind);

extern MEMORY_KIND
CR_get_pointee_memory_kind (CODEREP *cr);

extern MEMORY_KIND
CR_get_memory_kind (CODEREP *cr);

extern MEMORY_KIND 
ST_get_memory_kind (ST *st);

#endif // ifdef TARG_NVISA

#endif // opt_memory_space_INCLUDED

