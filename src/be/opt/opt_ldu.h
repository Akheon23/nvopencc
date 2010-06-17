/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef opt_ldu_INCLUDED
#define opt_ldu_INCLUDED


#ifdef TARG_NVISA


// For keeping information about region-LDU CRs.
class CR_Region_LDU_INFO {

private:
  // for each block, a list of CODEREPs which are region LDU candidate
  // the information cannot be saved for each CR, since one CR might appear
  // in multiple BB_NODEs, therefore it might appear in multiple LOOPs
  STACK<CODEREP *> **bb_region_ldu_list;

  // list of preheaders for loops with region LDU candidate
  // the Inline ASM statement for invalidating uniform cache
  // will be emitted at the beginning of the loop preheader
  STACK<BB_NODE *> *region_preheaders_list;

  MEM_POOL _mem_pool;

public:
  CR_Region_LDU_INFO (CODEMAP *codemap);

  ~CR_Region_LDU_INFO();

  BOOL BB_is_region_ldu_preheader (BB_NODE *bb);

  BOOL CR_is_region_ldu (BB_NODE *bb, CODEREP *cr);
};

// interface for checking if an IVAR/ILOAD CODEREP is uniform - ThreadID variant and read only
extern BOOL CR_is_uniform (CODEREP *);

// Interface for checking is a BB_NODE is a region LDU preheader
extern BOOL BB_is_region_ldu_preheader (BB_NODE *bb);

// Set the current emitting BB_NODE, for checking if a CR is a region LDU
extern void Set_current_emitting_bb_node (BB_NODE *bb);

#endif // ifdef TARG_NVISA

#endif // opt_ldu_INCLUDED

