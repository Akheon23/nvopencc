/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifdef TARG_NVISA

// ====================================================================
//
// Module: opt_ldu.cxx
//
//    This module is for implementing LDU Analysis on SSA-based IR, and
//    provide interfaces to EMITTER to generate LDU WHIRL code.
// ====================================================================
//
#include "cxx_memory.h"
#include "opt_base.h"
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

static CR_Region_LDU_INFO *_current_region_ldu_info = NULL;
static BB_NODE *_current_emitting_bb_node = NULL;

// Interface for setting the current BB_NODE during emitting WHIRL
// Since SSA-based IR is a dag, a CR node could be used in multiple 
// statements and blocks. In order to correctly locate the correct
// region LDU candidate, the current emitting BB_NODE is needed.
//
void
Set_current_emitting_bb_node (BB_NODE *bb)
{
  _current_emitting_bb_node = bb;
}

// Try to find LDA in the current Ilod_base tree 
static CODEREP *
Find_lda_in_Ilod_base(CODEREP *cr)
{
  CODEREP *base;

  if (cr->Kind() == CK_LDA)
    return cr;

  if (cr->Kind() == CK_OP) {
    switch (cr->Opr()) {
    case OPR_ADD:
      base = Find_lda_in_Ilod_base(cr->Opnd(0));
      if (base == NULL)
        base = Find_lda_in_Ilod_base(cr->Opnd(1));
      return base;
    case OPR_SUB:
      base = Find_lda_in_Ilod_base(cr->Opnd(0));
      return base;
    default:
      return NULL;
    }
  }

  return NULL;
}

BOOL
BB_is_region_ldu_preheader (BB_NODE *bb)
{
  if (_current_region_ldu_info == NULL) {
    return FALSE;
  }

  return _current_region_ldu_info->BB_is_region_ldu_preheader(bb);
}

// Check if an ILOAD CODEREP is uniform in the program unit
static BOOL
CR_is_PU_uniform (CODEREP *cr)
{
  if (Get_current_CR_var_map()->CR_get_tid_var_state(cr->Ilod_base()) != ThreadIdx_Invariant) {
    return FALSE;
  }

  // check __constant__ qualifier
  if (WOPT_Enable_CR_Memory_Space_Check == FALSE && WOPT_Enable_Cons2glb > 0) {
    CODEREP *base = Find_lda_in_Ilod_base(cr->Ilod_base());
    if (base) {
      ST* st = base->Lda_base_st();
      if (st && ST_in_constant_mem(st)) {
        return TRUE;
      }
    }
  }

  if (WOPT_Enable_CR_Memory_Space_Check) {
    MEMORY_KIND kind = Get_current_CR_var_map()->CR_memory_map()->CR_get_memory_kind(cr);
    if (kind == MEMORY_KIND_CONST || kind == MEMORY_KIND_GLB_CONST) {
      if (DEBUG_Ldu_Analysis) {
        printf ("Found uniform const-ptr CR: ");
        cr->Print_src(1, stdout);
        printf("\n");
        fflush(stdout);
      }
      return TRUE;
    }

    if (kind == MEMORY_KIND_UNKNOWN || kind == MEMORY_KIND_INIT) {
      if (DEBUG_Ldu_Analysis) {
        printf ("Found uniform ");
        if (kind == MEMORY_KIND_UNKNOWN) {
          printf (":unknown ");
        } else {
          printf (":unchecked ");
        }
        cr->Print_src(1, stdout);
        printf(" : MTYPE %d", cr->Dtyp());
        printf("\n");
        fflush(stdout);
      }
      return FALSE;
    }

    if (kind & (MEMORY_KIND_SHARED | MEMORY_KIND_LOCAL)) {
       if (DEBUG_Ldu_Analysis) {
        printf ("Found uniform ");
        if (kind & MEMORY_KIND_SHARED) {
          printf ("shared-ptr ");
        }
        if (kind & MEMORY_KIND_LOCAL) {
          printf ("local-ptr ");
        }
        cr->Print_src(1, stdout);
        printf("\n");
        fflush(stdout);
      }
      return FALSE;
    }
  }


  // check const qualifier
  if (WOPT_Enable_LDU_Const_Load) {
    if (TY_is_const(cr->Ilod_ty())) {
      if (DEBUG_Ldu_Analysis) {
        printf ("Found uniform const CR: (ILoad %d, Base %d)\n", cr->Coderep_id(), cr->Ilod_base()->Coderep_id());
        fflush(stdout);
      }
      return TRUE;
    }
  }

  if (WOPT_Enable_LDU_Entry_Reached_Only_Load) {
    CODEREP *mu_opnd = cr->Ivar_mu_node()->OPND();
    if (Get_current_CR_var_map()->CR_get_read_only(mu_opnd)) {
      if (DEBUG_Ldu_Analysis) {
        printf ("Found uniform entry reached only CR: (ILoad %d, Base %d)\n", cr->Coderep_id(), cr->Ilod_base()->Coderep_id());
        fflush(stdout);
      }
      return TRUE;
    }
  }

  return FALSE;
}

// Check if an ILOAD CODEREP is uniform either in a region or in a PU
BOOL
CR_is_uniform (CODEREP *cr)
{
  Is_True(cr->Opr() == OPR_ILOAD, ("CR_is_uniform: OPR_ILOAD only"));

  if (Get_current_CR_var_map() == NULL) {
    return FALSE;
  }

  if (WOPT_Enable_Region_LDU_Analysis && 
      _current_region_ldu_info != NULL &&
      _current_region_ldu_info->CR_is_region_ldu (_current_emitting_bb_node, cr)) {
    return TRUE;
  }

  return CR_is_PU_uniform (cr);
}


// Check if an ILOAD CODEREP is uniform
static BOOL
CR_is_loop_ldu_candidate (CODEREP *cr, BB_LOOP *loop)
{
  if (Get_current_CR_var_map()->CR_get_tid_var_state(cr->Ilod_base()) != ThreadIdx_Invariant) {
    return FALSE;
  }

  if (WOPT_Enable_CR_Memory_Space_Check == FALSE) {
    // memory space information is needed in order to perform loop level LDU analysis
    return FALSE;
  }

  MEMORY_KIND kind = Get_current_CR_var_map()->CR_memory_map()->CR_get_memory_kind(cr);

  if (kind == MEMORY_KIND_UNKNOWN || kind == MEMORY_KIND_INIT) {
    return FALSE;
  }

  if (kind & (MEMORY_KIND_SHARED | MEMORY_KIND_LOCAL)) {
    return FALSE;
  }

  CODEREP *mu_opnd = cr->Ivar_mu_node()->OPND();
  if (mu_opnd && mu_opnd->Is_flag_set(CF_IS_ZERO_VERSION)) {
    return FALSE;
  }

  BB_NODE *def_bb = mu_opnd->Defbb();
  if (def_bb == NULL) {
    return FALSE;
  }

  if (def_bb->In_loop(loop) == FALSE) {
    return TRUE;
  }

  return FALSE;
}


// Check if the CR is a loop-uniform candidate.
static void
CR_check_loop_uniform_load (CODEREP *cr, INT32 last_line,
                            BB_LOOP *loop, 
                            STACK<CODEREP *> *rc_ldu_list, 
                            STACK<CODEREP *> *pu_ldu_list)
{
  if (cr->Kind() == CK_IVAR) {
    if (cr->Opr() == OPR_ILOAD) {
      if (CR_is_PU_uniform (cr)) {
        // this is procedure-level LDU candiadte
        pu_ldu_list->Push(cr);
      } else if (CR_is_loop_ldu_candidate (cr, loop)) {
        if (DEBUG_Ldu_Analysis) {
          printf("At line %d, LOOP LDU candidate - ", last_line);
          cr->Print_src(1, stdout);
          printf("\n");
          fflush(stdout);
	}
        rc_ldu_list->Push(cr);
      }
      if (cr->Ilod_base()) {
        CR_check_loop_uniform_load (cr->Ilod_base(), last_line, loop, 
                                    rc_ldu_list, pu_ldu_list);
      } else if (cr->Istr_base()) {
        CR_check_loop_uniform_load (cr->Istr_base(), last_line, loop, 
                                    rc_ldu_list, pu_ldu_list);
      }
    } else if (cr->Opr() == OPR_MLOAD) {
      CR_check_loop_uniform_load (cr->Mload_size(), last_line, loop,
                                  rc_ldu_list, pu_ldu_list);
    } else if (cr->Opr() == OPR_ILOADX) {
      CR_check_loop_uniform_load (cr->Index(), last_line, loop,
                                  rc_ldu_list, pu_ldu_list);
    } 
  } else if (cr->Kind() == CK_OP) {
    for (int i = 0; i < cr->Kid_count(); i++) {
      CODEREP *opnd = cr->Opnd(i);
      CR_check_loop_uniform_load (opnd, last_line, loop, rc_ldu_list, pu_ldu_list);
    }
  }
}
  
// Check if there are loop-level uniform loads candidates in a loop
static void
Check_loop_uniform_load (BB_LOOP *loop, MEM_POOL *mem_pool,
			 STACK<BB_NODE *> *region_preheaders_list,
			 STACK<CODEREP *> *bb_region_ldu_list[])
{
  if (loop->Preheader() == NULL) {
    // Need to insert invalidation instruction in prehader, give up
    return;
  }

  BB_NODE_SET_ITER *biter = new BB_NODE_SET_ITER();
  biter->Init(loop->True_body_set());
  STACK<CODEREP *> *rc_ldu_list = new STACK<CODEREP *>(mem_pool);
  STACK<CODEREP *> *pu_ldu_list = new STACK<CODEREP *>(mem_pool);

  INT32 last_line = 0;
  INT32 rc_ldu_count = 0;

  for (BB_NODE *bb = biter->First_elem(); bb != NULL; bb = biter->Next_elem()) {
    STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
    const STMTREP *stmt;

    FOR_ALL_NODE(stmt, stmt_iter, Init()) {
      OPERATOR stmt_opr = stmt->Opr();

      if (stmt_opr == OPR_PRAGMA) {
        continue;
      }

      INT32 cur_line = Srcpos_To_Line(stmt->Linenum());
      if (cur_line == 0) {
        cur_line = last_line;
      } else {
        last_line = cur_line;
      }

      if (stmt->Rhs() != NULL) {
        CR_check_loop_uniform_load (stmt->Rhs(), cur_line, loop, rc_ldu_list, pu_ldu_list);
      }
    } // FOR_ALL_NODE(stmt, stmt_iter, Init())

    if (rc_ldu_list->Elements() > 0) {
      bb_region_ldu_list[bb->Id()] = rc_ldu_list;
      rc_ldu_count += rc_ldu_list->Elements();
      rc_ldu_list = new STACK<CODEREP *>(mem_pool);
    }
  }

#define MIN_REGION_LDU 0

  if (rc_ldu_count > MIN_REGION_LDU) {
    // simple heuristics here, as long as number of region LDU candidates
    // is higher than the minimum threashold, those region LDU candidates
    // will be marked as LDU candiadtes, and an uniform cache invalidation
    // instruction will be inserted at the beginning of the loop.
    region_preheaders_list->Push (loop->Preheader());

    if (DEBUG_Ldu_Analysis) {
      printf ("Found %d loop-level LDU candidates,  %d proc-level LDU candidates\n", 
              rc_ldu_count, pu_ldu_list->Elements());
    }
  } else if (rc_ldu_count > 0) {
    // the loop is not a region LDU candidate, so reset the region LDU coderep list
    for (BB_NODE *bb = biter->First_elem(); bb != NULL; bb = biter->Next_elem()) {
      if (bb_region_ldu_list[bb->Id()] != NULL) {
        delete bb_region_ldu_list[bb->Id()];
        bb_region_ldu_list[bb->Id()] = NULL;
      }
    }
  }

  // FIXME, pu_ldu_list is currently not used in the heuristics yet
  // the information is collected for purposes of debugging
  delete pu_ldu_list;

  // only outmost loop level is checked, since the cost of invalidating uniform cache
  // inside a loop may cancel all the LDU benefits
}

CR_Region_LDU_INFO::CR_Region_LDU_INFO (CODEMAP *codemap)
{
  bb_region_ldu_list = NULL;
  region_preheaders_list = NULL;

  MEM_POOL_Initialize(&_mem_pool, "CR_Region_LDU_INFO Pool", FALSE);
  MEM_POOL_Push(&_mem_pool);

  // for kernel routine, do region LDU analysis
  if (Current_PU_is_Global() == FALSE) {
    return;
  }

  if (codemap->Cfg()->Loops() == NULL) {
    return;
  }

  INT32 total_BBs = codemap->Cfg()->Total_bb_count() + 1;

  bb_region_ldu_list = (STACK<CODEREP *> **)MEM_POOL_Alloc (&_mem_pool, 
                total_BBs * sizeof(STACK<CODEREP *> *));

  memset (bb_region_ldu_list, 0, total_BBs * sizeof(STACK<CODEREP *> *));

  region_preheaders_list = new STACK<BB_NODE *>(&_mem_pool);

  BB_LOOP_ITER loop_iter(codemap->Cfg()->Loops());
  BB_LOOP *child;
  FOR_ALL_NODE(child, loop_iter, Init()) {
    Check_loop_uniform_load (child, &_mem_pool, region_preheaders_list, bb_region_ldu_list);
  }

  _current_region_ldu_info = this;
}

CR_Region_LDU_INFO::~CR_Region_LDU_INFO() 
{
  MEM_POOL_Pop(&_mem_pool);
  MEM_POOL_Delete(&_mem_pool);
  _current_region_ldu_info = NULL;
}

BOOL 
CR_Region_LDU_INFO::BB_is_region_ldu_preheader (BB_NODE *bb)
{
  for (int i = 0; i < region_preheaders_list->Elements(); i++) {
    if (region_preheaders_list->Top_nth(i) == bb) {
      return TRUE;
    }
  }

  return FALSE;
}

BOOL
CR_Region_LDU_INFO::CR_is_region_ldu (BB_NODE *bb, CODEREP *cr)
{
  if (bb_region_ldu_list[bb->Id()] == NULL) {
    return FALSE;
  }

  for (int i = 0; i < bb_region_ldu_list[bb->Id()]->Elements(); i++) {
    if (bb_region_ldu_list[bb->Id()]->Top_nth(i) == cr) {
      return TRUE;
    }
  }
  return FALSE;
}



#endif // ifdef TARG_NVISA

