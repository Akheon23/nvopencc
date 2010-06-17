/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#include "optimizer.h"
#include "opt_alias_class.h"
#include "opt_cfg.h"
#include "opt_main.h"
#include "opt_dump_ir.h"
#include "opt_sym.h"
#include "opt_htable.h"
#include "config_wopt.h"

// This file collect some interfaces for dumping IRs inside Pre_Optimizer.


static COMP_UNIT *current_comp_unit;    // This is set when COMP_UNIT is created in Pre_Optimizer, 
                                        // and reset when the COMP_UNIT is deleted.

// Dump routines used in released version for tracking both correctness and performance problems

// Interface for set the current COMP_UNIT
void
Set_current_comp_unit (COMP_UNIT *unit)
{
  current_comp_unit = unit;
}

// Interface for retrieving the current COMP_UNIT
COMP_UNIT *
Get_current_comp_unit (void)
{
  return current_comp_unit;
}


// find the block number for a Label number after CFG is created.
int
get_block_number_on_label(int label_no)
{
  if (Get_current_comp_unit() == NULL || 
    Get_current_comp_unit()->Cfg() == NULL) {
    return 0;
  }
  CFG_ITER *c_iter = new CFG_ITER(Get_current_comp_unit()->Cfg());
  for (BB_NODE *bb = c_iter->First(); bb; bb = c_iter->Next()) {
    if (bb->Labnam() == label_no) {
      return bb->Id();
    }
  }
  //should not be here
  return 0;
}

void
print_aux_detail (FILE *f, INT32 aux_id, INT32 ver)
{
  // traverse the aux_stab first
  fprintf(f, "aux_%d", aux_id);
  if (ver != 0) {
    fprintf(f, "_v%d", ver);
  }

  AUX_STAB_ENTRY *aux = Get_current_comp_unit()->Opt_stab()->Aux_stab_entry(aux_id);
  if (aux->St()) {
    fprintf (f, "_%s", ST_name(aux->St()));
  }
  if (aux->St_ofst() != 0) {
    fprintf (f, "_%lld", aux->St_ofst());
  }

  if (TY_is_const(aux->Ty())) {
    fprintf(f, "_C");
  }
  if (TY_is_volatile(aux->Ty())) {
    fprintf(f, "_V");
  }
  if (TY_is_restrict(aux->Ty())) {
    fprintf (f, "_R");
  }
}

// Default routine for printing aux table entry information in detail
void
print_wn_aux_extra (FILE *f, WN *wn)
{
  ST_IDX st_idx = WN_st_idx(wn);
  INT32 aux_id;
  INT32 ver;

  if ((Wn_flags(wn) & WN_FLAG_ST_TYPE) == WN_ST_IS_VER) {
    VER_STAB_ENTRY *vse = Get_current_comp_unit()->Opt_stab()->Ver_stab_entry(st_idx);
    aux_id = vse->Aux_id();
    ver = vse->Version();
  } else if ((Wn_flags(wn) & WN_FLAG_ST_TYPE) == WN_ST_IS_AUX) {
    aux_id = st_idx;
    ver = 0;
  } else {
    fprintf (f, "%s", ST_name(st_idx));
    return;
  }

  print_aux_detail (f, aux_id, ver);
}

// Default routine for printing CODEREP information in detail
void
print_cr_extra (FILE *f, const CODEREP *cr)
{
  INT32 aux_id = cr->Aux_id();
  INT32 ver = cr->Version();

  // traverse the aux_stab first
  fprintf(f, "cr%d_aux_%d", cr->Coderep_id(), aux_id);
  if (ver != 0) {
    fprintf(f, "_v%d", ver);
  }

  AUX_STAB_ENTRY *aux = Get_current_comp_unit()->Opt_stab()->Aux_stab_entry(aux_id);
  if (aux->St()) {
    fprintf (f, "_%s", ST_name(aux->St()));
  }
  if (aux->St_ofst() != 0) {
    fprintf (f, "_%lld", aux->St_ofst());
  }

  if (TY_is_const(aux->Ty())) {
    fprintf(f, "_C");
  }
  if (TY_is_volatile(aux->Ty())) {
    fprintf(f, "_V");
  }
  if (TY_is_restrict(aux->Ty())) {
    fprintf (f, "_R");
  }
}

// Set default aux print routine, so that more details can be dumped
void
Set_ssa_print_rout (void)
{
  extern void set_print_ssa_var_info (void (*new_print_ssa_var_info)(FILE *, WN *));
  set_print_ssa_var_info (&print_wn_aux_extra);
}

// Reset default aux print routine, no more details
void
Reset_ssa_print_rout (void)
{
  extern void set_print_ssa_var_info (void (*new_print_ssa_var_info)(FILE *, WN *));
  set_print_ssa_var_info (NULL);
}

// Set the default CR Variable print routine for dumping more details.
void
Set_cr_print_rout (void)
{
  extern void set_print_cr_var_info (void (*new_print_cr_var_info)(FILE *, const CODEREP *));
  set_print_cr_var_info (&print_cr_extra);
}

// Reset the default CR Variable print routine, no more details
void
Reset_cr_print_rout (void)
{
  extern void set_print_cr_var_info (void (*new_print_cr_var_info)(FILE *, const CODEREP *));
  set_print_cr_var_info (NULL);
}

/* Dump control flow graph when AUX_TABLE is available */
void
print_cfg_aux (void)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_ssa_print_rout();
  }

  CFG_ITER *c_iter = new CFG_ITER(Get_current_comp_unit()->Cfg());
  for (BB_NODE *bb = c_iter->First(); bb; bb = c_iter->Next()) {
    printf ("\nBLOCK %d\n", bb->Id());
    bb->Print_aux(stdout);
    fflush(stdout);
  }

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Reset_ssa_print_rout();
  }
}

/* Dump control flow graph when CODEREP is available */
void
print_cfg_cr (void)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_cr_print_rout();
  }

  CFG_ITER *c_iter = new CFG_ITER(Get_current_comp_unit()->Cfg());
  for (BB_NODE *bb = c_iter->First(); bb; bb = c_iter->Next()) {
    printf ("\nBLOCK %d\n", bb->Id());
    bb->Print_cr(stdout);
    fflush(stdout);
  }

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Reset_cr_print_rout();
  }
}

/* Dump control flow based on WHIRL node. */
void
print_cfg_wn (void)
{
  CFG_ITER *c_iter = new CFG_ITER(Get_current_comp_unit()->Cfg());
  for (BB_NODE *bb = c_iter->First(); bb; bb = c_iter->Next()) {
    printf ("\nBLOCK %d\n", bb->Id());
    bb->Print_src();
    fflush(stdout);
  }
}

/* Dump a BB_NODE for a given block number where AUX_TABLE is created. */
void
bnpr_aux (int bno)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_ssa_print_rout();
  }

  BB_NODE *bb = Get_current_comp_unit()->Cfg()->Get_bb(bno);
  printf ("\nBLOCK %d\n", bno);
  bb->Print_aux(stdout);
  fflush(stdout);

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Reset_ssa_print_rout();
  }
}

/* Dump a BB_NODE for a given block number where CODEREP is created. */
void
bnpr_cr (int bno)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_ssa_print_rout();
  }

  BB_NODE *bb = Get_current_comp_unit()->Cfg()->Get_bb(bno);
  printf ("\nBLOCK %d\n", bno);
  bb->Print_cr(stdout);
  fflush(stdout);

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Reset_ssa_print_rout();
  }
}

/* Dump a BB_NODE's original WN for a given block number in C-style */
void
bnpr_wn_src (int bno)
{
  BB_NODE *bb = Get_current_comp_unit()->Cfg()->Get_bb(bno);
  printf ("\nBLOCK %d\n", bno);
  bb->Print_src();
  fflush(stdout);
}

/* Dump a BB_NODE's original WN for a given block number in tree-format */
void
bnpr_wn (int bno)
{
  BB_NODE *bb = Get_current_comp_unit()->Cfg()->Get_bb(bno);
  printf ("\nBLOCK %d\n", bno);
  bb->Print(stdout);
  fflush(stdout);
}

/* Dump SSA infoormation for a block given the block number */
void
bnpr_ssa (int bno)
{
  BB_NODE *bb = Get_current_comp_unit()->Cfg()->Get_bb(bno);
  printf ("\nBLOCK %d\n", bno);
  bb->Print_ssa();
  fflush(stdout);
}

/* Dump a block's WN */
void
print_bb (BB_NODE *bb)
{
  bb->Print();
}

/* Dump a block's SSA information */
void
bpr_ssa (BB_NODE *bb)
{
  bb->Print_ssa();
}


/* Dump a BB_NODE for a given block pointer when AUX_TABLE is available */
void 
bpr_aux (BB_NODE *bb)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_ssa_print_rout();
  }

  printf ("\nBLOCK %d\n", bb->Id());
  bb->Print_aux(stdout);
  fflush(stdout);

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Reset_ssa_print_rout();
  }
}  

/* Dump a BB_NODE for a given block pointer when CODEREP is available. */
void 
bpr_cr (BB_NODE *bb)
{
  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_cr_print_rout();
  }

  printf ("\nBLOCK %d\n", bb->Id());
  bb->Print_cr(stdout);
  fflush(stdout);

  if (WOPT_Enable_Dump_SSA_VAR_Detail) {
    Set_cr_print_rout();
  }
}  

void
print_block_cr (int bno)
{
  bnpr_cr(bno);
}

void
print_block_aux (int bno)
{
  bnpr_aux(bno);
}

void
print_block_wn (int bno)
{
  bnpr_wn(bno);
}

void
print_block_wn_src (int bno)
{
  bnpr_wn_src(bno);
}

// Print a block's SSA control info
void
print_block_ssa (BB_NODE *bb)
{
  bpr_ssa(bb);
}

// Print pre-SSA-based IR inside a block
void
print_block_aux (BB_NODE *bb)
{
  bpr_aux(bb);
}

// Print SSA-based IR inside a block
void
print_block_cr (BB_NODE *bb)
{
    bpr_cr(bb);
}

// Print all CODEREPs which represent threadIdxs
void
print_thread_crs(void)
{
  CODEREP_ITER cr_iter;
  CODEREP *cr;
  CODEMAP *htable = Get_current_comp_unit()->Htable();

  // traverse the aux_stab first, for VAR cr
  AUX_ID i;
  AUX_STAB_ITER aux_stab_iter(Get_current_comp_unit()->Opt_stab());
  FOR_ALL_NODE(i, aux_stab_iter, Init()) {
    AUX_STAB_ENTRY *aux = Get_current_comp_unit()->Opt_stab()->Aux_stab_entry(i);
    if (strcmp(aux->Base_name(),"threadIdx") == 0) {
      mINT64 offset = aux->St_ofst();
      
      FOR_ALL_NODE(cr, cr_iter, Init(aux->Cr_list())) {
        if (offset == 0) {
         printf ("Thread x: ");
        } else if (offset == 4) {
          printf ("Thread y: ");
        } else if (offset == 8) {
          printf ("Thread z: ");
        }
        htable->Print_CR(cr, stdout);
        fflush(stdout);
      } // FOR_ALL_NODE
    } // if 
  } // FOR_ALL_NODE(i
}

// Print a CR with a given CR number, for getting
// details of a CR
//
void
print_aux (int aux_id)
{
  print_aux_detail (stdout, aux_id, 0);
  printf("\n");
  fflush(stdout);
}

// Print a CR with a given CR number, for getting
// details of a CR
//
void
print_cr (int cr_id)
{
  CODEMAP_ITER codemap_iter;
  CODEREP *bucket, *cr;
  CODEMAP *htable = Get_current_comp_unit()->Htable();
  CODEREP_ITER cr_iter;

  // traverse the aux_stab first, for VAR cr
  AUX_ID i;
  AUX_STAB_ITER aux_stab_iter(Get_current_comp_unit()->Opt_stab());
  FOR_ALL_NODE(i, aux_stab_iter, Init()) {
    AUX_STAB_ENTRY *aux = Get_current_comp_unit()->Opt_stab()->Aux_stab_entry(i);
    FOR_ALL_NODE(cr, cr_iter, Init(aux->Cr_list())) {
      if (cr->Coderep_id() == cr_id) {
        htable->Print_CR(cr, stdout);
	if (TY_is_const(cr->Lod_ty())) {
          printf(" : const");
	}
	if (TY_is_volatile(cr->Lod_ty())) {
	  printf(" : volatile");
	}
	if (TY_is_restrict(cr->Lod_ty())) {
	  printf (" : restrict");
	}
        printf("\n");
        fflush(stdout);
        return;
      }
    }
  }

  // try to find other cr
  FOR_ALL_ELEM(bucket, codemap_iter, Init(htable)) {
    FOR_ALL_NODE(cr, cr_iter, Init(bucket)) {
      while (cr != NULL) {
        if (cr->Coderep_id() == cr_id) {
          htable->Print_CR(cr, stdout);
	  if (cr->Kind() == CK_IVAR) {
            if (TY_is_const(cr->Ilod_ty())) {
              printf(" : const");
	    }
	    if (TY_is_volatile(cr->Ilod_ty())) {
	      printf(" : volatile");
	    }
	    if (TY_is_restrict(cr->Ilod_ty())) {
	      printf (" : restrict");
	    }
            if (TY_is_const(cr->Ilod_base_ty())) {
              printf(" : p-const");
	    }
	    if (TY_is_volatile(cr->Ilod_base_ty())) {
	      printf(" : p-volatile");
	    }
	    if (TY_is_restrict(cr->Ilod_base_ty())) {
	      printf (" : p-restrict");
	    }
	  }
          printf("\n");
          fflush(stdout);
          return;
        }
        cr = (CODEREP *)cr->Next();
      }
    }
  }


  printf ("CR %d does not exist\n", cr_id);
  fflush(stdout);
}


// Print a CR with a given CR number, for getting
// details of a CR
//
void
print_cr_src (int cr_id)
{
  CODEMAP_ITER codemap_iter;
  CODEREP *bucket, *cr;
  CODEMAP *htable = Get_current_comp_unit()->Htable();
  CODEREP_ITER cr_iter;

  // traverse the aux_stab first, for VAR cr
  AUX_ID i;
  AUX_STAB_ITER aux_stab_iter(Get_current_comp_unit()->Opt_stab());
  FOR_ALL_NODE(i, aux_stab_iter, Init()) {
    AUX_STAB_ENTRY *aux = Get_current_comp_unit()->Opt_stab()->Aux_stab_entry(i);
    FOR_ALL_NODE(cr, cr_iter, Init(aux->Cr_list())) {
      if (cr->Coderep_id() == cr_id) {
        cr->Print_src(1, stdout);
	printf("\n");
        fflush(stdout);
        return;
      }
    }
  }

  // try to find other cr
  FOR_ALL_ELEM(bucket, codemap_iter, Init(htable)) {
    FOR_ALL_NODE(cr, cr_iter, Init(bucket)) {
      while (cr != NULL) {
        if (cr->Coderep_id() == cr_id) {
          cr->Print_src(1, stdout);
	  printf("\n");
          fflush(stdout);
          return;
        }
        cr = (CODEREP *)cr->Next();
      }
    }
  }


  printf ("CR %d does not exist\n", cr_id);
  fflush(stdout);
}

