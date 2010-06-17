/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifdef TARG_NVISA

#include "cxx_memory.h"
#include "opt_wn.h"
#include "opt_cfg.h"
#include "opt_sym.h"
#include "opt_htable.h"
#include "opt_ssa.h"
#include "opt_mu_chi.h"
#include "bb_node_set.h"
#include "opt_bb.h"

#include "config_wopt.h"
#include "opt_htable.h"
#include "opt_memory_space.h"
#include "opt_dump_ir.h"

// This is the module for implementing memory space analysis on SSA-based IR.
// Basic algorithm is,
//
//    - Initialize all CODEREPs' memory kind to MEMORY_KIND_INIT (unchecked)
//
//    - Collect an initial work-list from LDA CODEREP, and entry CHI_LIST
//
//    - while the work-list is not EMPTY
//          POP a def-CR from the work-list
//          Following the data-flow edges, and for each useCR,
//            propgate the def-CR's memory state to useCR, and useCR's memory state is changed,
//               PUSH useCR to work-list.
//  
// 

#define WOPT_Enable_Memory_Space_Check_Dump (WOPT_Enable_CR_Memory_Space_Check & 2)
#define WOPT_Enable_Memory_Space_Check_Debug (WOPT_Enable_CR_Memory_Space_Check & 4)

#define MEMORY_KIND_INIT_NAME ":Unchecked"
#define MEMORY_KIND_LOCAL_NAME ":Local"
#define MEMORY_KIND_CONST_NAME ":Const"
#define MEMORY_KIND_SHARED_NAME ":Shared"
#define MEMORY_KIND_GLOBAL_NAME ":Global"
#define MEMORY_KIND_PARAM_NAME ":Param"
#define MEMORY_KIND_GLB_CONST_NAME ":Global from const"
#define MEMORY_KIND_UNKNOWN_NAME ":Unknown"

// Interface for generating a name to represent the memory kind
// for debugging
char *Gen_memory_kind_name (MEMORY_KIND kind, char name[])
{
  if (kind == MEMORY_KIND_INIT) {
    strcpy(name, MEMORY_KIND_INIT_NAME);
  } else if (kind == MEMORY_KIND_UNKNOWN) {
    strcpy(name, MEMORY_KIND_UNKNOWN_NAME);
  } else {
    int count = 0;
    if (kind & MEMORY_KIND_LOCAL) {
      strcpy(&name[count], MEMORY_KIND_LOCAL_NAME);
      count += strlen(MEMORY_KIND_LOCAL_NAME);
    }
    if (kind & MEMORY_KIND_SHARED) {
      strcpy(&name[count], MEMORY_KIND_SHARED_NAME);
      count += strlen(MEMORY_KIND_SHARED_NAME);
    }
    if (kind & MEMORY_KIND_PARAM) {
      strcpy(&name[count], MEMORY_KIND_PARAM_NAME);
      count += strlen(MEMORY_KIND_PARAM_NAME);
    }
    if (kind & MEMORY_KIND_GLB_CONST) {
      strcpy(&name[count], MEMORY_KIND_GLB_CONST_NAME);
      count += strlen(MEMORY_KIND_GLB_CONST_NAME);
    }
    if (kind & MEMORY_KIND_CONST) {
      strcpy(&name[count], MEMORY_KIND_CONST_NAME);
      count += strlen(MEMORY_KIND_CONST_NAME);
    }
    if (kind & MEMORY_KIND_GLOBAL) {
      strcpy(&name[count], MEMORY_KIND_GLOBAL_NAME);
      count += strlen(MEMORY_KIND_GLOBAL_NAME);
    }

    name[count] = '\0';
  }

  return name;
}

MEMORY_KIND 
ST_get_memory_kind (ST *st)
{
  switch (ST_memory_space(st)) {

  case MEMORY_GLOBAL:
    return MEMORY_KIND_GLOBAL;

  case MEMORY_LOCAL:
    return MEMORY_KIND_LOCAL;

  case MEMORY_SHARED:
    return MEMORY_KIND_SHARED;

  case MEMORY_CONSTANT:
    return MEMORY_KIND_CONST;

  case MEMORY_PARAM:
    return MEMORY_KIND_PARAM;

  default:
    if (ST_sclass(st) == SCLASS_AUTO || ST_sclass(st) == SCLASS_FORMAL) {
      return MEMORY_KIND_LOCAL;
    }
    return MEMORY_KIND_UNKNOWN;
  }
}

// Dump potential memory space conflicts
void
CR_MemoryMap::Dump_conflicts (CODEREP *parent_cr, const STMTREP *stmt)
{

  MEMORY_KIND kind = CR_get_memory_kind(parent_cr);

  if (kind == MEMORY_KIND_CONST
      || kind == MEMORY_KIND_LOCAL
      || kind == MEMORY_KIND_SHARED
      || kind == MEMORY_KIND_GLOBAL
      || kind == MEMORY_KIND_PARAM
      || kind == MEMORY_KIND_GLB_CONST) {
    return;
  }

  char name[80];

  INT32 last_line = Srcpos_To_Line(stmt->Linenum());
  printf("At LINE %d - ", last_line);
  printf("CR %d accesses conflict memory space %s - ", 
         parent_cr->Coderep_id(), Gen_memory_kind_name(kind, name));
  parent_cr->Print_src(1, stdout);
  printf("\n");
  fflush(stdout);
}

// Check if a CR and its kids have any conflicting memory space
void
CR_MemoryMap::CR_dump_conflicts (CODEREP *cr, const STMTREP *stmt) 
{
  if (cr->Kind() == CK_LDA) {
    return;
  }

  if (cr->Kind() == CK_IVAR) {

    if (cr->Opr() == OPR_ILOAD) {
      if (cr->Ilod_base() != NULL) {
        Dump_conflicts (cr, stmt);
      }

      if (cr->Istr_base() != NULL) {
        Dump_conflicts (cr, stmt);
      }
    }
    return;
  }

  if (cr->Kind() == CK_OP) {

    for (int i=0; i < cr->Kid_count(); i++) {
      CR_dump_conflicts (cr->Opnd(i), stmt);
    }
  }
}

// Traverse statements in a BB_NODE, and find
// CODEREPs with conflicting memory space
void
CR_MemoryMap::BB_dump_conflicts (BB_NODE *bb)
{
  STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
  const STMTREP *stmt;

  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    OPERATOR stmt_opr = stmt->Opr();

    if (stmt_opr == OPR_PRAGMA) {
      continue;
    }

    if (OPERATOR_is_scalar_istore(stmt->Opr())) {
      
      Dump_conflicts (stmt->Lhs(), stmt);
    }

    if (stmt->Rhs() != NULL) {
      CR_dump_conflicts (stmt->Rhs(), stmt);
    }
  } // FOR_ALL_NODE(stmt, stmt_iter, Init())

}

// Go through the program unit, and find
// any ILOADs which have conflicting memory space
void 
CR_MemoryMap::CFG_dump_conflicts (void)
{
  CFG_ITER cfg_iter(_codemap->Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE( bb, cfg_iter, Init() ) {
    BB_dump_conflicts (bb);
  }
}

//#define DEBUG_MEMORY_SPACE_CHECK

// Find all LDA CODEREP in the DAG based on the cr, and setup the 
// LDA's memory kind base on the ST's memory kind in LDA CODEREP.
void
CR_MemoryMap::CR_init_memory_kind(CODEREP *cr, STACK<CODEREP *> *work_list)
{
  if (cr->Kind() == CK_LDA) {
    ST *st = _codemap->Opt_stab()->Aux_stab_entry(cr->Lda_aux_id())->St();
    MEMORY_KIND kind = ST_get_memory_kind (st);
    CR_set_is_pointer (cr, TRUE);
    CR_set_pointee_memory_kind (cr, kind);

    if (WOPT_Enable_Memory_Space_Check_Debug) {
      char names[80];
      printf("lda cr%d (&%s) : pointee-kind %s\n", cr->Coderep_id(), ST_name(st), 
             Gen_memory_kind_name(kind, names));
    }
    work_list->Push(cr);
    return;
  }

  if (cr->Kind() == CK_IVAR) {

    if (cr->Opr() == OPR_ILOAD) {
      if (cr->Ilod_base() != NULL) {
        CR_init_memory_kind (cr->Ilod_base(), work_list);
      } else if (cr->Istr_base() != NULL) {
        CR_init_memory_kind (cr->Istr_base(), work_list);
      }
    }

    return;
  }

  if (cr->Kind() == CK_OP) {

    for (int i=0; i < cr->Kid_count(); i++) {
      CR_init_memory_kind (cr->Opnd(i), work_list);
    }
  }
}

// Check if a structure member is a pointer
BOOL
Is_pointer_struct_member (mINT64 input_ofst, TY_IDX ty, mINT64 off)
{
  FLD_ITER fld_iter;

  // found structure, iterate over fields.
  fld_iter = Make_fld_iter(TY_fld(ty)); 
  do {
    FLD_HANDLE fld(fld_iter);
    if (off + FLD_ofst(fld) > input_ofst) {
      return FALSE;
    }

    if (off + FLD_ofst(fld) == input_ofst &&
      TY_kind(FLD_type(fld)) == KIND_POINTER) {
      return TRUE;
    } else if (TY_kind(FLD_type(fld)) == KIND_STRUCT) {
      if (Is_pointer_struct_member (input_ofst, FLD_type(fld), off + FLD_ofst(fld))) {
        return TRUE;
      }
    }
  } while (!FLD_last_field(fld_iter++));

  return FALSE;
}

// Traverse statements in a BB_NODE, and find
// init work list for tracking memory kind
void
CR_MemoryMap::BB_init_memory_kind (BB_NODE *bb, STACK<CODEREP *> *work_list)
{
  STMTREP_CONST_ITER  stmt_iter(bb->Stmtlist());
  const STMTREP *stmt;

  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    OPERATOR stmt_opr = stmt->Opr();

    if (stmt_opr == OPR_PRAGMA) {
      continue;
    }

    if (stmt->Lhs() != NULL) {
      CR_init_memory_kind (stmt->Lhs(), work_list);
    }

    if (stmt->Rhs() != NULL) {
      CR_init_memory_kind (stmt->Rhs(), work_list);
    }

    if (stmt_opr == OPR_OPT_CHI && stmt->Has_chi()) {
      // All CODEREPs in the Entry CHI list are 
      // considered for the initial worklist
      CHI_NODE *cnode;
      CHI_LIST_ITER chi_iter; 
      FOR_ALL_NODE(cnode, chi_iter, Init(stmt->Chi_list())) {
        if (cnode->Live()) {
          CODEREP *res = cnode->RESULT();
          AUX_ID aid = res->Aux_id();
          AUX_STAB_ENTRY *aux_entry = _codemap->Opt_stab()->Aux_stab_entry(aid);
          ST *st = aux_entry->St();
          if (st != NULL) {
            // only pointer types are considered
            MEMORY_KIND kind = ST_get_memory_kind (st);
            if (kind == MEMORY_KIND_GLOBAL) {
              // the const2global may convert some CUDA constant to globals
              // during processing previous kernel functions
              if (ST_is_const_var(st)) {
                kind = MEMORY_KIND_GLB_CONST;
              }
            }
            CR_set_memory_kind (res, kind);

            if (WOPT_Enable_Memory_Space_Check_Debug) {
	      char names[80];
              printf("init cr%d :kind = %s, st_name %s : ofst %d\n", 
                     res->Coderep_id(), Gen_memory_kind_name(kind, names), 
                     ST_name(st), (int)aux_entry->St_ofst());
	    }

            if (TY_kind(ST_type(st)) == KIND_POINTER) {
              CR_set_is_pointer(res, TRUE);
              if (Current_PU_is_Global()) {
                if (kind == MEMORY_KIND_PARAM) {
                  CR_set_pointee_memory_kind (res, MEMORY_KIND_GLOBAL);
                }
              }
              if (kind == MEMORY_KIND_CONST) {
                CR_set_pointee_memory_kind (res, MEMORY_KIND_GLOBAL);
              }
              work_list->Push(res); 
            } else if (TY_kind(ST_type(st)) == KIND_STRUCT) {
              if (Is_pointer_struct_member(aux_entry->St_ofst(), 
                                              ST_type(st), 0)) {
                CR_set_is_pointer(res, TRUE);
                if (Current_PU_is_Global()) {
                  if (kind == MEMORY_KIND_PARAM) {
                    CR_set_pointee_memory_kind (res, MEMORY_KIND_GLOBAL);
                  }
                }
                if (kind == MEMORY_KIND_CONST) {
                  CR_set_pointee_memory_kind (res, MEMORY_KIND_GLOBAL);
                }
                work_list->Push(res);
              }
            }
          }
        }
      } // FOR_ALL_NODE

    } // if

  } // FOR_ALL_NODE(stmt, stmt_iter, Init())

}


// Traverse each BB_NODE in the program unit, and find
// the init work list for tracking memory kind
void 
CR_MemoryMap::CFG_init_memory_kind (STACK<CODEREP *> *work_list)
{
  CFG_ITER cfg_iter(_codemap->Cfg());
  BB_NODE *bb;

  FOR_ALL_NODE( bb, cfg_iter, Init() ) {
    BB_init_memory_kind (bb, work_list);
  }
}

// Follow the forward data-flow, and propagate the memory kind
// in the program unit
void 
CR_MemoryMap::Walk_coderep_def_use (CODEREP *defCR, CODEREP *useCR, 
                                    STACK<CODEREP *> *work_list)
{
  if (WOPT_Enable_Memory_Space_Check_Debug) {
    printf("\tuse cr%d, def cr%d\n", useCR->Coderep_id(), defCR->Coderep_id());
  }

  if (useCR->Kind() != CK_VAR && useCR->Kind() != CK_OP && useCR->Kind() != CK_IVAR) {
    if (WOPT_Enable_Memory_Space_Check_Debug) {
      printf("useCR%d CK_KIND %d is not (CK_VAR | CK_IVAR | CK_OP)\n", useCR->Coderep_id(), useCR->Kind());
      _codemap->Print_CR(useCR, stdout);
      putchar('\n');
    }
    return;
  }

  if (useCR->Kind() == CK_OP) {
    if (useCR->Opr() != OPR_SELECT && useCR->Opr() != OPR_ADD && 
        useCR->Opr() != OPR_SUB && useCR->Opr() != OPR_CVT &&
        useCR->Opr() != OPR_ARRAY) {
      if (WOPT_Enable_Memory_Space_Check_Debug) {
        printf("useCR %d is not SELECT / ADD / SUB\n", useCR->Opr());
      }
      return;
    }
    if (useCR->Opr() == OPR_SELECT) {
      // first operand of SELECT can be ignored
      // onlt operand 1 and 2 are considered
      if (defCR == useCR->Opnd(0)) {
        if (WOPT_Enable_Memory_Space_Check_Debug) {
          printf("def is the first oeprand of SELECT\n");
	}
        return;
      }
    }
  }

  if (CR_get_is_pointer(defCR) == FALSE) {
    if (WOPT_Enable_Memory_Space_Check_Debug) {
      printf("defCR type is not pointer\n");
    }
    return;
  }

  MEMORY_KIND dkind = CR_get_pointee_memory_kind(defCR);
  if (useCR->Kind() != CK_IVAR) {
    CR_set_is_pointer(useCR, TRUE);

    if (Add_pointee_memory_kind (useCR, dkind)) {
      if (WOPT_Enable_Memory_Space_Check_Debug) {
        char names[80];
        printf ("use cr%d changed : kind %s\n", useCR->Coderep_id(), 
                Gen_memory_kind_name(CR_get_pointee_memory_kind(useCR), names));
      }
      work_list->Push(useCR);
    }
  } else {

    if (Add_memory_kind (useCR, dkind)) {
      if (WOPT_Enable_Memory_Space_Check_Debug) {
        char names[80];
        printf ("use cr%d changed : kind %s\n", useCR->Coderep_id(), 
                Gen_memory_kind_name(CR_get_memory_kind(useCR), names));
      }
      if (TY_kind(useCR->Ilod_ty()) == KIND_POINTER) {
        work_list->Push(useCR);
      }
    } 
  }
}

// The main interface routine for building up the Memory kind information.
void 
CR_MemoryMap::Setup_coderep_memory_kind (CODEREP_AFFECT_MAP *CR_affect_map)
{
  if (WOPT_Enable_Memory_Space_Check_Debug) {
    print_cfg_cr();
  }

  MEM_POOL bb_mem_pool;

  MEM_POOL_Initialize(&bb_mem_pool, "CR_memory_tmp_pool", FALSE);
  MEM_POOL_Push(&bb_mem_pool);

  STACK<CODEREP *> *work_list = CXX_NEW(STACK<CODEREP*>(&bb_mem_pool), &bb_mem_pool);

  // Collect initial work-list - from LDA, and Entry def
  CFG_init_memory_kind (work_list);

  // Follow the data-flow edges, and propagate the memory kind information
  while (work_list->Elements() != 0) {

    CODEREP *cr = work_list->Pop();
    CR_affect_map->Walk_all_affected_coderep (cr, FALSE, this, work_list);

  }

  // Free the memory for BB info
  MEM_POOL_Pop(&bb_mem_pool);
  MEM_POOL_Delete(&bb_mem_pool);

  if (WOPT_Enable_Memory_Space_Check_Dump) {
    CFG_dump_conflicts();
  }
}

// The constructor, allocate and initialize needed memory
CR_MemoryMap::CR_MemoryMap (CODEMAP *codemap)
{
  _codemap = codemap;
  MEM_POOL_Initialize(&_mem_pool, "CR_memory_pool", FALSE);
  MEM_POOL_Push(&_mem_pool);
  _total_CRs = codemap->Coderep_id_cnt() + 1;
  _CR_mem_info = (struct cr_mem_info *)
    MEM_POOL_Alloc (&_mem_pool, _total_CRs * sizeof(struct cr_mem_info));

  // initialize all CODEREPs' memory kind to MEMORY_KIND_INIT, and not a pointer 
  memset (_CR_mem_info, 0, _total_CRs * sizeof(struct cr_mem_info));
}

// The destructor, free up the memory space
CR_MemoryMap::~CR_MemoryMap (void)
{
  MEM_POOL_Pop(&_mem_pool);
  MEM_POOL_Delete(&_mem_pool);
}

static class WN_Memory_Space_MAP *current_wn_memory_space_map = NULL;

// For saving and retrieving memory space information.
// Since the memory space information is built on SSA-based IR, the
// information is saved during emitting WHIRL, and used in the next
// round of alias analysis.
//
class WN_Memory_Space_MAP {
  WN_MAP              _map;
  MEM_POOL           *_pu_pool;
  CR_MemoryMap       *_cr_map;

public:

  WN_Memory_Space_MAP(MEM_POOL *pu_pool, CR_MemoryMap *cr_map)
  {
    _pu_pool = pu_pool;
    _map = WN_MAP32_Create(pu_pool);
    _cr_map = cr_map;
    current_wn_memory_space_map = this;
  }

  ~WN_Memory_Space_MAP(void) { 
    if (_cr_map != NULL) {
      delete _cr_map;
    }
    WN_MAP_Delete(_map);
    current_wn_memory_space_map = NULL;
  }

  void Save_info(WN *wn, MEMORY_KIND kind) 
  {
    WN_MAP32_Set (_map, wn, (INT32)kind);
  }

  MEMORY_KIND Restore_info(WN *wn)
  {
    return (MEMORY_KIND)WN_MAP32_Get (_map, wn);
  }

  CR_MemoryMap *CR_map() { return _cr_map; }

  void Delete_CR_Map() { delete _cr_map; _cr_map = NULL; }

  MEM_POOL *Pu_pool(void)  { return _pu_pool; }
};

void
Create_Current_WN_Memory_Space_Map (CODEMAP *codemap, MEM_POOL *pu_pool)
{
  CR_MemoryMap *cr_map = new CR_MemoryMap(codemap);
  CODEREP_AFFECT_MAP *cr_aff = new CODEREP_AFFECT_MAP (codemap);
  cr_aff->CFG_setup_coderep_affect_info();
  cr_map->Setup_coderep_memory_kind(cr_aff);

  new WN_Memory_Space_MAP (pu_pool, cr_map);
}

void
Delete_Current_WN_Memory_Space_Map (void)
{
  delete current_wn_memory_space_map;
}

// Interface for deleting the CR Memory Space Map used during memory space analysis.
// After emitting, the CR Memory Space MAP is no longer needed, this is to allow
// the memory be released as soon as it is not needed.
void
Delete_Current_CR_Memory_Map (void)
{
  if(current_wn_memory_space_map) {
    current_wn_memory_space_map->Delete_CR_Map();
  }
}

// Interface for retrieving the Memory Space Information.
MEMORY_KIND 
WN_get_memory_kind (WN *wn)
{
  if (current_wn_memory_space_map == NULL) {
    return MEMORY_KIND_UNKNOWN;
  } else {
    return current_wn_memory_space_map->Restore_info(wn);
  }
}

// Interface for saving the memory space information.
void
WN_set_memory_kind (WN *wn, MEMORY_KIND kind)
{
  if (current_wn_memory_space_map == NULL) {
    return;
  } else {
    return current_wn_memory_space_map->Save_info(wn, kind);
  }
}

// Interface for retrieving Memory Space information on a CR node.
MEMORY_KIND
CR_get_memory_kind (CODEREP *cr)
{
  if (current_wn_memory_space_map == NULL) {
    return MEMORY_KIND_UNKNOWN;
  }
  return current_wn_memory_space_map->CR_map()->CR_get_memory_kind(cr);
}



// Interface for retrieving Memory Space information on a CR node.
MEMORY_KIND
CR_get_pointee_memory_kind (CODEREP *cr)
{
  if (current_wn_memory_space_map == NULL) {
    return MEMORY_KIND_UNKNOWN;
  }
  return current_wn_memory_space_map->CR_map()->CR_get_pointee_memory_kind(cr);
}

#endif // ifdef TARG_NVISA

