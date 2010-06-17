/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//-*-c++-*-
//=================================================================
//=================================================================
//
// Description:
//
//   Expression Reshaping
//   Reshape expression trees for better PRE/SR
// 
// Remarks:
//
//   Perform multilevel reassociation, distribution, folding
//
//  The optimization:
//    1) expects valid -- SSA, CFG, Coderep/Stmtrep representation
//    2) updates -- SSA, CODEREP
//    3) run anywhere in optimizer (where ssa/codereps exist)
//
//=================================================================

#ifndef opt_reshape_INCLUDED
#define opt_reshape_INCLUDED

#include "defs.h"
#include "opt_defs.h"
#include "cxx_memory.h"
#include "opcode_gen_core.h"

#include <vector>

#ifdef __STL_USE_NAMESPACES
using std::slist;
using std::vector;
#endif

class CFG;
class BB_NODE;
class BB_LOOP;
class CODEMAP;
class OPT_STAB;
class CODEREP;
class CHI_LIST;
class PHI_LIST;
class PHI_NODE;
class CODEREP;
class STMTREP;

#define RANK INT32

// Controls for the algorithm =============================

// the max depth of trees to consider
#define MAX_RESHAPE_LEVEL 16 

// control which combination technique - for experimentation
#define REASSOCIATION_OPTION 2

// Should we separate induction variables from others
#define SEPARATE_IV_LIST 0

// Toggle between top level/multi level reassociation
#define DO_ONE_LEVEL_ADD 1

// reassociate trees with multiple multiplies
#define REASSOCIATE_MULTIPLIES 0

// allow reassociation for blocks control dominated
// by loop header
#define SPECIAL_CASE_HEADER_FOR_CD 0

// find secondary iv's and mark their trees accordingly
#define FIND_SECONDARY_IV 1

// if the tree has a data-type that we don't like, and it is
// not an operator we transform, it still might be fine to
// transform this tree, since this entire node might be hoisted
#define TRY_RECOVERY_FROM_WRONG_TYPES 1

// use only the subset of operators that we see in our codes
// if you see an operator that is not getting reassociated
// either add it to list of set this flag to 0
#define LIMIT_OPERATORS 1


//==========================================================
// Description:
//  
//  CR_SUMMARY is the information collected about a coderep
//  by the RANKER
//
//==========================================================

typedef struct _CR_SUMMARY {
  RANK     rank;
  INT32    level;
  BOOL     reassoc;
  BOOL     has_iv;

public:

  _CR_SUMMARY(): rank(-1), level(-1), reassoc(FALSE), has_iv(FALSE) {}
  _CR_SUMMARY(const _CR_SUMMARY& SUM) : rank(SUM.rank), 
                                        level(SUM.level), 
                                        has_iv(SUM.has_iv), 
                                        reassoc(SUM.reassoc) {}
} CR_SUMMARY;


//==========================================================
// Description:
//
//   RANKER ranks the expressions (codereps)
//   Expression reshaping using ranks to drive choices
//   For efficiency collect other relevant information
//      
// Remarks:
//
//   Here the information collected about codereps is context
//   insensitive, since same ocderep might appear in multiple
//   contexts
//
//   Other collected info
//   level: max distance from leaf
//   has_iv: has primary(and/or) secondary iv as itself/child
//   reassoc: way to control which trees are looked at
//
//==========================================================

class RANKER {
public:
  typedef mempool_allocator<CR_SUMMARY> CR_SUMMARY_ALLOCATOR;
  typedef vector<CR_SUMMARY, CR_SUMMARY_ALLOCATOR> CR_SUMMARY_VECTOR;
  typedef mempool_allocator<INT32> BB_TO_RPO_ALLOCATOR;
  typedef vector<INT32, BB_TO_RPO_ALLOCATOR> BB_TO_RPO_VECTOR;

private:

  CFG      *_cfg;                   // cfg of the program
  CODEMAP  *_codemap;               // codemap for program
  MEM_POOL *_loc_pool;              // local pool, should be temp, memory leaked
  BOOL     _trace;                  // trace just the results
  BOOL     _verbose_trace;          // trace results and process
  INT32    _max_entry;              // max number that can be ranked

  BB_TO_RPO_VECTOR  bb_to_rpo;      // vector for bb_id to bb_rpo_id
  CR_SUMMARY_VECTOR cr_to_summary;  // vector for per coderep info

  // variable state (per block)
  BB_NODE  *_bb;           // current basic block
  INT32     _bb_rpo_id;    // current bb's rpo id
  BB_LOOP  *_loop;         // current loop

  // Local accessors
  BB_NODE* Bb()             { return _bb; }
  INT32    Bb_rpo_id()      { return _bb_rpo_id; }
  CFG *    Cfg()            { return _cfg; }
  BB_LOOP* Loop()           { return _loop; }
  INT32    Max_entry()      { return _max_entry; }
  BOOL     Trace()          { return _trace; }
  BOOL     Verbose_trace()  { return _verbose_trace; }

  // Helper functions
  BOOL   Are_leaves_invariant(CODEREP *cr);
  BOOL   Is_phi_for_iv(PHI_NODE *phi);
  void   Rank_chis(CHI_LIST *chi_list, RANK stmt_rank);
  void   Rank_phis(PHI_LIST *phi_list);
  void   Rank_statement(STMTREP *stmt);
  BOOL   Should_Reassociate_op(CODEREP *cr);

public:
  RANKER(MEM_POOL *lpool, CFG *cfg, CODEMAP *codemap, BOOL trace);
  ~RANKER(void) {};

  // Main driver
  void   Rank_function();  

  // Incremental driver (when Re_rank = 1)
  void   Rank_expression(CODEREP *cr, BOOL Re_rank);

  // helpers
  INT32  Get_rpo_id(INT32 bb_id);
  BOOL   Is_distribute_op(CODEREP *cr);
  BOOL   Is_additive_op(CODEREP *cr);
  BOOL   Is_permissible_type(CODEREP *cr);

  // coderep summary getters
  BOOL   Get_has_iv(CODEREP *cr);
  INT32  Get_level(CODEREP *cr);
  RANK   Get_rank(CODEREP *cr);
  BOOL   Get_reassociate(CODEREP *cr);

  // ranking algorithm, per basic block state setters
  void   Set_bb_rpo_id(INT32 bb_rpo_id) { _bb_rpo_id = bb_rpo_id; }
  void   Set_bb(BB_NODE * bb)           { _bb = bb; }
  void   Set_loop(BB_LOOP * loop)       { _loop = loop; }

  // coderep summary setters
  void   Set_has_iv(CODEREP *cr, BOOL has_iv);
  void   Set_level(CODEREP *cr, INT32 level);
  void   Set_rank(CODEREP *cr, RANK rank);
  void   Set_reassociate(CODEREP *cr, BOOL reassoc);
  void   Set_reassociate_check_type(CODEREP *cr, BOOL reassoc);
};


//==========================================================
// Description:
//
//   Linked list (LLIST_NODE and LLIST) is a helper for
//   reassociation.
//
//   This stores a flattened list of terms which are to
//   be put together using ADD and SUB operators to
//   reconstruct reassociated expression
//
//   This contains sorting logic
//
// Remark:
//
//   The multiple iv_heads are for experimentation
//   list can be used as if there was just one list
//   or two separate ones
//
//==========================================================

typedef struct _LLIST_NODE {
  CODEREP     *cr;       // coderep this node represents
  BOOL         sign;     // sign of coderep in final result
  BOOL         op_sign;  // sign of ops (LDA triggers unsigned)
  RANK         rank;     // rank of tree
  BOOL         has_iv;   // does the tree contain an iv cr
  _LLIST_NODE *next;     // pointer to next node in list

  BOOL Comes_after(_LLIST_NODE* node);  // sorting function
} LLIST_NODE;


class LLIST {
  LLIST_NODE *head;        // head of one list
  LLIST_NODE *iv_head;     // head of iv list
  MEM_POOL   *_loc_pool;   // pool to use for creating nodes

private:
  MEM_POOL* Pool() { return _loc_pool; }

public:
  LLIST(MEM_POOL *lpool) { 
    _loc_pool = lpool;
    head = NULL; 
    iv_head = NULL;
  }
  ~LLIST(void) {};

  // Main list construction function
  void Add_node(CODEREP *cr, BOOL sign, RANK rank, BOOL has_iv, BOOL has_sign);

  // Accessor/Setters
  LLIST_NODE* Head(BOOL iv) { return iv? iv_head: head; }
  LLIST_NODE* Next(LLIST_NODE * node) { return node->next; }
  LLIST_NODE* Pop(BOOL iv);

  void Set_head(LLIST_NODE *head, BOOL has_iv);
};

//==========================================================
// Description:
//
//   RESHAPER reshapes expression tree primarily to
//   contition the code for better code motion (partial
//   redundancy elimination) and flexible strength reduction
//
//==========================================================

enum RS_OPT_LEVEL {
  RS_ILLEGAL              = 0x00,
  RS_OPT_LEVEL_NONE       = 0x01,
  RS_OPT_LEVEL_RESTRICTED = 0x02,
  RS_OPT_LEVEL_FULL       = 0x03
};

class RESHAPER {

private:
  // Members
  CFG         *_cfg;         // cfg of the program
  CODEMAP     *_htable;      // codemap of the program
  OPT_STAB    *_opt_stab;    // symbol table
  MEM_POOL    *_loc_pool;    // pool for local allocation
  RANKER      *_ranker;      // ranking to drive reassociation
  BOOL         _trace;       // if tracing is on
  INT32        _max_level;   // max level to reshape

  // per block/expression state
  STMTREP     *_stmt;        // current statement
  BB_NODE     *_bb;          // current basic block
  BB_LOOP     *_loop;        // current loop
  INT32        _rpo_id;      // current block's rpo id
  OPCODE       _add_op;      // current add operator
  OPCODE       _sub_op;      // current sub operator
  OPCODE       _add_op_sign; // current add operator
  OPCODE       _sub_op_sign; // current sub operator
  RS_OPT_LEVEL _opt_level;   // current opt level
  
  // Getters
  OPCODE        Add_op(BOOL sign) { return sign? _add_op_sign : _add_op; }
  BB_NODE*      Bb()              { return _bb; }
  CFG         * Cfg()             { return _cfg; }
  CODEMAP     * Codemap()         { return _htable; }
  STMTREP     * Current_stmt()    { return _stmt; }
  BB_LOOP     * Loop()            { return _loop; }
  INT32         Max_level()       { return _max_level; }
  RS_OPT_LEVEL  Opt_level()       { return _opt_level; }
  OPT_STAB    * Opt_stab()        { return _opt_stab; }
  MEM_POOL    * Pool()            { return _loc_pool; }
  RANKER      * Ranker()          { return _ranker; }
  INT32         Rpo_id()          { return _rpo_id; }
  OPCODE        Sub_op(BOOL sign) { return sign? _sub_op_sign : _sub_op; }
  BOOL          Trace()           { return _trace; }

  // Setters
  void       Set_Add_Opcode(OPCODE op, BOOL sign) { if (sign) _add_op_sign = op;
                                                    else _add_op = op;  }
  void       Set_bb(BB_NODE * bb)                 { _bb = bb; }
  void       Set_current_stmt(STMTREP *stmt)      { _stmt = stmt; }
  void       Set_loop(BB_LOOP * loop)             { _loop = loop; }
  void       Set_opt_level(RS_OPT_LEVEL level)    { _opt_level = level; }
  void       Set_rpo_id(INT32 id)                 { _rpo_id = id; }
  void       Set_Sub_Opcode(OPCODE op, BOOL sign) { if (sign) _sub_op_sign = op;
                                                    else _sub_op = op;  }

  // Helpers
  CODEREP*   Add_bin_node(OPCODE opc, CODEREP *c1, CODEREP* c2);
  void       Add_to_list(CODEREP *cr, BOOL sign, RANK rank, LLIST* list);
  void       Build_flatten(CODEREP *cr, LLIST* list, BOOL sign);
  void       Consider_expression(CODEREP *cr, CODEREP *parent);
  void       Consider_statement(STMTREP *stmt);
  void       Delete(CODEREP *cr);
  CODEREP*   Distribute_node(CODEREP *cr, CODEREP *parent, INT32 child);
  CODEREP*   Distribute_tree(CODEREP *cr, CODEREP *parent, BOOL *changed);
  void       Evaluate_optimization_level(BB_NODE* bb);
  RANK       Get_rank(CODEREP *cr);
  BOOL       Has_iv(CODEREP *cr);
  BOOL       Has_lda(CODEREP *cr);
  INT32      Invariant_loop_number(CODEREP *cr);
  BOOL       Is_iv(CODEREP *cr);
  void       Print_flattened_list(LLIST *list);
  void       Rank();
  CODEREP*   Reassociate_additive_node(CODEREP *cr, CODEREP *parent);
  CODEREP*   Reassociate_additives(CODEREP *cr, CODEREP *parent,
                                   BOOL ignore_parent);
  CODEREP*   Reassociate_distributive_node(CODEREP *cr, CODEREP *parent,
                                           INT32 child);
  void       Reshape_tree(CODEREP *cr, CODEREP *parent);
  CODEREP*   Rebuild_tree(CODEREP *cr, CODEREP *parent, LLIST* list);
  void       Reduce_list(LLIST* list, BOOL iv);
  void       Remove_child(CODEREP *cr, BOOL replace_all_additives);
  void       Replace_child(CODEREP *parent, CODEREP *old_child,
                           CODEREP *new_child, BOOL replace_all_additives);
  void       Set_additive_opcodes(CODEREP *cr);
  BOOL       Should_distribute(CODEREP *other, CODEREP *add);
  BOOL       Should_reassociate_distributive(CODEREP *cr, INT32 child);
  void       Update_rank(CODEREP *cr);

 public:
  RESHAPER(MEM_POOL *lpool, CFG *cfg, OPT_STAB *opt_stab, CODEMAP *htable);
  ~RESHAPER(void) {};

  // Main Driver
  void Perform_expression_reshaping();
  
};

#endif // opt_reshape_Included
