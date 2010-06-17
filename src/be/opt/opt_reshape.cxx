/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//-*-c++-*-
//=================================================================
//
// Description:
//
//  Multilelvel expression reshaping
//
//  See header for more details
//
//=================================================================

#ifdef USE_PCH
#include "opt_pch.h"
#endif // USE_PCH

#include "defs.h"
#include "opt_reshape.h"
#include "tracing.h"
#include "wn.h"
#include "optimizer.h"
#include "opt_bb.h"
#include "opt_cfg.h"
#include "opt_sym.h"
#include "opt_htable.h"
#include "opt_etable.h"
#include "opt_main.h"
#include "opt_base.h"
#include "opt_mu_chi.h"
#include "opt_fold.h"
#include "bb_node_set.h"

using std::max;

//==========================================================
// RANKER functions
//
//==========================================================

RANKER::RANKER(MEM_POOL *lpool, CFG *cfg,
               CODEMAP *codemap, BOOL trace)
{
  _loc_pool = lpool; 
  _cfg = cfg;
  _trace = trace;
  _codemap = codemap;

  // for now use only one level for tracing
  _verbose_trace = trace;

  // The size is wasteful, vector can grow when needed
  // but we expect a default value to be always there
  // remove this when growing with default insertion is done

  _max_entry = codemap->Coderep_id_cnt() * 2;

  // create stack object using default values to be
  // replicated throughout the vector

  CR_SUMMARY a;
  cr_to_summary =
    CR_SUMMARY_VECTOR(_max_entry, a,
                      CR_SUMMARY_VECTOR::allocator_type(lpool));
  bb_to_rpo = 
    BB_TO_RPO_VECTOR(_max_entry, 0, 
                       BB_TO_RPO_VECTOR::allocator_type(lpool));
}

//==========================================================
// Description:
// 
//   Basic Block id to Reverse post order id mapping
//  
//==========================================================

INT32 RANKER::Get_rpo_id(INT32 bb_id) 
{
  if (bb_id > Max_entry())
    return -1;
  
  return bb_to_rpo[bb_id];
}

//==========================================================
// Description:
// 
//   Is a permissible type for distributive/additive
//   operation
//  
//==========================================================

BOOL RANKER::Is_permissible_type(CODEREP *cr)
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Is_permissible_type"));
  Is_True(cr->Kind() == CK_OP, ("Not CK_OP:Is_permissible_type"));

  MTYPE rty = OPCODE_rtype(cr->Op());
  
  if (!(   (rty == MTYPE_I4) 
           || (rty == MTYPE_I8)
           || (rty == MTYPE_U4) 
           || (rty == MTYPE_U8)))
    return FALSE;
  return TRUE;
}

//==========================================================
// Description:
// 
//   Rank getter/setter
//  
//==========================================================

void RANKER::Set_rank(CODEREP *cr, RANK rank) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Set_rank"));

  if (cr->Coderep_id() > Max_entry())
    return;

  if (Trace())
    fprintf(TFile, "--set_rank(cr%d)=%d ",
            cr->Coderep_id(), rank);

  cr_to_summary[cr->Coderep_id()].rank = rank;
}

RANK RANKER::Get_rank(CODEREP *cr) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Get_rank"));
  
  if (cr->Coderep_id() > Max_entry())
    return -1;
  
  return cr_to_summary[cr->Coderep_id()].rank;
}

//==========================================================
// Description:
// 
//   Level getter setter 
//   (level is max distance from leaves)
//  
//==========================================================

void RANKER::Set_level(CODEREP *cr, INT32 level) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Set_level"));

  if (cr->Coderep_id() > Max_entry())
   return;

  if (Trace())
    fprintf(TFile, "--set_level(cr%d)=%d ", 
            cr->Coderep_id(), level);

  cr_to_summary[cr->Coderep_id()].level = level;
}

INT32 RANKER::Get_level(CODEREP *cr) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Get_level"));

  if (cr->Coderep_id() > Max_entry())
    return -1;

  return cr_to_summary[cr->Coderep_id()].level;
}

//==========================================================
// Description:
// 
//   Has_iv getter setter 
//   there is iv under the current coderep tree
//  
//==========================================================

void RANKER::Set_has_iv(CODEREP *cr, BOOL has_iv) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Set_level"));

  if (cr->Coderep_id() > Max_entry())
   return;

  if (Trace())
    fprintf(TFile, "--set_has_iv(cr%d)=%d ", 
            cr->Coderep_id(), has_iv);

  cr_to_summary[cr->Coderep_id()].has_iv = has_iv;
}

BOOL RANKER::Get_has_iv(CODEREP *cr) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Get_level"));

  if (cr->Coderep_id() > Max_entry())
    return FALSE;

  return cr_to_summary[cr->Coderep_id()].has_iv;
}

//==========================================================
// Description:
// 
//   Reassociate getter/setter
//
// Remarks:
//
//   The entire tree under a coderep marked reassociable 
//   can be reassociated
//   This marks only the immediate coderep, not the tree
//  
//==========================================================

void RANKER::Set_reassociate(CODEREP *cr, BOOL reassoc)
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Set_reassociate"));

  if (cr->Coderep_id() > Max_entry())
    return;

  if (Trace())
    fprintf(TFile, "--set_reassoc(cr%d)=%d ",
            cr->Coderep_id(), reassoc);

  cr_to_summary[cr->Coderep_id()].reassoc = reassoc;
}

BOOL RANKER::Get_reassociate(CODEREP *cr) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Get_reassociate"));

  if (cr->Coderep_id() > Max_entry())
    return FALSE;

  return cr_to_summary[cr->Coderep_id()].reassoc;
}

void RANKER::Set_reassociate_check_type(CODEREP *cr,
                                        BOOL reassoc) 
{
  Is_True(cr != NULL, ("NULL CR: RANKER:Set_reassociate"));

  //discard things with wrong types
  if ((cr->Kind() == CK_OP) && (reassoc == TRUE)) {
    if (!Is_permissible_type(cr)) {
      if (Trace())
        fprintf(TFile, "--reassoc_requested=%d ",reassoc);

      reassoc = FALSE;

      if (Trace())
        fprintf(TFile, "--reassoc_granted=%d, rtype=%d \n",
                reassoc, OPCODE_rtype(cr->Op()));
    }
  }
  Set_reassociate(cr, reassoc);
}

//==========================================================
//  Description:
// 
//    If the cr is an operator, do we reassociate it
//  
//==========================================================

BOOL RANKER::Should_Reassociate_op(CODEREP *cr)
{
  if (!LIMIT_OPERATORS)
    return TRUE;

  switch (cr->Opr()) {
  case OPR_SUB:
  case OPR_ADD:
  case OPR_MPY:
    return TRUE;

  case OPR_CVT:
  case OPR_BAND:
  case OPR_INTRINSIC_OP:
  case OPR_LSHR:
  case OPR_LT:
  case OPR_GT:
  case OPR_LE:
  case OPR_GE:
  case OPR_SELECT:
  case OPR_SHL:
  case OPR_ASM_INPUT:
    return TRUE;
  default:
    return FALSE;
  }
}

//==========================================================
// Description:
// 
//   Whether the current operation is distributive/additive
//   The intention is to return the operators we want to
//   handle not the subset of operators in the IR
//   that have this property
//
// Remark:
//
//   Distributive op is guaranteed to be commutative
//   Additve_op is not
//
//==========================================================

BOOL RANKER::Is_distribute_op(CODEREP *cr) {
  if (cr->Kind() != CK_OP)
    return FALSE;
  if (cr->Opr() == OPR_MPY)
    return TRUE;
  return FALSE;
}

//==========================================================
// Description:
// 
//   Whether the current operation is distributive/additive
//   The intention is to return the operators we want to
//   handle not the subset of operators in the IR
//   that have this property
//
// Remark:
//
//   Distributive op is guaranteed to be commutative
//   Additve_op is not
//
//==========================================================

BOOL RANKER::Is_additive_op(CODEREP *cr) {
  if (cr->Kind() != CK_OP)
    return FALSE;
  if ((cr->Opr() == OPR_ADD)
      || (cr->Opr() == OPR_SUB))
    return TRUE;
  return FALSE;
}

//==========================================================
// Description:
// 
//   Check if leaves are invariant in current loop. This
//   is to acertain this information before the coderep
//   is ranked
//  
//==========================================================

BOOL RANKER::Are_leaves_invariant(CODEREP *cr)
{
  INT32 bb_rpo_id = Bb_rpo_id();
  
  if (Get_rank(cr) != -1)
    return (Get_rank(cr) < bb_rpo_id);
  
  switch (cr->Kind()) {

  case CK_CONST:
  case CK_RCONST:
    return TRUE;
    break;

  case CK_OP:
    {
      BOOL is_invariant = TRUE;
      for (INT32 i = 0; i < cr->Kid_count(); i++) {
        CODEREP * i_child = cr->Opnd(i); 
        is_invariant &= Are_leaves_invariant(i_child);
      }
      return is_invariant;
    }
    break;

  default:
    return FALSE;
    break;
  } // switch (coderep kind)
}

//==========================================================
// Description:
// 
//   Check if phi is for primary/secondary iv
//
// Remarks:
//
//   The secondary iv test is limited in that it does not
//   chase through copies yet
//  
//==========================================================

BOOL RANKER::Is_phi_for_iv(PHI_NODE *phi)
{
  //check primary iv
  const BB_LOOP *loop = Loop();
  if ((loop->Iv() != NULL)
      && (phi->RESULT()->Aux_id() == loop->Iv()->Aux_id()))
    return TRUE;


  //check marked iv
  if (phi->Opnd_iv_update() )
    return TRUE;

          
  //check secondary iv (approximation)
  if (FIND_SECONDARY_IV) {
    PHI_OPND_ITER phi_opnd_iter(phi);
    CODEREP *opnd;
    FOR_ALL_ELEM(opnd, phi_opnd_iter, Init()) {
      // don't know where the def of zero-version vars are
      // don't want the hassle of phi/chi updates ones
      if ((opnd->Is_flag_set(CF_IS_ZERO_VERSION)) ||
          (opnd->Is_flag_set((CR_FLAG)(CF_DEF_BY_CHI|CF_DEF_BY_PHI)))){
        continue;
      }

      STMTREP *stmt = opnd->Defstmt();
      CODEREP *lhs = stmt->Lhs();
      CODEREP *rhs = stmt->Rhs();
      const OPERATOR opr = stmt->Opr();

      // Check restricted form
      if (!OPERATOR_is_scalar_store(opr) ||
          !MTYPE_is_integral(OPCODE_desc(stmt->Op())) ||
          stmt->Volatile_stmt() ||
          (lhs->Kind() != CK_VAR) ||
          (rhs->Kind() != CK_OP) ||
          !Is_additive_op(rhs)) {
        continue;
      }

      CODEREP * curr_node = rhs;

      // we are finding if tree contains an add of iv
      // and invariant. We can recurse with while loop
      // since in every generation all but one child
      // should be invariant, we follow that child

      while((curr_node != NULL) &&
            (curr_node->Kind() == CK_OP) &&
            (Is_additive_op(curr_node))) {

        CODEREP * child_1 = curr_node->Opnd(0);
        CODEREP * child_2 = curr_node->Opnd(1);
        BOOL child_1_invar = Are_leaves_invariant(child_1);
        BOOL child_2_invar = Are_leaves_invariant(child_2);

        BOOL consider_child_2 = (curr_node->Opr() == OPR_ADD);

        if ((child_1->Kind() == CK_VAR) &&
            (child_2_invar) &&
            (child_1->Aux_id() == lhs->Aux_id()) &&
            (child_1->Coderep_id() == phi->RESULT()->Coderep_id())) {

          return TRUE;
        }
      
        if ((child_2->Kind() == CK_VAR) &&
            (child_1_invar) &&          
            (consider_child_2) &&
            (child_2->Aux_id() == lhs->Aux_id()) &&
            (child_2->Coderep_id() == phi->RESULT()->Coderep_id())) {

          return TRUE;
        }

        // If only one child is variant, see if we can go down

        if (child_1_invar != child_2_invar) {

          curr_node = child_1_invar ? 
            (consider_child_2 ? child_2 : NULL) :
            (child_1);

        } else {
          curr_node = NULL;
        }

      } // while( )

    } // FOR_ALL_ELEM phi opnd
  } // FIND_SECONDARY_IV

  return FALSE;
}

//==========================================================
// Description:
// 
//   Ranks all the phi's in the list
//   Marks iv's for all loops
// 
// Remarks:
//
//   Phi dst's get the rank of basic block (its rpo_id)
//   this ensures that loop variants (who have phis)
//   get higher rank than loop invariants
//  
//==========================================================

void RANKER::Rank_phis(PHI_LIST *phi_list) 
{
  INT32 bb_rpo_id = Bb_rpo_id();
  PHI_NODE     *phi;
  PHI_LIST_ITER phi_iter;

  BOOL find_ivs = FALSE;
  if ((Bb() != NULL) 
      && (Loop() != NULL)
      && (Loop()->Header() != NULL)
      && Bb()->Id() == Loop()->Header()->Id())
    find_ivs = TRUE;

  FOR_ALL_ELEM(phi, phi_iter, Init(phi_list)) {

    if (phi->Live()) {

      if (find_ivs) {
        // If we determined this an iv update, mark it so
        if (Is_phi_for_iv(phi)) {

          if (Trace())
            fprintf(TFile,"\nphi is iv, mark operands");

          Set_has_iv(phi->RESULT(), TRUE);
          PHI_OPND_ITER phi_opnd_iter(phi);
          CODEREP *opnd;
          FOR_ALL_ELEM(opnd, phi_opnd_iter, Init()) { 
            Set_has_iv(opnd, TRUE);
          }
        }
      } // if (find_ivs)

      if (Verbose_trace())
        fprintf(TFile, "\nRank phi:");

      Set_rank(phi->RESULT(), (RANK) bb_rpo_id);
    }
  }
}

//==========================================================
// Description:
// 
//   Ranks all the chi's in the list
// 
// Remarks:
//
//   if statement a1<-, may-defs b1<-, into b2<-chi(b1) 
//   b2's rank should be max of a1 and b1, since it is
//   written by one of them
//
//==========================================================

void RANKER::Rank_chis(CHI_LIST *chi_list, RANK stmt_rank) 
{
  INT32 bb_rpo_id = Bb_rpo_id();
   CHI_LIST_ITER chi_iter;
   CHI_NODE     *chi;

   FOR_ALL_NODE(chi, chi_iter, Init(chi_list)) {

      if (chi->Live()) {

        CODEREP *chi_result = chi->RESULT();
        CODEREP *chi_source = chi->OPND();

        // This max amongst other things ensures that
        // undefined things get reasonable ranks
        // including parameters (see trace of example)
        RANK chi_rank = max(Get_rank(chi_source), stmt_rank);

        if (Verbose_trace())
          fprintf(TFile, "\nRank chi:");

        Set_rank(chi_result, chi_rank);
      }
   }
}

//==========================================================
// Description:
// 
//   Recursively rank codereps.
//
//   Re-rank option is when transformation has been made
//   to an expression tree. The leaves are expected to be
//   ranked generally (except new folded constants) though
//   leaves may also be re-ranked (using default values)
//
// Note:
//
//   When calling with re-rank option, ensure that the 
//   per basic block state information (bb,loop, bb_rpo_id)
//   is current
// 
// Remarks:
//
//   The rules are:
//   1) LDA/default    - block rank
//   2) CONSTS         - zero (0)
//   3) Operators/Ilod - max of kids
//   4) LDID           - the value loaded should already
//                      be ranked when it was stored since
//                      we are doing a RPO walk of the tree
//
//   And this function has evolved to be ugly, clean up later
//
// Parameters:
//
//  Re-rank forces re-ranking from leaves,see note
//  
//==========================================================

void RANKER::Rank_expression(CODEREP *cr, BOOL Re_rank)
{
  INT32 bb_rpo_id = Bb_rpo_id();

  if (Verbose_trace())
    fprintf(TFile, "\nLooking at Coderep: %d :", 
            cr->Coderep_id());

  // If Re_rank, leaves do not need to be re-ranked

  if (Re_rank){
    switch (cr->Kind()) {
    case CK_CONST:
    case CK_RCONST:
      if (Get_rank(cr) != -1)
        return;
      // looks like a new constant created by folding
      Set_rank(cr, 0);
      Set_level(cr, 0);
      Set_reassociate_check_type(cr, TRUE);
      
    case CK_LDA:
    case CK_VAR:
      if (Get_rank(cr) != -1)
        return;
    }
  }

  // Rank if not already ranked
  if ((Get_rank(cr) == -1) || Re_rank) {
    
    switch (cr->Kind()) {

    case CK_LDA:
      if (Verbose_trace())
        fprintf(TFile, "\nLDA block ranked");
      Set_rank(cr, (RANK) bb_rpo_id);
      Set_level(cr, 0);
      Set_reassociate_check_type(cr, TRUE);
      break;


    case CK_CONST:
    case CK_RCONST:
      if (Verbose_trace())
        fprintf(TFile, "\nConst zero ranked");
      Set_rank(cr, (RANK) 0);
      Set_level(cr, 0);
      Set_reassociate_check_type(cr, TRUE);
      break;


    case CK_IVAR:
      if ((cr->Opr() == OPR_ILOAD) ||
          (cr->Opr() == OPR_PARM)) {
        RANK rank = 0;
        INT32 level = 0;
        BOOL has_iv = FALSE;
        // iload can be in laoad or store
        CODEREP * base = cr->Ilod_base();
        if (base == NULL)
          base = cr->Istr_base();

        if (base != NULL) {
          Rank_expression(base, Re_rank);
          rank = Get_rank(base);
          level = Get_level(base);
          has_iv = Get_has_iv(base);
        }
        if (Verbose_trace())
          fprintf(TFile, "\nIlod Base ranked");
        Set_rank(cr, rank);
        Set_level(cr, level+1);
        Set_has_iv(cr, has_iv);
        Set_reassociate_check_type(cr, (cr->Opr() == OPR_PARM));
      }
      break;


    case CK_OP:
      {
        RANK rank = 0;
        INT32 level = 0;
        BOOL reassociate = TRUE;
        BOOL has_iv = FALSE;

        for (INT32 i = 0; i < cr->Kid_count(); i++) {
          CODEREP * i_child = cr->Opnd(i); 
          Rank_expression(i_child, Re_rank);
          rank  = max(rank,  Get_rank(i_child));
          level = max(level, Get_level(i_child));

          BOOL this_child_reassoc = Get_reassociate(i_child);

          if (this_child_reassoc == FALSE) {
            if (TRY_RECOVERY_FROM_WRONG_TYPES
                && (reassociate == TRUE)
                && (Get_reassociate(i_child) == FALSE) 
                && (!Is_additive_op(cr) && !Is_distribute_op(cr))
                && (i_child->Kind() == CK_OP)
                && !Is_permissible_type(i_child)
                && Is_permissible_type(cr)
                ) {
              this_child_reassoc = TRUE;
            }
          }

          reassociate = reassociate & this_child_reassoc;
          has_iv      = has_iv      | Get_has_iv(i_child);
        }
        if (Verbose_trace())
          fprintf(TFile, "\nMax child ranked");
        Set_rank(cr, rank);
        Set_level(cr, level+1);
        Set_has_iv(cr, has_iv);
        Set_reassociate_check_type(cr, reassociate &&
                                   Should_Reassociate_op(cr));        

      }
      break;

    case CK_VAR:
      // since we are in ssa we should not be coming here too often
      Set_rank(cr, bb_rpo_id - 1);
      Set_level(cr, 0);
      Set_reassociate_check_type(cr, TRUE);
      break;
	 
    case CK_DELETED:
    default:
      FmtAssert(FALSE, ("Unexpected CR kind in RANKER"));
      break;
    } // switch (coderep kind)

    if (Get_rank(cr) == -1) {
      if (Verbose_trace())
        fprintf(TFile, "\nDefault ranked");
      Set_rank(cr, bb_rpo_id);
      Set_reassociate_check_type(cr, FALSE);
    }
    
  } else { // if (Get_rank(cr) == -1)

    // already ranked coderep

    if (Verbose_trace())
      fprintf(TFile, "--[[ old_rank:cr-%d<--%d ]]", 
              cr->Coderep_id(), Get_rank(cr));

    // level, reassoc may not be set it got rank
    // from being lhs of statement

    if (Get_level(cr) == -1) {
      switch (cr->Kind()) {
      case CK_VAR:        
      case CK_CONST:
      case CK_RCONST:
      case CK_LDA:        
        Set_reassociate_check_type(cr, TRUE);
        Set_level(cr, 0);
        break;

      case CK_IVAR:        
      case CK_OP:	 
      case CK_DELETED:
      default:
        FmtAssert(FALSE, ("Unexpected CR kind with no level"));
        break;
      }
    }
    return;
  }
}

//==========================================================
// Description:
// 
//   Rank one statement
// 
//==========================================================

void RANKER::Rank_statement(STMTREP *stmt)
{
  CODEREP  *rhs = stmt->Rhs();
  CODEREP  *lhs = stmt->Lhs();

  INT32 bb_rpo_id = Bb_rpo_id();

  if (!stmt->Black_box()) {

    switch (stmt->Opr()) {
     
    case OPR_STID:
      Is_True(lhs->Kind() == CK_VAR, ("Unexpected opc: RANKER"));
      Rank_expression(rhs, FALSE);
      if (Verbose_trace())
        fprintf(TFile, "\nSTMT ranked");
      Set_rank(lhs, max(0, Get_rank(rhs)));
      break;

    case OPR_ISTORE:
      Is_True(lhs->Kind() == CK_IVAR, ("Unexpected opc: RANKER"));
      Rank_expression(lhs->Istr_base(), FALSE);
      Rank_expression(rhs, FALSE);
      //      Set_rank(lhs, max(Get_rank(lhs), Get_rank(rhs)));
      if (Verbose_trace())
        fprintf(TFile, "\nSTMT ranked-lhs-rhs-");
      break;

    default:
      RANK rank = bb_rpo_id;
      if (rhs != NULL) {
        if (Verbose_trace())
          fprintf(TFile, "\nSTMT ranked");
        Rank_expression(rhs, FALSE);
        rank = Get_rank(rhs);
      }
      if (lhs != NULL) {
        if (Verbose_trace())
          fprintf(TFile, "\nSTMT ranked");
        Set_rank(lhs, rank);
      } 
      break;
    }
  }

  // Handle he chis(may-defs) associated with this statement
  if (stmt->Has_chi()) {
    RANK stmt_rank = (lhs == NULL)? bb_rpo_id : Get_rank(lhs);
    Rank_chis(stmt->Chi_list(), stmt_rank);
  }

  // newline between statements
  if (Verbose_trace())
    fprintf(TFile, "\n"); 

}

//==========================================================
// Description:
// 
//   This is the main driver
//
//   Do an rpo traveral of blocks and rank statements and
//   expressions
//  
//==========================================================

void RANKER::Rank_function()
{
  if (Trace())
    fprintf(TFile, "Ranking the function\n");

  BB_NODE   * bb;
  RPOBB_ITER  cfg_iter(Cfg());

  INT32 bb_rpo_id = 1;

  FOR_ALL_ELEM(bb, cfg_iter, Init()) {
    if (Trace())
      fprintf(TFile, 
              "\nBASIC BLOCK: ID(%d): RPO(%d)============\n",
              bb->Id(), bb_rpo_id);

    // Build the map for reshaper incrementally
    bb_to_rpo[bb->Id()] = bb_rpo_id;

    // Set the state
    Set_bb(bb);
    Set_bb_rpo_id(bb_rpo_id);
    Set_loop(bb->Innermost());   // can be null

    // Rank phi's and statements (and chis)

    Rank_phis(bb->Phi_list());
      
    STMTREP     *stmt;
    STMTREP_ITER stmt_iter(bb->Stmtlist());

    FOR_ALL_NODE(stmt, stmt_iter, Init()) {
      Rank_statement(stmt);
    }
    bb_rpo_id++;
  }

  return;
}

//==========================================================
//  LLIST/ LLIST_NODE functions
//==========================================================

//==========================================================
// Description:
// 
//   This primarily decides the order for reassociation
//   by achieving sorting as values are added into the list
//
// Parameters:
//
//   node - decide if node comes after this in sorting
//
//   return TRUE if 'this' should come after the 'node'
//
// Remarks:
//
//   Criteria for deciding order
//   0) LDA         Vs non-LDA   => Symbol compare
//   1) Rank        Vs Rank     
//   2) Leaf        Vs Non-Leaf  
//   3) Leaf        Vs Leaf      => Symbol compare
//   4) Non-leaf    Vs Non-leaf  => Operator compare
//
//   Memory offsets must be fully evaluated before adding
//   addresses, otherwise there may be negative values in
//   unsigned pointer types.  Thus, LDAs are after
//   everything else in the sorted list.
//  
//==========================================================

BOOL _LLIST_NODE::Comes_after(LLIST_NODE* node)
{
  CODEREP *cr1       = node->cr;

  // LDA Vs non-LDA
  if (cr->Kind() == CK_LDA) {
    if (cr1->Kind() != CK_LDA)
      return TRUE;
  } else {
    if (cr1->Kind() == CK_LDA)
      return FALSE;
  }

  // rank Vs rank
  if (node->rank > rank)
    return FALSE;
  else if (node->rank < rank)
    return TRUE;

  BOOL     cr_leaf   = !cr->Non_leaf();
  BOOL     cr1_leaf  = !cr1->Non_leaf();

  if (has_iv && !node->has_iv)
    return TRUE;
  else if (!has_iv && node->has_iv)
    return FALSE;

  // leaf vs non-leaf
  if (cr_leaf != cr1_leaf) {
    return (cr_leaf)? FALSE: TRUE;

  // both leaves
  } else if (cr_leaf) {

    // const and ivar don't have associated symbols
    if (cr->Kind() == cr1->Kind()) {
      if (cr->Kind() == CK_CONST || cr->Kind() == CK_IVAR)
        return TRUE;
      else
        return CR_Compare_Symbols(cr, cr1);
    }
    
    // what kind of leaf?
    switch (cr->Kind()) {
      
    case CK_CONST:
    case CK_RCONST:
      return TRUE;
      
    case CK_VAR:
      switch(cr1->Kind()) {
      case CK_CONST:
      case CK_RCONST:
        return FALSE;
      default:
        return TRUE;
      }
    case CK_LDA:        
    default:
      return FALSE;
    }

  // both operators of equal rank, improve this
  } else {
    return TRUE;
  }
}

//==========================================================
// Description:
// 
//   This sets the head of one of the lists
//
// Parameters:
//
//   is_iv - which head? TRUE implies head of iv list
//
//==========================================================

void LLIST::Set_head(LLIST_NODE *node, BOOL is_iv) 
{
  if (is_iv)
    iv_head = node;
  else
    head = node;
}

//==========================================================
// Description:
// 
//   This creates a node and adds it to appropriate list
//
// Parameters:
//
//   has_iv - if cr tree (or leaf) has an iv under it
//
//==========================================================

void LLIST::Add_node(CODEREP *cr, BOOL sign, RANK rank, 
                     BOOL has_iv, BOOL op_sign)
{
  LLIST_NODE *node;
  node = (LLIST_NODE*) CXX_NEW(LLIST_NODE, Pool());  
  node->cr = cr;
  node->sign = sign;
  node->rank = rank;
  node->has_iv = has_iv;
  node->op_sign = op_sign;

  if (!SEPARATE_IV_LIST)
    has_iv = FALSE;

  LLIST_NODE * cur_head = Head(has_iv);

  if (cur_head == NULL) {
    cur_head = node;
    node->next = NULL;

  } else {

    if (rank < cur_head->rank ||
	(cur_head->cr->Kind() == CK_LDA &&  // LDAs must be at tail
	 node->cr->Kind() != CK_LDA)) {
      node->next = cur_head;
      cur_head = node;
    } else {
      LLIST_NODE * insert_after = cur_head;
      while ((insert_after->next != NULL)
             && node->Comes_after(insert_after->next)) {
        insert_after = insert_after->next;
      }
      node->next = insert_after->next;
      insert_after->next=node;
      // only in this case head does need updating
      return;
    }
  }
  Set_head(cur_head, has_iv);
}

//==========================================================
// Description:
// 
//   This pops (and returns) the head of one list
//
// Parameters:
//
//   is_iv - which head? TRUE implies head of iv list
//
//==========================================================

LLIST_NODE* LLIST::Pop(BOOL iv)
{
  LLIST_NODE * ret = Head(iv);
  if (ret != NULL)
    Set_head(Next(ret), iv);

  return ret;
}

//==========================================================
//  RESHAPE functions
//==========================================================

RESHAPER::RESHAPER(MEM_POOL *lpool, CFG *cfg,
                  OPT_STAB *opt_stab, CODEMAP *htable) { 
  _loc_pool = lpool; 
  _htable = htable;
  _cfg = cfg;
  _opt_stab = opt_stab;
  _trace = FALSE;
  _max_level = MAX_RESHAPE_LEVEL;

  if (Get_Trace(TP_WOPT2, RESHAPE_FLAG)) 
    _trace = TRUE;

}

//==========================================================
// Description:
// 
//  Helper functions
//
//   Is_iv: if the immediate coderep represents an iv 
//          variable in the current loop nest
//
//   Has_iv:(recursive) if the coderep includes an iv 
//          in its tree
//
//   Has_lda: if coderep includes lda in its tree
//
//==========================================================

BOOL RESHAPER::Is_iv(CODEREP *cr)
{
  if (cr->Kind() != CK_VAR)
    return FALSE;
  
  const BB_LOOP *loop = Current_stmt()->Bb()->Innermost();

  while(loop != NULL) {
    if ((loop->Iv() != NULL)
        && (cr->Coderep_id() == loop->Iv()->Coderep_id()))
      return TRUE;
    loop = loop->Parent();
  }
  return FALSE;
}

BOOL RESHAPER::Has_iv(CODEREP *cr)
{
  if (cr->Kind() == CK_OP) {
    BOOL has_iv = FALSE;
    for (INT32 i = 0; i < cr->Kid_count(); i++)
      has_iv = has_iv | Has_iv(cr->Opnd(i)); 
    return has_iv;

  } else if (cr->Kind() == CK_VAR) {
    return Is_iv(cr);
  }
  return FALSE;
}

BOOL RESHAPER::Has_lda(CODEREP *cr)
{
  if (cr->Kind() == CK_LDA)
    return TRUE;

  if (cr->Kind() == CK_OP) {
    BOOL has_lda = FALSE;
    for (INT32 i = 0; i < cr->Kid_count(); i++)
      has_lda = has_lda | Has_lda(cr->Opnd(i)); 
    return has_lda;
  }

  return FALSE;
}

//==========================================================
// Description:
// 
//  Reduce a use of coderep
//  Since use counts are not accurate, only decrement
//  a usecount. Not sure of how to handle this correctly
//
//==========================================================

void RESHAPER::Delete(CODEREP *cr) {
  if (cr->Usecnt() > 1)
    cr->DecUsecnt();
  //  else
  //  Codemap()->Remove(cr);
}

//==========================================================
// Description:
// 
//   Add a node in list with right header
//
//==========================================================

void RESHAPER::Add_to_list(CODEREP *cr, BOOL sign, RANK rank,
                          LLIST* list) 
{
  BOOL has_iv = (Loop())? Has_iv(cr) : FALSE;

  // use unsigned ops when lda present in tree
  BOOL has_op_sign = Has_lda(cr)? FALSE: TRUE;

  // small optimization
  if (cr->Kind() == CK_OP) {
    if (cr->Opr() == OPR_NEG) {
      cr = cr->Opnd(0);
      sign = !sign;
    }
  }
  list->Add_node(cr, sign, rank, has_iv, has_op_sign);
}

//==========================================================
// Description:
// 
//   Remove_child and Replace_child work in concert to
//   replace a coderep with another.
//
//   In terms of functionality, there are 2 things
//
//   1) 1-Level replace (default)
//      This replaces just the immediate old_child with
//      new child
//
//   2) Multi-level replace (replace_all_additives = true)
//      This assumes that a parallel tree has been created
//      and old_child and all its childred which are
//      additive_ops are redundant and need to be removed
//
// Parameter:
//
//   parent: parent coderep whose old_child needs to be
//           replaced by old child. If parent is null it
//           is assumed that parent is the current stmt
//           (and cannot be represented as Coderep *)
//   others: see description above
//
//==========================================================

void RESHAPER::Remove_child(CODEREP *cr, 
                           BOOL replace_all_additives) 
{
  if (!replace_all_additives) {
    Delete(cr);
    return;
  }
  
  if ((cr->Kind() == CK_OP) && Ranker()->Is_additive_op(cr)) {
    for (INT32 i = 0; i < cr->Kid_count(); i++)
      Remove_child(cr->Opnd(i), replace_all_additives);
    Delete(cr);
  }
}

void RESHAPER::Replace_child(CODEREP *parent, 
                            CODEREP *old_child, 
                            CODEREP *new_child,
                            BOOL replace_all_additives) {

  if (parent == NULL) {
    
    if (Current_stmt()->Lhs() == old_child) {
      Remove_child(old_child, replace_all_additives);
      Current_stmt()->Set_lhs(new_child);
      
    } else if (Current_stmt()->Rhs() == old_child) {
      Remove_child(old_child, replace_all_additives);
      Current_stmt()->Set_rhs(new_child);
      
    } else if (Current_stmt()->Opr() == OPR_ISTORE) {
      FmtAssert(Current_stmt()->Lhs()->Istr_base() == old_child,
                ("Replace_child: no parent"));  
      Remove_child(old_child, replace_all_additives);
      new_child->IncUsecnt();
      Current_stmt()->Lhs()->Set_istr_base(new_child);
      if (Current_stmt()->Lhs()->Ilod_base() == old_child)
        Current_stmt()->Lhs()->Set_ilod_base(new_child);
      
    } else {
      FmtAssert(FALSE, ("Replace_child: no parent"));  
    }
    return;
  }
  
  if (parent->Kind() == CK_OP) {
    for (INT32 i = 0; i < parent->Kid_count(); i++)
      if (old_child == parent->Opnd(i)) {
        Remove_child(old_child, replace_all_additives);
        new_child->IncUsecnt();
        parent->Set_opnd(i, new_child);
        break;
      }
  } else if (parent->Opr() == OPR_ILOAD) {
    
    if (parent->Ilod_base() != NULL) {
      FmtAssert(parent->Ilod_base() == old_child,
                ("distribute ILOAD wrong base"));  
      Remove_child(old_child, replace_all_additives);
      new_child->IncUsecnt();
      parent->Set_ilod_base(new_child);


    }else if (parent->Istr_base() != NULL) {
      FmtAssert(parent->Istr_base() == old_child,
                ("distribute ISTR wrong base"));  
      Remove_child(old_child, replace_all_additives);
      new_child->IncUsecnt();
      parent->Set_istr_base(new_child);
    }
  } else {
    FmtAssert(FALSE, ("Replace_child: wrong parent"));  
  }
}

//==========================================================
// Description:
// 
//  Wrapper to have a uniform way for this optimization
//  to add a binary node into codemap
//
//  Attempts simplification since many times distribution
//  results in opportunities for constant folding
//
//=========================================================

CODEREP* RESHAPER::Add_bin_node(OPCODE opc, CODEREP *c1, CODEREP* c2)
{
  CODEREP* cr = NULL;
  
  if ((c1->Kind() == CK_CONST) ||
      (c2->Kind() == CK_CONST)) {
    cr = SIMPNODE_SimplifyExp2(opc, c1, c2);
  }
  
  if (cr == NULL) {
    cr =  Codemap()->Add_bin_node(opc, // opcode
                                  c1,  // kid 0
                                  c2); // kid 1
  }
  return cr;
}

//==========================================================
// Description:
// 
//   Determine the number of loops starting outward from
//   current loop, where this coderep is invariant 
//
//=========================================================

INT32 RESHAPER::Invariant_loop_number(CODEREP *cr) 
{
  INT32 num = 0;
  BB_LOOP* loop = Loop();
  INT32 cr_rpo_id = (INT32) Ranker()->Get_rank(cr);
  
  while (loop != NULL) {
    BB_NODE * header = loop->Header();
    if (header == NULL) {
      FmtAssert((header != NULL), ("Loop null header"));   
      break;
    }
    INT32 loop_rpo_id = Ranker()->Get_rpo_id(header->Id());

    if (cr_rpo_id < loop_rpo_id)
      num++;
    else if (Trace())
      fprintf(TFile, "cr%d-loop_variant:-cr_rpo_id(%d)>loop_rpo_id(%d)\n", 
              cr->Coderep_id(), cr_rpo_id, loop_rpo_id);

    loop = loop->Parent();
  }
  return num;
}

//==========================================================
// Description:
// 
//   Evaluate how much optimization to do based on control
//   dependence. This is a measure of the ability of
//   downstream PRE and SR to remove code from this basic
//   block
//
// Remarks:
//
//   Takes into account control dependence and early exits
//
//==========================================================

void RESHAPER::Evaluate_optimization_level(BB_NODE* bb) 
{
  BB_LOOP * loop = Loop();

  if (loop == NULL) {
    Set_opt_level(RS_OPT_LEVEL_NONE);
    return;
  }

  if (Control_dependence_implies_down_safety(bb,
                                             loop,
                                             Pool())) {
    if (Trace())
      fprintf(TFile, "\nSame CD, full optimization");
    Set_opt_level(RS_OPT_LEVEL_FULL);
    return;
  }

  if (SPECIAL_CASE_HEADER_FOR_CD) {

    BB_NODE * header = loop->Header();
    BB_NODE_SET * bb_ctrl = bb->Rcfg_dom_frontier();
    
    if ((header == NULL) || (bb_ctrl == NULL)) {
      Set_opt_level(RS_OPT_LEVEL_NONE);
      return;
    }

    if (bb_ctrl->MemberP(header) && 
        (bb_ctrl->Size() == (BS_ELT) 1)) {
      Set_opt_level(RS_OPT_LEVEL_RESTRICTED);
      return;
    }
  }
  if (Trace())
    fprintf(TFile, 
            "\nSkip completely. block:(%d), header:(%d), body:(%d)\n",
            bb->Id(), Loop()->Header()->Id(), Loop()->Body()->Id());
  
  Set_opt_level(RS_OPT_LEVEL_NONE);
  return;
}

//==========================================================
// Description:
// 
//   Physically Distribute cr over one child. Example:
//
//   Before ==:
//   
//         CHILD1
//         CHILD2
//       ADD
//       CHILD3
//     MUL(cr)
//   (parent)
//
//   After ==:
//
//         CHILD1
//         CHILD3
//       MUL
//         CHILD2
//         CHILD3
//       MUL
//     ADD (tree)
//   parent
//
// Parameter:
//
//   child: distribute over which child (0/1). Represents
//          the add operator in Before in example
//   others: see description above
//
// Returns:
//
//    coderep * representing the new coderep equivalent to
//    the passed in coderep cr
//
//==========================================================

CODEREP* RESHAPER::Distribute_node(CODEREP *cr, CODEREP *parent,
                                  INT32 child)
{
  if (Trace()) {
    fprintf(TFile, "\nDistribute node before cr%d\n", cr->Coderep_id());
    cr->Print(1, TFile);
  }

  // a * (b+c) to a*b + a*c
  INT32 other_child = !child;
  CODEREP *plus = cr->Opnd(child);
  CODEREP *a = cr->Opnd(other_child);
  CODEREP *b = plus->Opnd(0);
  CODEREP *c = plus->Opnd(1);
  
  CODEREP *mul1 = Add_bin_node(cr->Op(), a, b);
  CODEREP *mul2 = Add_bin_node(cr->Op(), a, c);
  CODEREP *tree = Add_bin_node(plus->Op(), mul1, mul2);

  // re-rank the new codereps using ranks of children
  Ranker()->Set_rank(mul1, Get_rank(mul1));
  Ranker()->Set_rank(mul2, Get_rank(mul2));
  Ranker()->Set_rank(tree, Get_rank(tree));

  Delete(plus);

  Replace_child(parent, cr, tree, FALSE);
  if (Trace()) {
    fprintf(TFile, "\nDistribute node after- cr%d\n", tree->Coderep_id());
    tree->Print(1, TFile);
  }

  return tree;
}

//==========================================================
// Description:
// 
//   Physically Reassociate to have outer child have highest
//   rank
//
//   Before ==:
//   
//         CHILD1 (max rank amongs 1,2,3)
//         CHILD2
//       MUL
//       CHILD3
//     MUL(cr)
//   (parent)
//
//   After ==:
//
//         CHILD3
//         CHILD2
//       MUL
//       CHILD1 (max rank)
//     MUL(cr)
//   (parent)
//
// Parameter:
//
//   child: distribute over which child (0/1). Represents
//          the add operator in Before in example
//   others: see description above
//
// Returns:
//
//    coderep * representing the new coderep equivalent to
//    the passed in coderep cr
//
//==========================================================

CODEREP* RESHAPER::Reassociate_distributive_node(CODEREP *cr,
                                                 CODEREP *parent,
                                                 INT32 child)
{
  // example:
  // Input:  a*(b*c) OR [a <cr> (b <mul> c)]
  // Output:  b*(a*c) or c*(a*b)
  if (Trace()) {
    fprintf(TFile, "\nReassociate Distribute node before cr%d\n", 
            cr->Coderep_id());
    cr->Print(1, TFile);
  }

  INT32 other_child = !child;
  CODEREP *mul = cr->Opnd(child);
  CODEREP *a = cr->Opnd(other_child);
  CODEREP *b = mul->Opnd(0);
  CODEREP *c = mul->Opnd(1);
  
  RANK rank_a = Ranker()->Get_rank(a);
  RANK rank_b = Ranker()->Get_rank(b);
  RANK rank_c = Ranker()->Get_rank(c);

  // rank_a should be the biggest in the end
  CODEREP *new_cr;

  if (rank_b > rank_c) {
    
    FmtAssert(rank_b > rank_a,
              ("Reassociate_distributive not respecting ranks")); 

    // replace b and a
    CODEREP *new_mul = Add_bin_node(mul->Op(), a, c);
    new_cr = Add_bin_node(cr->Op(), new_mul, b);

    Delete(mul);
    Replace_child(parent, cr, new_cr, FALSE);

  } else {

    FmtAssert(rank_c > rank_a,
              ("Reassociate_distributive not respecting ranks")); 

    // replace c and a
    CODEREP *new_mul = Add_bin_node(mul->Op(), a, b);
    new_cr = Add_bin_node(cr->Op(), new_mul, c);

    Delete(mul);
    Replace_child(parent, cr, new_cr, FALSE);
  }
  if (Trace()) {
    fprintf(TFile, "\nReassociate Distribute node after cr%d\n", new_cr->Coderep_id());
    new_cr->Print(1, TFile);
  }

  return new_cr;
}

//==========================================================
// Description:
// 
//  Determine if it is profitable to distribute 
//  <param: other> over <param: add>
//
//==========================================================

BOOL RESHAPER::Should_distribute(CODEREP *other, CODEREP *add)
{
  BOOL should_distribute = TRUE;
  RANK dist_rank = Ranker()->Get_rank(other);
  RANK add_rank = Ranker()->Get_rank(add);
  RANK c_rank_1 = Ranker()->Get_rank(add->Opnd(0));
  RANK c_rank_2 = Ranker()->Get_rank(add->Opnd(1));

  INT32 invar_1 = Invariant_loop_number(add->Opnd(0));
  INT32 invar_2 = Invariant_loop_number(add->Opnd(1));

  // sanity - add and dist should be ranked
  should_distribute &= (add_rank != -1 && dist_rank != -1);

  // distribute lower ranked over higher ranked
  should_distribute &= (dist_rank <= add_rank);

  // atleast one should be hoistable from current loop
  should_distribute &= (invar_1 > 0 || invar_2 > 0);

  // both should be hoistable out of different loops
  // otherwise we are just increasing the muls
  should_distribute &= (invar_1 != invar_2); 

  // avoid distribute if other is a CK_OP except it has a lower rank
  should_distribute &= (other->Kind() != CK_OP || dist_rank < add_rank);

  // avoid distribute over multi-level add trees
  should_distribute &= (Ranker()->Get_level(add) <=2);

  // Do tracing
  if (Trace() && !should_distribute) {
    fprintf(TFile, "\nRejected distribution over add cr%d\n", 
            add->Coderep_id());
    fprintf(TFile, "\nadd_rank:%d -- dist_rank:%d -- invar_1:%d -- invar_2:%d\n",
            add_rank, dist_rank, invar_1, invar_2);  
  }

  return should_distribute;
}

//==========================================================
// Description:
// 
//  Determine if it is profitable to reassociate multiple
//  multiplies
//
//==========================================================

BOOL RESHAPER::Should_reassociate_distributive(CODEREP *cr, INT32 child)
{
  FmtAssert( ((child == 1) || (child == 0)),
             ("Wrong child for should distribute")); 

  INT32 other_child = !child;
  
  CODEREP * child1 = cr->Opnd(child);
  CODEREP * child2 = cr->Opnd(other_child);

  // ensure the same operator
  if (child1->Opr() == cr->Opr()) {
    RANK far_child1_rank = Ranker()->Get_rank(child1->Opnd(0));
    RANK far_child2_rank = Ranker()->Get_rank(child1->Opnd(1));
    RANK child2_rank = Ranker()->Get_rank(child2);

    // if child2_rank is not the max
    if ((far_child1_rank > child2_rank) ||
        (far_child2_rank > child2_rank)) {
      return TRUE;
    }
  }
  return FALSE;
}

//==========================================================
// Description:
// 
//   Recursively go down the tree and distribute multiplies
//   over adds where deemed profitable
//
// Remark:
//
//  Notice that return of new coderep from Distribute_node
//  and calling of distribution correctly ensures that
//  (a+b)*(c+d) is distributed to 4 terms in single pass
//
//==========================================================

CODEREP* RESHAPER::Distribute_tree(CODEREP *cr, CODEREP *parent,
                                   BOOL *changed)
{
  if (Ranker()->Is_distribute_op(cr)) {
    
    FmtAssert(cr->Kid_count() == 2,
              ("Wrong num_kids for distribute"));  
    INT32 additive_kid = -1;
    
    // See if operand0 is a suitable additive kid
    if (Ranker()->Is_additive_op(cr->Opnd(0)))
      if (Should_distribute(cr->Opnd(1), cr->Opnd(0)))
          additive_kid = 0;
    
    // if operand0 was not suitable check operand1
    if ((additive_kid == -1) && Ranker()->Is_additive_op(cr->Opnd(1)))
      if (Should_distribute(cr->Opnd(0), cr->Opnd(1)))
        additive_kid = 1;

    if (Trace() && additive_kid == -1)
      fprintf(TFile, "\nNo additive kid, skip cr%d", cr->Coderep_id());

    // if we found a suitable distribution, physically do it
    if (additive_kid != -1) {
      cr = Distribute_node(cr, parent, additive_kid);
      *changed = TRUE; // mark changed implies retry

    } else if (REASSOCIATE_MULTIPLIES &&
               (Opt_level() == RS_OPT_LEVEL_FULL)) {

      // There are no additive children, check if there are
      // any identical distributive ops
      // may be and opportunity to reassociate a*b*c
      INT32 reassociate_kid = -1;

      if (Ranker()->Is_distribute_op(cr->Opnd(0)) &&
          Should_reassociate_distributive(cr, 0)) {

            reassociate_kid = 0;

      } else if (Ranker()->Is_distribute_op(cr->Opnd(1)) &&
                 Should_reassociate_distributive(cr, 1)) {

            reassociate_kid = 1;
      }
      if (reassociate_kid != -1) {
        cr = Reassociate_distributive_node(cr, parent, reassociate_kid);
        *changed = 1;
      }
    } // else if !skip
  } // end if (Is_distribute_op(cr)) 
  

  // Go down the tree to see if there are multiplies in the 
  // sub trees to be distributed

  if ((cr->Kind() == CK_IVAR)&&
      (cr->Opr() == OPR_ILOAD)) {
    CODEREP * base = cr->Ilod_base();
    if (base == NULL)
      base = cr->Istr_base();
    if (base != NULL)
      Distribute_tree(base, cr, changed);
  }
    
  if ((cr->Kind() != CK_OP) )
    return cr;

  // distribute any children of the tree
  for (INT32 i = 0; i < cr->Kid_count(); i++) {
    Distribute_tree(cr->Opnd(i), cr, changed);
  }
  return cr;
}

//==========================================================
// Description:
// 
//  Wrapper to set appropriate state before
//  re-ranking an expression
//
//==========================================================

void RESHAPER::Update_rank(CODEREP *cr)
{
  // Set the per basic block state before calling
  // re-rank
  Ranker()->Set_bb(Bb());
  Ranker()->Set_loop(Loop());
  Ranker()->Set_bb_rpo_id(Rpo_id());
  
  Ranker()->Rank_expression(cr, TRUE);
}
   
//==========================================================
// Description:
// 
//   Get rank from ranker or compute it from one level down
//   Primarily exists to not duplicate code when creating
//   new operators over ranked children
//
//==========================================================

RANK RESHAPER::Get_rank(CODEREP *cr)
{
  RANK rank = Ranker()->Get_rank(cr);

  if ((cr->Kind() != CK_OP) 
      || (rank != -1))
    return rank;

  rank = 0;
  for (INT32 i = 0; i < cr->Kid_count(); i++) 
    rank = max(rank, Get_rank(cr->Opnd(i)));
  
  return rank;
}

//==========================================================
// Description:
// 
//   Build a flattened representation of the tree under
//   the current coderep recursively in the list
//
// Parameters:
//
//   sign - sign of the current term in final tree
//          second term in sub has its sign flipped
//
//==========================================================

void RESHAPER::Build_flatten(CODEREP *cr, LLIST* list,
                            BOOL sign)
{
  if (cr->Kind() != CK_OP) {
    Add_to_list(cr, sign, Get_rank(cr), list);
    return;
  }
  
  switch (cr->Opr()) {

  case OPR_SUB:
    Build_flatten(cr->Opnd(0), list, sign);
    Build_flatten(cr->Opnd(1), list, !sign);
    break;

  case OPR_ADD:
    Build_flatten(cr->Opnd(0), list, sign);
    Build_flatten(cr->Opnd(1), list, sign);
    break;

  default:
    Add_to_list(cr, sign, Get_rank(cr), list);
    break;
  }
  return;
}

//==========================================================
// Description:
// 
//   Build a tree from the right list. After the function
//   the head of tree is the only elment in the list
//
// Parameters:
//
//   iv - which head to start from in the list
//
//==========================================================

void RESHAPER::Reduce_list(LLIST* list, BOOL iv)
{
  while (1) {
    LLIST_NODE * one = list->Head(iv);
    
    // empty list
    if (one == NULL)
      return;
    
    LLIST_NODE * two = list->Next(one);
    
    // one element left in list, reduction completed
    if (two == NULL)
      return;
    
    BOOL one_neg = one->sign;
    BOOL two_neg = two->sign;

    // even if one is unsigned, we go unsigned
    BOOL op_sign = one->op_sign && two->op_sign;

    if (one_neg == two_neg) {
      
      CODEREP *add = Add_bin_node(Add_op(op_sign), one->cr, two->cr);

      list->Pop(iv); // leak the returned value
      two->cr = add;
      two->op_sign = op_sign;
      // the sign of two is the correct one,
      // no update needed
    } else {
      // the positive one is first
      CODEREP *first = (!one_neg)? one->cr: two->cr;
      CODEREP *second = (one_neg)? one->cr: two->cr;
      CODEREP *sub = 
        Codemap()->Add_bin_node(Sub_op(op_sign), first, second);
      
      list->Pop(iv); // leak the returned value
      two->cr = sub;
      two->sign = FALSE;
      two->op_sign = op_sign;
    }
  }
}

//==========================================================
// Description:
// 
//   Rebuild a tree from flattened representation
//
// Remarks:
//
//   Various reassociations can be tried here
//   
//   REASSOCIATION_OPTION == 1
//   Rebuild trees of iv and other list separately
//   Combine them here
//
//   REASSOCIATION_OPTION == 2
//   Merge the two lists with non-iv list coming first
//   Combine them as if they were one list
//
//==========================================================

CODEREP* RESHAPER::Rebuild_tree(CODEREP *cr, CODEREP *parent,
                                LLIST* list)
{
  BOOL iv = TRUE;

  if (REASSOCIATION_OPTION == 1) {

    // Reduce iv and non-iv lists to one member each
    Reduce_list(list, !iv);
    Reduce_list(list, iv);

    CODEREP *new_cr = NULL;
  
    // Now create on cr for those members
    LLIST_NODE * one = list->Head(iv);
    LLIST_NODE * two = list->Head(!iv);

    if (one == NULL) {
      new_cr = two->cr;

    } else if (two == NULL) {
      new_cr = one->cr;

    } else {
    
      BOOL one_neg = one->sign;
      BOOL two_neg = two->sign;

      // even if one is unsigned, we go unsigned
      BOOL op_sign = one->op_sign && two->op_sign;

      if (one_neg == two_neg) {
        FmtAssert(!one_neg, ("Both can't be negative"));
        new_cr = Add_bin_node(Add_op(op_sign), one->cr, two->cr);
      } else {
        CODEREP *first = one_neg? one->cr: two->cr;
        CODEREP *second = (!one_neg)? one->cr: two->cr;
        new_cr = Add_bin_node(Sub_op(op_sign), first, second);
      }
    }
    if (cr != new_cr)
      Replace_child(parent, cr, new_cr, TRUE);
    return new_cr;

  } else if (REASSOCIATION_OPTION == 2) {

    LLIST_NODE * node = list->Head(!iv);

    // Find the end of first list and
    // append the head of second

    if (node != NULL) {
      while (node->next != NULL) {
        node = list->Next(node);
      }
      node->next = list->Head(iv);

    } else {
      list->Set_head(list->Head(iv), !iv);
    }
    Reduce_list(list, !iv);
    CODEREP *new_cr = list->Head(!iv)->cr;
    if (cr != new_cr)

      Replace_child(parent, cr, new_cr, TRUE);
    return new_cr;
  }
  return NULL;
}

//==========================================================
// Description:
// 
//  Trace dumper
//
//==========================================================

void RESHAPER::Print_flattened_list(LLIST *list)
{
  fprintf(TFile,
          "\n\nFlattened representation======\n");
  for (INT32 i = 0; i < 2; i++) {
    BOOL iv = (BOOL) i;
    if (iv)
      fprintf(TFile, "\n\nIV-list======\n");
    else
      fprintf(TFile, "\n\nNon-IV-list======\n");
      
    LLIST_NODE *node = list->Head(iv);
    int num = 0;
    while(node != NULL) {
      fprintf(TFile, "\nNODE (%d)---", num);    
      fprintf(TFile, "\nRank %d", node->rank);
      fprintf(TFile, "\nIsIv %d", node->has_iv);   
      fprintf(TFile, "\nSign %d", node->sign);
      fprintf(TFile, "\nOpSign %d\n", node->op_sign);
      node->cr->Print(1, TFile);
      node = list->Next(node);
      num++;
    }
  }
  fprintf(TFile, "\n=============================\n");
}

//==========================================================
// Description:
// 
//  Set the opcodes to be used for signed/unsigned
//  operations based on size of coderep
//
//==========================================================

void RESHAPER::Set_additive_opcodes(CODEREP *cr)
{
  // Set the operators to be used for add/sub
  INT32 size = MTYPE_byte_size(OPCODE_rtype(cr->Op()));
  BOOL op_sign = TRUE;
      
  if (size == 4) {
    Set_Add_Opcode(OPC_I4ADD, op_sign);
    Set_Sub_Opcode(OPC_I4SUB, op_sign);
    Set_Add_Opcode(OPC_U4ADD, !op_sign);
    Set_Sub_Opcode(OPC_U4SUB, !op_sign);
        
  } else if (size == 8) {
    Set_Add_Opcode(OPC_I8ADD, op_sign);
    Set_Sub_Opcode(OPC_I8SUB, op_sign);
    Set_Add_Opcode(OPC_U8ADD, !op_sign);
    Set_Sub_Opcode(OPC_U8SUB, !op_sign);
        
  } else {
    FmtAssert(FALSE, 
              ("Type not correct: Disribute_tree"));
  }
}

//==========================================================
// Description:
// 
//  Physically reassociate this add/sub and all add/sub
//  operations that are linked by parent child relation
//
//==========================================================

CODEREP* RESHAPER::Reassociate_additive_node(CODEREP *cr,
                                CODEREP *parent)
{
  if (Trace()) {
    fprintf(TFile,
            "\n\nBefore Reassociation ======\n");
    cr->Print(1, TFile);
  }

  CODEREP *child_1 = cr->Opnd(0);
  CODEREP *child_2 = cr->Opnd(1);

  // There should be atleast 2 level of adds/subs
  if (!(
        ((child_1->Kind() == CK_OP) &&
         Ranker()->Is_additive_op(child_1))
        || 
        ((child_2->Kind() == CK_OP) &&
         Ranker()->Is_additive_op(child_2)) 
        )
      ) {
    if (Trace())
      fprintf(TFile,
              "\n\nNothing to be done ======\n");
    return cr;
  }
    
  // Make a flattened list representation
  LLIST *list = CXX_NEW(LLIST(Pool()), Pool());
  Build_flatten(cr, list, FALSE);

  if (Trace())
    Print_flattened_list(list);
  
  // Rebuild the tree from this representation
  cr = Rebuild_tree(cr, parent, list);

  if (Trace()) {
    fprintf(TFile, 
            "\nAfter Reassociation===================\n");
    cr->Print(1, TFile);
  }
  return cr;
}

//==========================================================
// Description:
// 
//  Recursively go down the tree and call reassociation of
//  additives.
//  We only start reassociation when the add op's parent is
//  not add to prevent multiple iterations on same nodes
//
// Parameter
//  ignore_parent is only true for top level call
//
//==========================================================

CODEREP* RESHAPER::Reassociate_additives(CODEREP *cr,
                                         CODEREP *parent,
                                         BOOL ignore_parent)
{
  // Check this node
  if (Ranker()->Is_additive_op(cr)) {
    if (ignore_parent ||
        ((parent != NULL) 
         && !Ranker()->Is_additive_op(parent))) {
      
      // Set the opodes to be used for add/sub
      Set_additive_opcodes(cr);
      
      // Perform the actual reassociation
      cr = Reassociate_additive_node(cr, parent);
    }
  }

  if (DO_ONE_LEVEL_ADD)
    return cr;

  // Go down the tree to see if there are multiplies in the 
  // sub trees to be distributed
  if ((cr->Kind() == CK_IVAR)&&
      (cr->Opr() == OPR_ILOAD)) {
    CODEREP * base = cr->Ilod_base();
    if (base == NULL)
      base = cr->Istr_base();
    if (base != NULL)
      Reassociate_additives(base, cr, FALSE);
  }
  
  if ((cr->Kind() != CK_OP))
    return cr;
  
  // Check the children
  for (INT32 i = 0; i < cr->Kid_count(); i++) {
    Reassociate_additives(cr->Opnd(i), cr, FALSE);
  }

  return cr;
}

//==========================================================
// Description:
// 
//   Once a tree has been chosen, reshape it
//
//   Right now it does the following:
//
//   1) distribution (keeps tree valid at all times)
//
//   2) builds a flattened representation of additive
//      parts of the tree in a side list (original tree 
//      intact)
//      This representation is sorted as it is built
//   
//   3) rebuilds a new tree and replaces the parallel old 
//      tree
//
//==========================================================

void RESHAPER::Reshape_tree(CODEREP *cr, CODEREP *parent)
{
  if (Trace()) {
    fprintf(TFile, "\nPerforming Reshaping on tree cr%d",
            cr->Coderep_id());
    fprintf(TFile, 
            "\nBefore reshaping =====================\n");
    cr->Print(1, TFile);
  }

  // handle without correct parent later
  if (!((parent == NULL)
       || (parent->Kind() == CK_OP) 
       || (parent->Opr() == OPR_ILOAD))) {
    if (Trace())
      fprintf(TFile, "\nparent - not handled");
    return;
  }

  // Recursively Distribute multiplies on additive operands
  // Reassociate multiple multiplies

  BOOL distribute_changes = FALSE;

  if (Loop() == NULL) {
    if (Trace())
      fprintf(TFile, 
              "\nSkipping distribute, no loop here\n");
    
  } else {
    BOOL changed;
    do {
      changed = FALSE;
      cr = Distribute_tree(cr, parent, &changed);

      if (changed) {
        distribute_changes = TRUE;
        if (parent != NULL) {
          Codemap()->Rehash_tree(parent, FALSE, &changed, NULL);
        } else {
          Codemap()->Rehash_tree(cr, FALSE, &changed, NULL);
        }
        Update_rank(cr);
      }
    } while(changed);

    if (Trace()) {
      fprintf(TFile,
              "\nAfter distribution =====================\n");
      cr->Print(1, TFile);
    }
  }

  // Recursive reassociate adds/subs
  //  if (distribute_changes || (Opt_level() == 2))
  cr = Reassociate_additives(cr, parent, TRUE);

  BOOL changed = FALSE;

  if (parent != NULL) {
    Codemap()->Canon_rhs(parent);
    Codemap()->Rehash_tree(parent, FALSE, &changed, NULL);
  } else {
    Codemap()->Canon_rhs(cr);
    Codemap()->Rehash_tree(cr, FALSE, &changed, NULL);
  }

  if (changed && Trace()) {
    fprintf(TFile, 
            "\nAfter final tree rehashing =====================\n");
    cr->Print(1, TFile);
  }
 return;
}

//==========================================================
// Description:
// 
//   Consider if the current tree is a reshaping candidate
//   reshape the marked candiate
//
//   For efficiency reasons, RANKER while it is ranking
//   also marks the suitability of the tree. This merely
//   checks what is marked
//
//==========================================================

void RESHAPER::Consider_expression(CODEREP *cr, CODEREP *parent)
{
  INT32 level = Ranker()->Get_level(cr);

  // level 1 is already handled elsewhere
  if (level < 2)
    return;

  // if marked visited, skip
  if ((cr->Kind() == CK_OP) &&
      cr->Is_isop_flag_set(ISOP_EXPR_RESHAPE_VISITED))
    return;

  // reassociate max reassociable tree of limited size
  if ((Ranker()->Get_reassociate(cr) == TRUE)
      && (level < Max_level())
      && (Ranker()->Get_has_iv(cr) == TRUE)
    ) {
    Reshape_tree(cr, parent);
    // cannot mark visited here, cr will change
    // parent and child might be marked, so it
    // should be fine
    return;
  }
  
  if (cr->Opr() == OPR_ILOAD) {
    CODEREP * base = cr->Ilod_base() ?
      cr->Ilod_base() : cr->Istr_base();
    if (base != NULL)
      Consider_expression(base, cr);

    // base might be a different coderep now
    // mark the new coderep for base
    base = cr->Ilod_base() ?
      cr->Ilod_base() : cr->Istr_base();

    if (base != NULL && base->Kind() == CK_OP)
      base->Set_isop_flag(ISOP_EXPR_RESHAPE_VISITED);
  }
  
  if (cr->Kind() != CK_OP)
    return;
  
  for (INT32 i = 0; i < cr->Kid_count(); i++) {
    Consider_expression(cr->Opnd(i), cr); 
  }
  cr->Set_isop_flag(ISOP_EXPR_RESHAPE_VISITED);
  return;
}

//==========================================================
// Description:
// 
//   Consider this stament for reshaping
//
//==========================================================

void RESHAPER::Consider_statement(STMTREP *stmt)
{
  CODEREP      *rhs = stmt->Rhs();
  CODEREP      *lhs = stmt->Lhs();

  if (stmt->Black_box())
    return;

  INT32 changed;
  switch (stmt->Opr()) {

  case OPR_STID:
    Consider_expression(rhs, NULL);
    Codemap()->Rehash_tree(rhs, FALSE, &changed, NULL);
    break;

  case OPR_ISTORE:
    Is_True(lhs->Istr_base() != NULL, ("Unexpected Istr_base for OPR_ISTORE: RESHAPER"));
    Consider_expression(lhs->Istr_base(), NULL);
    Codemap()->Rehash_tree(lhs->Istr_base(), FALSE, &changed, NULL);
    Consider_expression(rhs, NULL);
    Codemap()->Rehash_tree(rhs, FALSE, &changed, NULL);
    break;

  default:
    // Be conservative for now, improve later
    if (lhs != NULL){
      Consider_expression(lhs, NULL);
      Codemap()->Rehash_tree(lhs, FALSE, &changed, NULL);
    }
    if (rhs != NULL){
      Consider_expression(rhs, NULL);
      Codemap()->Rehash_tree(rhs, FALSE, &changed, NULL);
    }
    break;
  }
}

//==========================================================
// Description:
// 
//   Consider all expressions in this program for reshaping
//
//   Top level call tree:
//
//   Perform_expression_reshaping()
//   -- Rank_Function() - Rank variables/expressions
//                      - mark codereps reassociable
//                      - collect level info to limit
//                        compile time
//
//  -- Foreach_statement in function
//     -- Consider_statement()
//        -- Consider_expression() <recursive>
//           -- Reshape_tree()
//              -- distribute_tree() <recursive>
//              -- build_flatten()
//              -- Rebuild_tree()
//
//==========================================================

void RESHAPER::Perform_expression_reshaping()
{
  if (Trace()) {
    fprintf(TFile, "\n=======================================\n");
    fprintf(TFile, "| BEFORE EXPRESSION RESHAPING");
    fprintf(TFile, "\n=======================================\n");
    Cfg()->Print(TFile);
  }

  // ensure we have valid loop information

  if (!Cfg()->Loops_valid())
    Cfg()->Analyze_loops();

  // First run the rank algorithm 

  _ranker = (RANKER*) CXX_NEW(RANKER(Pool(), Cfg(), Codemap(),
                                     Trace()), Pool());
  _ranker->Rank_function();
  
  if (Trace()) {
    fprintf(TFile, "\nSTART RESHAPING============\n");
  }

  if (Trace()) {
    fprintf(TFile, "\nControl dependence graph==\n");
    Cfg()->PrintCDVis(TFile);
  }

  // Consider all the statements for reshaping

  BB_NODE    *bb;
  INT32       bb_rpo_id = 1;
  RPOBB_ITER  cfg_iter(Cfg());

  Set_opt_level(RS_OPT_LEVEL_NONE);

  FOR_ALL_ELEM(bb, cfg_iter, Init()) {
    
    if (Trace())
      fprintf(TFile, 
              "\nCONSIDER BASIC BLOCK: ID(%d): ==========\n",
              bb->Id());

    // Only consider blocks in loops
    if (bb->Innermost() != NULL) {

      // Set per basic block state
      Set_bb(bb);
      Set_rpo_id(bb_rpo_id);
      Set_loop(bb->Innermost());
      
      Evaluate_optimization_level(bb);
      
      if (Opt_level() > RS_OPT_LEVEL_NONE) {
        
        STMTREP     *stmt;
        STMTREP_ITER stmt_iter(bb->Stmtlist());
        
        FOR_ALL_NODE(stmt, stmt_iter, Init()) {
          Set_current_stmt(stmt);
          Consider_statement(stmt);
        }
      }
    } // if bb->innermost
    bb_rpo_id++;
  }
  
  if (Trace()) {
    fprintf(TFile, "\n=======================================\n");
    fprintf(TFile, "| AFTER EXPRESSION RESHAPING");
    fprintf(TFile, "\n=======================================\n");
    Cfg()->Print(TFile);
  }
}

//==========================================================
// Description:
// 
//   Reshape expression tree primarily for better code
//   motion and strength reduction
//
//==========================================================

void
COMP_UNIT::Do_expression_reshaping(void)
{
  MEM_POOL reshape_local_pool;

  OPT_POOL_Initialize(&reshape_local_pool, "reshape local pool", FALSE, -1);
  OPT_POOL_Push(&reshape_local_pool, -1);
  {
    RESHAPER reshaper(&reshape_local_pool, Cfg(), Opt_stab(), Htable());
    reshaper.Perform_expression_reshaping();
  } // the reshaper destructor is called here
  OPT_POOL_Pop(&reshape_local_pool, -1);
  OPT_POOL_Delete(&reshape_local_pool, -1);
}
