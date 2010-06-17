/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//-*-c++-*-

/*
 * Copyright 2003, 2004, 2005, 2006 PathScale, Inc.  All Rights Reserved.
 */

// ====================================================================
// ====================================================================
//
// Module: opt_estr.cxx
// $Revision: 1.1.1.1 $
// $Date: 2005/10/21 19:00:00 $
// $Author: marcel $
// $Source: /proj/osprey/CVS/open64/osprey1.0/be/opt/opt_estr.cxx,v $
//
// ====================================================================
//
// Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of version 2 of the GNU General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it would be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// Further, this software is distributed without any warranty that it
// is free of the rightful claim of any third person regarding
// infringement  or the like.  Any license provided herein, whether
// implied or otherwise, applies only to this software file.  Patent
// licenses, if any, provided herein do not apply to combinations of
// this program with other software, or any other product whatsoever.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write the Free Software Foundation,
// Inc., 59 Temple Place - Suite 330, Boston MA 02111-1307, USA.
//
// Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
// Mountain View, CA 94043, or:
//
// http://www.sgi.com
//
// For further information regarding this notice, see:
//
// http://oss.sgi.com/projects/GenInfo/NoticeExplan
//
// ====================================================================
// ====================================================================


#ifdef USE_PCH
#include "opt_pch.h"
#endif // USE_PCH
#pragma hdrstop


#define opt_estr_CXX	"opt_estr.cxx"
#ifdef _KEEP_RCS_ID
static char *rcs_id = opt_estr_CXX"$Revision: 1.12 $";
#endif /* _KEEP_RCS_ID */

#include "defs.h"
#include "config.h"
#include "cxx_memory.h"

#include "opt_base.h"
#include "opt_config.h"
#include "opt_bb.h"
#include "opt_cfg.h"
#include "opt_htable.h"
#include "opt_etable.h"
#include "opt_estr.h"
#include "opt_ssa.h"
#include "opt_wn.h"
#include "tracing.h"
#include "opt_cvtl_rule.h"


/* CVTL-RELATED start */

// Restriction for strength reduction of CVT.
//
//   There are two situations a CVT might be involved in strength reduction:
//    1.  in the strength-reduced expr, e.g.  (CVT i) * 4
//    2.  in the iv update statement, e.g.  i = (CVT i) + 1
//
//   After discussion with Shin and Peng and me (Raymond), we decided not to
//   handle iv update of the form i = (CVT i) + 1.  The only CVT we are interested
//   is U8U4CVT (I8I4CVT is a nop CVT and wont appear in the iv update statement).
//   The decision is to canonicalize i to U4U4LDID all the time in the htable.
//
//   There are several occurrences of CVTs in strength-reduced exprs:
//    1.  explict CVT, e.g.  U8U4CVT
//    2.  the nop CVT (those CVTs are nops and therefore not represented in the IR)
//        between the operator type and operand type.
//    3.  implicit CVT embedded in LDIDs, e.g., U8U4LDID
//         a U8U4LDID should be equivalent to the expr U8U4CVT U4U4LDID.
//         This has been totally ignored in v7.0 and v7.1 compilers, although
//         ucode has done these correctly.
//
//   A CVT is not strength reduced for one of the following reasons:
//    1.  The result type size is smaller than I4 or U4. 
//          To strength reduce such CVT, a truncation would be needed at
//          each injury update.
//    2.  The result type size is larger than the data size and data type allows overflow.
//          Consider the case when the iv is I4 and the strength-reduced expr is I8.
//          If the iv overflows to -ve number (e.g., a I4 wraps from +ve to -ve), 
//          iv * 4 should be negative, but the strength reduced value is still +ve.
//        !!!!!!!
//          However, since U8I4CVT is too important not to optimize in -64 compilation,
//          we decide to produce occasionally wrong result!  The production of correct
//          result is guaranteed only if -OPT:wrap_around_unsafe_opt=off.
//        !!!!!!!
//
//   These CVT are always safe to strength reduce:
//     1.  The result size is smaller than the data type and the
//         result size is 4.   The U4/I4 operations include the
//         implicit truncation.
//
//   These CVT are *assumed* to be safe to strength reduce:
//     1.  The CVT that Need_type_conversion() says NOT_AT_ALL for 
//         the performance reason mentioned above.
//
//   U8U4CVT is theoretically safe to strength reduce because a unsigned is
//   not supposed to wrap around (ANSI say so). Practically strength reduction is wrong
//   because the compiler might introduced a wrap around unsigned value. e.g.
//     while (1) {
//        i = i + 1
//        (i-1) u8* 4;
//     }
//   The initial value of (i-1) at the loop entry is -1 (not representable 
//   with unsigned).  We used to disable str-red of U8U4CVT in v7.1, so
//   performance of loops with unsigned IV compiled -64 suffers.
//
//   We noticed that the above situation only happen with the strength
//   reduction of (CVT (OP ....)), but it is OK with CVT LDID.
//   Therefore, the solution to get U8U4CVT strength reduced and produce correct result:
//     1.  Do not strength reduce (CVT (OP ...)
//     2.  Do strength reduce (CVT LDID).
//           Since (OP ...) eventually becomes a (LDID temp_preg) in EPRE,
//           we need to add a bit in the aux_stab to distinguish LDID of
//           original variables from EPRE pregs.
//     3.  Canonicalize expression so that CVT are pushed down to the leafs.
//         e.g.     
//              4 
//                U4LDID
//                1
//               U4SUB
//              U8U4CVT
//             U8MPY
//
//          to
//
//              4 
//                U4LDID
//               U8U4CVT
//               1
//              U8SUB
//             U8MPY
//       This canonicalization is correct because any computation carried out in U4
//       without overflow can be computed identically in U8.  Although ANSI C
//       says overflow in unsigned is undefined, we would still turn this opt off
//       using -OPT:wrap_around_unsafe_opt.
//


//======================================================================
// Determine if the CVT is a linear function, and the value being
// converted can thus be used as an iv.
// N.B. cr can be either a CVT or an LDID.
//======================================================================

BOOL
STR_RED::Is_cvt_linear( const CODEREP *cr ) const
{
  // screen out non-register size types
  if (MTYPE_size_min(cr->Dsctyp()) < MTYPE_size_min(MTYPE_I4)) {
    if (WOPT_Enable_STR_Short && (MTYPE_size_min(cr->Dsctyp()) == MTYPE_size_min(MTYPE_I2))) {
      return TRUE;
    }
    return FALSE;
  }

  // e.g., disable str-red of I8I4CVT ... if do not allow wrap around opt
  if (! Allow_wrap_around_opt && 
      MTYPE_size_min(cr->Dsctyp()) != MTYPE_size_min(cr->Dtyp()))
    return FALSE;

  // For bug compatible with v7.1 and performance reason,
  // assume these CVTs to be linear.
  //if (Need_type_conversion(cr->Dsctyp(),cr->Dtyp(),NULL)==NOT_AT_ALL)
  // return TRUE;
  // Fix 761037:
  // Do not use Need_type_conversion, check allow I8I4CVT and U8I4CVT ... directly
  // because Need_type_conversion is target dependent.
  //
  if (cr->Dtyp() == cr->Dsctyp())
    return TRUE;

  if ((cr->Dtyp() == MTYPE_U8 || cr->Dtyp() == MTYPE_I8) &&
      cr->Dsctyp() == MTYPE_I4)
    return TRUE;

  // all truncations are linear wrt to the result type.
  if (MTYPE_size_min(cr->Dtyp()) < MTYPE_size_min(cr->Dsctyp()))
    return TRUE;

  // allow U8U4CVT strength reduction if the CVT is a 1st order expr
  if ((cr->Dtyp() == MTYPE_U8 || cr->Dtyp() == MTYPE_I8)
      && cr->Dsctyp() == MTYPE_U4) {
#ifndef TARG_X8664
    Is_True(cr->Opnd(0)->Kind() == CK_VAR, 
	    ("STR_RED::Is_cvt_linear:  invalid str red expr."));
#ifdef TARG_NVISA
    // Tried to comment out this check, 
    // cause wanted to optimize U8U4CVT to help 64bit perf, 
    // and it worked on Tesla, but not when no longer truncate 64bit addresses.
    if (Htable()->Opt_stab()->Aux_stab_entry(cr->Opnd(0)->Aux_id())->EPRE_temp())
      DevWarn("will not strength reduce this U8U4CVT");
#endif // NVISA
    if (!Htable()->Opt_stab()->Aux_stab_entry(cr->Opnd(0)->Aux_id())->EPRE_temp())
#endif
      return TRUE;
  }
      
  return FALSE;
}


BOOL 
STR_RED::Is_implicit_cvt_linear(MTYPE opc_type, MTYPE opnd_type) const
{
  // The check below is commented because 
  //     U4U4LDID
  //    U8MPY  
  // is a legal sequence generated by the frontend and allowed in
  // the htable as well.  Its meaning is the composite of U8I4CVT I4U4CVT.
  // Both CVTs are noop CVTs.

  // Is_True( Need_type_conversion(opc_type, opnd_type, NULL) == NOT_AT_ALL,
  //  ("STR_RED::Is_implicit_cvt_linear:  cvt is not implicit."));

  // e.g., disable str-red of I8I4CVT ... if do not allow wrap around opt
  if (! Allow_wrap_around_opt && 
      MTYPE_size_min(opc_type) != MTYPE_size_min(opnd_type))
    return FALSE;
  return TRUE;
}
/* CVTL-RELATED finish */

//======================================================================
// Does the rhs (with any parens/converts removed) match the lhs var?
//======================================================================

CODEREP *
STR_RED::Matches_lhs( const CODEREP *lhs, const CODEREP *rhs ) const
{
  Is_True( lhs->Kind() == CK_VAR,
    ("STR_RED::Matches_lhs: Lhs not a var") );
  const IDTYPE lhs_id = lhs->Aux_id();

  if ( rhs->Kind() == CK_VAR ) {
    if ( lhs_id == rhs->Aux_id() &&
         MTYPE_size_min(lhs->Dsctyp()) <= MTYPE_size_min(rhs->Dsctyp()) ) {
      // lhs and rhs match
      return (CODEREP*)rhs;
    }
  }
  else if ( rhs->Kind() == CK_OP ) {
    const OPERATOR opr = rhs->Opr();
    if (opr == OPR_PAREN)
      return Matches_lhs( lhs, rhs->Opnd(0) );

    // Note that OPR_CVT is not allowed. 
  }

  return NULL;
}

//======================================================================
// Does the IV update happen rarely enough relative to the point where
// its result would be used that it's worth repairing any injury it
// causes? Without feedback, we guess that any update in deeper-nested
// loop than the use will happen too frequently, so we might as well
// recompute at the use point. With feedback, we say the injury isn't
// worth repairing if it happens at least INJURY_USE_FREQ_CUTOFF_RATIO
// times as often as the use. TODO: Let the effective value of
// INJURY_USE_FREQ_CUTOFF_RATIO depend on whether the operation being
// strength-reduced is multiplication or addition/subtraction. We
// should be more tolerant of a high IV update frequency in the former
// case, and less tolerant in the latter.
//======================================================================

#define INJURY_USE_FREQ_CUTOFF_RATIO 2.0

BOOL
STR_RED::Update_happens_rarely_enough(      BB_NODE *update_bb,
				            BB_NODE *innermost_use_bb,
				      const CODEREP *use_expr) const
{
  if (Cfg()->Feedback ()) {
    FB_FREQ freq1 = Cfg()->Feedback()->Get_node_freq_out (update_bb->Id ());
    FB_FREQ freq2 =
      Cfg()->Feedback()->Get_node_freq_out (innermost_use_bb->Id ());

    if (freq1.Known () && freq2.Known ())
      return (freq1 <= freq2 * INJURY_USE_FREQ_CUTOFF_RATIO);
  }
  return In_same_or_lower_nesting(innermost_use_bb, update_bb);
}

//======================================================================
// Is the given expression seen in the bb loop invariant for the
// innermost loop that contains bb?
//======================================================================

BOOL
STR_RED::Is_const_or_loop_invar( const CODEREP *expr, BB_NODE *bb ) const
{
  if ( inCODEKIND(expr->Kind(), CK_LDA|CK_CONST|CK_RCONST) ) {
    return TRUE;
  }
  else {
    const BB_LOOP *loop = Cfg()->Find_innermost_loop_contains( bb );
    if ( loop && 
	 loop->True_body_set()->MemberP( bb ) ) {
      // it's really inside the loop
      // make sure it is loop-invariant
      if ( loop->Invariant_cr((CODEREP *)expr) ) {
	// if it's a variable, we also need to make sure that its
	// definition is at a point that dominates this block
	if ( inCODEKIND(expr->Kind(), CK_VAR) ) {
	  BB_NODE *defbb = expr->Defbb();
	  if ( !defbb->Dominates(bb) ) {
	    return FALSE;
	  }
	}

	return TRUE;
      }
    }
  }

  return FALSE;
}

//======================================================================
// Find the definition of the variable, following the u-d chain
// until we get a real store to the variable, and return its rhs.
//
// This should only be useful when the rhs of an IV-update was CSE'd
// and we need to get back to the original the original updated rhs.
// (i = i+<expr>)
//======================================================================

CODEREP *
STR_RED::Find_real_defs_rhs( const CODEREP *var ) const
{
  // is_true note:  we should never call this function because we
  // should never cse the rhs of an IV.
  Is_True( var != NULL,
    ("STR_RED::Find_real_def: should not have called this function!") );

  Is_True( var->Kind() == CK_VAR,
    ("STR_RED::Find_real_def: not a var") );

  if ( var->Is_flag_set((CR_FLAG)(CF_DEF_BY_CHI|
				  CF_IS_ZERO_VERSION)) )
  {
    FmtAssert( FALSE,
      ("STR_RED::Find_real_def: def'd by chi/zero-version") );
  }
  else if ( var->Is_flag_set(CF_DEF_BY_PHI) ) {
    PHI_NODE *phi = var->Defphi();
    CODEREP *real_rhs = NULL;
    BB_NODE *pred;
    BB_LIST_ITER pred_iter;
    INT pred_num = 0;
    FOR_ALL_ELEM( pred, pred_iter, Init(phi->Bb()->Pred()) ) {
      // we should only have to get the first one
      CODEREP *opnd = phi->OPND(pred_num);
      CODEREP *tmp_rhs = Find_real_defs_rhs( opnd );

      // should always get the same result
      Is_True( real_rhs == NULL || real_rhs->Match(tmp_rhs),
	("STR_RED::Find_real_def: cr's don't match") );

      real_rhs = tmp_rhs;

      pred_num++;
    }

    Is_True( real_rhs != NULL,
      ("STR_RED::Find_real_defs_rhs: no real_rhs found") );
    return real_rhs;
  }
  else {
    STMTREP *vardef = var->Defstmt();
    Is_True( OPERATOR_is_scalar_store (vardef->Opr()),
      ("STR_RED::Find_real_def: not def'd by store") );

    return vardef->Rhs();
  }

  return NULL;
}

//======================================================================
// Simple function to aid in getting type-corrected operand
//======================================================================

inline CODEREP *
Str_red_get_fixed_operand( const CODEREP *cr, INT which_opnd )
{
  Is_True( cr->Kind() == CK_OP,
    ("Str_red_get_fixed_operand: not an op") );

  CODEREP *opnd = cr->Get_opnd(which_opnd);
  return opnd;
}

//======================================================================
// Returns the threshold adjusted by taking into account the content of the
// innermost loop that bb belongs to
//======================================================================
INT
STR_RED::Local_autoaggstr_reduction_threshold(BB_NODE *bb) const
{
  if (! WOPT_Enable_Str_Red_Use_Context)
    return WOPT_Enable_Autoaggstr_Reduction_Threshold;
  const BB_LOOP *loop = Cfg()->Find_innermost_loop_contains( bb );
  if (loop == NULL)
    return WOPT_Enable_Autoaggstr_Reduction_Threshold;
  return MAX(0,
          WOPT_Enable_Autoaggstr_Reduction_Threshold - loop->Size_estimate()/18);
}

//======================================================================
// Determine if the statement updates an induction variable, and
// return the induction variable being updated (one on rhs), the
// increment amount, and whether or not it's an increment or decrement
//======================================================================

BOOL
STR_RED::Find_iv_and_incr( STMTREP *stmt, CODEREP **updated_iv,
			   CODEREP **incr_amt, BOOL *is_add,
			   BOOL aggstr_cand ) const
{
  CODEREP *lhs = stmt->Lhs();
  CODEREP *rhs = stmt->Rhs();

  Is_True( lhs->Kind() == CK_VAR,
    ("STR_RED::Find_iv_and_incr: Lhs not a var") );

  if (aggstr_cand)
    // check if the number of induction expression injured by this stmt
    // exceeds the threshold 
    if (stmt->Str_red_num() >= Local_autoaggstr_reduction_threshold(stmt->Bb()))
      return FALSE;

  // it's possible we've CSE'd the rhs of the iv update, so we need
  // to find the true update.  Only true if we already know the stmt
  // is an iv update.
  if ( rhs->Kind() == CK_VAR && stmt->Iv_update() ) {
    rhs = Find_real_defs_rhs( rhs );
  }

  if ( rhs->Kind() == CK_OP ) {
    const OPERATOR opr = rhs->Opr();
    if ( opr == OPR_ADD ) {
      CODEREP *rhs_iv;
      if ( (rhs_iv = Matches_lhs( lhs, rhs->Opnd(0) )) != NULL ) {
	// i = i+E
	if ( Is_const_or_loop_invar( rhs->Opnd(1), stmt->Bb() ) ) {
	  *updated_iv = rhs_iv;
	  *incr_amt = Str_red_get_fixed_operand(rhs,1);
	  *is_add = TRUE;
	  return TRUE;
	}
      }
      else if ( (rhs_iv = Matches_lhs( lhs, rhs->Opnd(1) )) != NULL ) {
	// i = E+i
	if ( Is_const_or_loop_invar( rhs->Opnd(0), stmt->Bb() ) ) {
	  *updated_iv = rhs_iv;
	  *incr_amt = Str_red_get_fixed_operand(rhs,0);
	  *is_add = TRUE;
	  return TRUE;
	}
      }
    }
    else if ( opr == OPR_SUB ) {
      CODEREP *rhs_iv;
      if ( (rhs_iv = Matches_lhs( lhs, rhs->Opnd(0) )) != NULL ) {
	// i = i-E
	if ( Is_const_or_loop_invar( rhs->Opnd(1), stmt->Bb() ) ) {
	  *updated_iv = rhs_iv;
	  *incr_amt = Str_red_get_fixed_operand(rhs,1);
	  *is_add = FALSE;
	  return TRUE;
	}
      }
    }
  }

  return FALSE;
}

//======================================================================
// Determine if the statement updates an induction variable
//
// i = i+E
// i = E+i
// i = i-E
//
// E must be either a constant, or be a loop-invariant expression in
// the same loop that contains the statement.
//
// Also return the "updated" induction variable (i.e., the appearance
// on the rhs) if updated is non-NULL and we're returning true.
//======================================================================

BOOL
STR_RED::Determine_iv_update( STMTREP *stmt, CODEREP **updated ) const
{
  // first check if we've already dealt with this statement
  if ( stmt->Not_iv_update() )
    return FALSE;
  if ( stmt->Iv_update() ) {
    // doesn't the caller want to know which value is being updated?
    if ( updated == NULL ) {
      return TRUE;
    }
  }

  // screen out non-STIDs
  // screen out non-integral types
  // screen out volatile stores
  const OPERATOR opr = stmt->Opr();
  if (! OPERATOR_is_scalar_store (opr) ||
      !MTYPE_is_integral(OPCODE_desc(stmt->Op())) ||
      stmt->Volatile_stmt()) {
    stmt->Set_not_iv_update();
    return FALSE;
  }

  CODEREP *dummy_iv, *dummy_incr;
  BOOL dummy_is_add;
  if ( Find_iv_and_incr(stmt, &dummy_iv, &dummy_incr, &dummy_is_add) ) {
    stmt->Set_iv_update();
    if ( updated ) *updated = dummy_iv;
    return TRUE;
  }

  // it's possible that the rhs of a statement that we originally
  // thought was an iv-update was CSE'd and there's just a LDID of the
  // temp.
  if ( stmt->Iv_update() ) {
    if ( stmt->Rhs()->Kind() == CK_VAR ) {
      // hack by finding the original rhs (whatever defined this var)
      // and temporarily replacing this rhs.
      CODEREP *cur_rhs = stmt->Rhs();
      CODEREP *new_rhs = Find_real_defs_rhs( cur_rhs );
      stmt->Set_rhs( new_rhs );
      BOOL result = Determine_iv_update( stmt, updated );
      stmt->Set_rhs( cur_rhs );

      Is_True( result,
	("STR_RED::Determine_iv_update: same stmt, diff answer"));

      return result;
    }
  }

  // catch any fall-through
  if ( stmt->Iv_update() ) {
    Is_Trace( Tracing(), (TFile, "STR_RED::Determine_iv_update:\n") );
    Is_Trace_cmd( Tracing(), stmt->Print(TFile) );
    Is_Trace( Tracing(), (TFile, "#########################\n") );
    Is_Trace_cmd( Tracing(), Cfg()->Print(TFile,TRUE) );
  }

  Is_True( !stmt->Iv_update(),
    ("STR_RED::Determine_iv_update: same statement, different answer"));
  stmt->Set_not_iv_update();
  return FALSE;
}

//======================================================================
// Determine if bb1 is in the same loop as bb2, or if bb1 is in a
// loop enclosed by bb2's loop (if any)
//======================================================================

BOOL
STR_RED::In_same_or_lower_nesting( BB_NODE *bb1, BB_NODE *bb2 ) const
{
  const BB_LOOP *loop1 = Cfg()->Find_innermost_loop_contains( bb1 );
  const BB_LOOP *loop2 = Cfg()->Find_innermost_loop_contains( bb2 );

  // second bb not in any loop?
  if ( loop2 == NULL ) {
    // bb2 not in a loop, and that contains all loops
    return TRUE;
  }
  else {
    // see if bb2's loop contains bb1's loop
    return loop2->Contains(loop1);
  }
}

//======================================================================
// Determine if any of the operands are defined by IV update statements
//======================================================================

BOOL
STR_RED::Determine_iv_update_phi(      PHI_NODE *phi,
				 const CODEREP  *cand_expr) const
{
  // do we already know the answer?
  if ( phi->Opnd_not_iv_update() )
    return FALSE;
  if ( phi->Opnd_iv_update() )
    return TRUE;

  BOOL iv_update = FALSE;
  PHI_OPND_ITER phi_opnd_iter(phi);
  CODEREP *opnd;
  INT      opnd_num = 0;
  FOR_ALL_ELEM(opnd, phi_opnd_iter, Init()) {
    // don't know where the def of zero-version vars are
    if ( opnd->Is_flag_set(CF_IS_ZERO_VERSION) ) {
      iv_update = FALSE;
      break;
    }

    // the variable can't be defined by either a chi or another phi
    if ( ! opnd->Is_flag_set((CR_FLAG)(CF_DEF_BY_CHI|CF_DEF_BY_PHI)) ) {
      STMTREP *stmt = opnd->Defstmt();
      if ( Determine_iv_update( stmt, NULL ) ) {
	// make sure the defining statement is either within the same
	// loop as this phi, or in an enclosing loop (or no loop).
	// This avoids the problem of having the update inside another
	// loop.
	if (Update_happens_rarely_enough(stmt->Bb(),
					 phi->Bb()->Nth_pred(opnd_num),
					 cand_expr)) {
	  iv_update = TRUE;
	}
	else {
	  Is_Trace(Tracing(), (TFile, "STR_RED::Determine_iv_update_phi: "
			     "loop nest level precludes.\n"));
	}
      }
    }
    ++opnd_num;
  }

  if ( iv_update )
    phi->Set_opnd_iv_update();
  else
    phi->Set_opnd_not_iv_update();
  
  return iv_update;
}

//======================================================================
// Determine if "last" is defined by a chain of IV updates, one of
// which has "first" on its RHS.
//======================================================================

BOOL
STR_RED::Updated_by_iv_update(const CODEREP *first, 
			      const CODEREP *last,
			            CODEREP *invar,
			            BB_NODE *innermost_use_bb,
			      const CODEREP *cand_expr,
			            BOOL aggstr_cand) const
{
  Is_True( first->Kind() == CK_VAR && last->Kind() == CK_VAR,
    ("STR_RED::Updated_by_iv_update: non-VARs") );
  Is_True( first->Aux_id() == last->Aux_id(),
    ("STR_RED::Updated_by_iv_update: non-matching IDs") );

  // do not handle indirect defs, or phi-defs
  if (last->Is_flag_set((CR_FLAG)( CF_DEF_BY_CHI|
				   CF_DEF_BY_PHI|
				   CF_IS_ZERO_VERSION)) )
  {
    return FALSE;
  }

  STMTREP *last_def = last->Defstmt();

  if (aggstr_cand)
    // check if the number of induction expression injured by this stmt
    // exceeds the threshold 
    if (last_def->Str_red_num() >= Local_autoaggstr_reduction_threshold(last_def->Bb()))
      return FALSE;

  // Continue following the chain of IV updates only if the innermost
  // loop containing last_def also contains the point of eventual use
  // of the putative injury repair result.
  if (Update_happens_rarely_enough(last_def->Bb(), innermost_use_bb,
				   cand_expr) ||
      Repaired(last_def)) {
    CODEREP *iv_on_rhs;
    if ( Determine_iv_update( last_def, &iv_on_rhs ) ) {
      Is_True( iv_on_rhs->Kind() == CK_VAR,
	      ("STR_RED::Updated_by_iv_update: not a var") );

      // we must check if the invar value is invariant with
      // respect to this iv-update
      if (invar != NULL && 
	  !Is_const_or_loop_invar(invar, last_def->Bb())) {
	  return FALSE;
	}

      if ( first == iv_on_rhs ) {
	// last_def defines last, and first is on its rhs
	return TRUE;
      }
      else {
	// last_def defines last, but some other version is on its rhs,
	// so continue up until we stop finding iv updates, or we find
	// first.
	return Updated_by_iv_update(first, iv_on_rhs, invar,
				    innermost_use_bb, cand_expr, aggstr_cand);
      }
    }
  }
  else {
    Is_Trace(Tracing(), (TFile, "-----\nSTR_RED::Updated_by_iv_update: "
			 "chain broken by loop nesting\n"));
    Is_Trace(Tracing(), (TFile, "   innermost use in BB%d\n",
			 innermost_use_bb->Id()));
    Is_Trace(Tracing(), (TFile, "   iv update in BB%d\n",
			 last_def->Bb()->Id()));
  }

  return FALSE;
}

//======================================================================
// Determine if the coderep is a variable defined by an IV-update
// (or chain of updates), and that def_cr is the value being updated
// (i.e., on rhs of one of the updates)
//======================================================================

BOOL
STR_RED::Defined_by_iv_update(const CODEREP *use_cr,
			      const CODEREP *def_cr,
			            CODEREP *invar,
			            BB_NODE *use_bb,
			      const CODEREP *cand_expr,
			            BOOL aggstr_cand) const
{
  Is_True(use_cr->Kind() != CK_OP, ("STR_RED::Defined_by_iv_update:  CK_OP"));

  // do not handle if defined by chi
  if ( use_cr->Kind() != CK_VAR ||
       use_cr->Is_flag_set((CR_FLAG)(CF_DEF_BY_CHI|CF_IS_ZERO_VERSION)))
  {
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "STR_RED::Defined_by_iv_update: "
	      "def by chi/zero-ver --> FALSE\n"));
    return FALSE;
  }

  // check defined by phi's
  if ( use_cr->Is_flag_set(CF_DEF_BY_PHI) ) {
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "STR_RED::Defined_by_iv_update: "
	      "def by phi\n"));
    PHI_NODE *phi = use_cr->Defphi();
    // is this phi a join point for an iv-update?
    if ( Determine_iv_update_phi(phi, cand_expr) ) {
      // determine which, if any, of the operands is defined by a
      // chain of iv updates, which ultimately updates def_cr
      PHI_OPND_ITER phi_opnd_iter(phi);
      CODEREP *opnd;
      FOR_ALL_ELEM(opnd, phi_opnd_iter, Init()) {
	if (Updated_by_iv_update(def_cr, opnd, invar, use_bb, cand_expr, aggstr_cand)) {
	  return TRUE;
	}
      }
    }
  }
  else {
    // is the use defined by a chain of IV updates that ultimately
    // updates def_cr
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "STR_RED::Defined_by_iv_update: "
	      "checking def:\n"));
    Is_Trace_cmd(Tracing() && WOPT_Enable_Verbose,
		 def_cr->Print(3, TFile));
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "        use:\n"));
    Is_Trace_cmd(Tracing() && WOPT_Enable_Verbose,
		 use_cr->Print(3, TFile));
    if (Updated_by_iv_update(def_cr, use_cr, invar, use_bb, use_cr, aggstr_cand)) {
      Is_Trace(Tracing() && WOPT_Enable_Verbose,
	       (TFile, "STR_RED::Defined_by_iv_update: "
		"updated_by_iv_update --> TRUE\n"));
      return TRUE;
    }
  }

  Is_Trace(Tracing() && WOPT_Enable_Verbose,
	   (TFile, "Defined_by_iv_update: FALSE\n"));
  // catch fall-through cases
  return FALSE;
}

//======================================================================
// Determine if the coderep is a variable defined by an IV-update
// (or chain of updates), and that the final rhs of the iv-update is
// defined in a block that dominates def_bb.
//
// Return through 'def_cr' the earliest updated IV.
//======================================================================

BOOL
STR_RED::Defined_by_iv_update_no_def(      CODEREP *use_cr,
				           BB_NODE *def_bb,
				           CODEREP **def_cr,
				           CODEREP *invar,
				           BB_NODE *use_bb,
				     const CODEREP *cand_expr,
				           BOOL aggstr_cand) const
{
  Is_Trace(Tracing() && WOPT_Enable_Verbose,
	   (TFile, "STR_RED::Defined_by_iv_update_no_def: use:\n"));
  Is_Trace_cmd(Tracing() && WOPT_Enable_Verbose,
	       use_cr->Print(3, TFile));
  Is_Trace(Tracing() && WOPT_Enable_Verbose,
	   (TFile, "STR_RED::Defined_by_iv_update_no_def: "
	    "use in BB%d\n", use_bb->Id()));

  Is_True(use_cr->Kind() != CK_OP, ("STR_RED::Defined_by_iv_update_no_def:  CK_OP"));

  // do not handle if zero-version
  if ( use_cr->Kind() != CK_VAR ||
       use_cr->Is_flag_set(CF_IS_ZERO_VERSION) )
  {
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "STR_RED::Defined_by_iv_update_no_def: "
	      "def by zero-ver --> FALSE\n"));
    return FALSE;
  }

  // keep following the use-def chain through IV-updates until we
  // get to something that stops us.
  INT num_iv_updates = 0;

  while ( ! use_cr->Is_flag_set( (CR_FLAG) (CF_DEF_BY_CHI|
					    CF_DEF_BY_PHI|
					    CF_IS_ZERO_VERSION) ) )
  {
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "STR_RED::Defined_by_iv_update_no_def: "
	      "def by real\n"));
    STMTREP *use_cr_def = use_cr->Defstmt();

    // is this var defined in a block that dominates def_bb 
    // (i.e., we've gone far enough up the use-def chain)
    //
    // The following "if" condition should be simplified to
    // if (!def_bb->Dominatates(use_cr_def->Bb())) ...
    if (use_cr_def->Bb() != def_bb && 
	use_cr_def->Bb()->Dominates(def_bb)) {
      // have we passed through any iv-updates?
      if ( num_iv_updates > 0 ) {
	if ( def_cr != NULL ) *def_cr = use_cr;
	Is_Trace(Tracing() && WOPT_Enable_Verbose,
		 (TFile, "STR_RED::Determine_iv_update_no_def: "
		  "def dominates with IV updates --> TRUE\n"));
	return TRUE;
      }

      Is_Trace(Tracing() && WOPT_Enable_Verbose,
	       (TFile, "STR_RED::Determine_iv_update_no_def: "
		"def dominates with no IV updates --> FALSE\n"));
      return FALSE;
    }

    // check if this variable is defined by an IV-update
    if (Determine_iv_update(use_cr_def, NULL)) {
      CODEREP *new_use_cr, *dummy_incr;
      BOOL dummy_is_add;
      if ( Find_iv_and_incr( use_cr_def, &new_use_cr, 
			     &dummy_incr, &dummy_is_add, aggstr_cand ) )
      {
	// determine if the invar value is indeed invariant with
	// respect to this update
	if (invar && 
	    !Is_const_or_loop_invar(invar,use_cr_def->Bb()) )
	{
	  Is_Trace(Tracing() && WOPT_Enable_Verbose,
		   (TFile, "STR_RED::Determine_iv_update_no_def: "
		    "invar not const or loop invariant --> FALSE\n"));
	  return FALSE;
	}

	// If the chain of IV updates passes through something (likely
	// to be) executed too much more frequently than the use,
	// treat the IV update as a kill, not an injury (unless it
	// already got fixed).
	if (!Update_happens_rarely_enough(use_cr_def->Bb(),
					  use_bb, use_cr) &&
	    !Repaired(use_cr_def)) {
	  Is_Trace(Tracing(),
		   (TFile, "STR_RED::Defined_by_iv_update_no_def: "
		    "chain broken by loop nesting\n"));
	  Is_Trace(Tracing(), (TFile, "   innermost use in BB%d\n",
			       use_bb->Id()));
	  Is_Trace(Tracing(), (TFile, "   iv update in BB%d\n",
			       use_cr_def->Bb()->Id()));
	  return FALSE;
	}

	use_cr = new_use_cr;
        num_iv_updates++;
	continue;
      }
    }

    // if make it here, we're done
    break;
  }

  // find out where the last use_cr is defined, and determine if
  // that block dominates the def_bb.
  BB_NODE *use_cr_def_bb = NULL;
  if ( use_cr->Is_flag_set(CF_IS_ZERO_VERSION) ) {
    FmtAssert( FALSE,
      ("STR_RED::Defined_by_iv_update_no_def: zero-version?") );
    return FALSE;
  }
  else if ( use_cr->Is_flag_set(CF_DEF_BY_PHI) ) {
    use_cr_def_bb = use_cr->Defphi()->Bb();

    // if the variable is defined by a phi in the same block, it is
    // considered to dominate the expression's phi, which is ultimately
    // what we're interested in.
    if ( use_cr_def_bb == def_bb ||
         use_cr_def_bb->Dominates( def_bb ) )
    {
      // have we passed through any iv-updates?
      if ( num_iv_updates > 0 ) {
	if ( def_cr != NULL ) *def_cr = use_cr;
	Is_Trace(Tracing() && WOPT_Enable_Verbose,
		 (TFile, "STR_RED::Determine_iv_update_no_def: "
		  "phi def dominates with IV updates --> TRUE\n"));
	return TRUE;
      }
    }
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "Defined_by_iv_update_no_def: "
	      "phi def; no iv updates || non-dominance --> FALSE\n"));
    return FALSE;
  }
  else {
    use_cr_def_bb = use_cr->Defstmt()->Bb();
    if ( use_cr_def_bb != def_bb && 
	 use_cr_def_bb->Dominates( def_bb ) )
    {
      // have we passed through any iv-updates?
      if ( num_iv_updates > 0 ) {
	if ( def_cr != NULL ) *def_cr = use_cr;
	Is_Trace(Tracing() && WOPT_Enable_Verbose,
		 (TFile, "STR_RED::Determine_iv_update_no_def: "
		  "chi or zero-ver def dominates with IV updates --> TRUE\n"));
	return TRUE;
      }
    }
    Is_Trace(Tracing() && WOPT_Enable_Verbose,
	     (TFile, "Defined_by_iv_update_no_def: "
	      "real def; no iv updates || non-dominance --> FALSE\n"));
    return FALSE;
  }

  /*NOTREACHED*/
  FmtAssert( FALSE,
    ("STR_RED::Defined_by_iv_update_no_def: didn't find def bb") );

  /*NOTREACHED*/
  return FALSE;
}

//======================================================================
// Reset array (for new expression)
//======================================================================

void
STR_RED::Clear_iv_expression_occurs() {
  if (_iv_occurs.Elements() > 0){
    _iv_occurs.Free_array();
    _iv_occurs.Resetidx();
  }
}

//======================================================================
// Add an expression occurrence to the list.
//======================================================================

void
#ifdef TARG_NVISA
STR_RED::Add_iv_expression_occurs(const CODEREP *cr, BB_NODE *bb, BOOL in_ivar)
#else
STR_RED::Add_iv_expression_occurs(const CODEREP *cr, BB_NODE *bb)
#endif
{
  BOOL found = FALSE;
  for (int i=0; i < Iv_expression_occurs_count(); i++) {
    if (Get_iv_expression_occurs(i).cr->Coderep_id() == cr->Coderep_id()) {
      found = TRUE;
      break;
    }
  }
  EXPRESSION_SR_OCCURS occur;
  occur.cr = (CODEREP *)cr;
  occur.bb = bb;
#ifdef TARG_NVISA
  occur.in_ivar = in_ivar;
#endif

  if (!found)
    _iv_occurs.AddElement(occur);
}

//======================================================================
// Get the recorded status for this iv, if any
//======================================================================

IV_SR_STATUS
STR_RED::Get_recorded_status(const CODEREP *iv) 
{
  if (iv->Kind() != CK_VAR)
    return IV_SR_STATUS_UNKNOWN;

  INT32 iv_id = iv->Aux_id();

  for (int i = 0; i < Iv_record_count(); i++) {
    if (iv_id == Get_iv_record(i).iv_aux_id)
      return Get_iv_record(i).status;
  }
  return IV_SR_STATUS_UNKNOWN;
}

//======================================================================
// Record the status of checking this iv
//======================================================================

void
STR_RED::Add_iv_record(const CODEREP *iv, IV_SR_STATUS status)
{
  Is_Trace(Tracing(), (TFile, "Status recorded:%d\n", status));
  Is_Trace_cmd(Tracing(), iv->Print_node(0,TFile));

  IV_SR_RECORD record;
  record.iv_aux_id = iv->Aux_id();
  record.status = status;

  _iv_occurs_record.AddElement(record);
}

//======================================================================
// Collect all expressions (op/ivar) in this cr tree
// for which iv is direct child
//======================================================================

BOOL
#ifdef TARG_NVISA
STR_RED::Collect_expression(const CODEREP *cr, const CODEREP *iv, BB_NODE *bb, BOOL in_ivar)
#else
STR_RED::Collect_expression(const CODEREP *cr, const CODEREP *iv, BB_NODE *bb)
#endif
{
  switch (cr->Kind()) {

  case CK_VAR:
    if (cr->Aux_id() == iv->Aux_id())
      return TRUE;
    break;
    
  case CK_IVAR:
    if ((cr->Opr() == OPR_ILOAD) || (cr->Opr() == OPR_PARM)) {
      CODEREP * base = cr->Ilod_base()? cr->Ilod_base(): cr->Istr_base();
#ifdef TARG_NVISA
      if (Collect_expression(base, iv, bb, TRUE))
        Add_iv_expression_occurs(cr, bb, FALSE);
#else
      if (Collect_expression(base, iv, bb))
        Add_iv_expression_occurs(cr, bb);
#endif
    }
    break;

  case CK_OP:
    for (INT32 i = 0; i < cr->Kid_count(); i++)
#ifdef TARG_NVISA
      if (Collect_expression(cr->Opnd(i), iv, bb, FALSE)) {
        Add_iv_expression_occurs(cr, bb, in_ivar);
#else
      if (Collect_expression(cr->Opnd(i), iv, bb)) {
        Add_iv_expression_occurs(cr, bb);
#endif
        break;
      }
    break;

  case CK_LDA:
  case CK_CONST:
  case CK_RCONST:
  default:
    break;
  }
  
  return FALSE;
}

//======================================================================
// Collect all expressions in this block for which iv is direct child
//======================================================================

BOOL 
STR_RED::Collect_iv_occurs(const CODEREP *iv, BB_NODE *bb)
{
  STMTREP     *stmt;
  STMTREP_ITER stmt_iter(bb->Stmtlist());
  
  FOR_ALL_NODE(stmt, stmt_iter, Init()) {
    
    // branches can be LFTR'ed, inaccurate test
    if ((stmt->Opr() == OPR_TRUEBR) ||
        (stmt->Opr() == OPR_FALSEBR)) {
      continue;
    }

    // skip iv update
    CODEREP *dummy_iv;
    if (Determine_iv_update(stmt, &dummy_iv) &&
        (iv->Aux_id() == dummy_iv->Aux_id()))
        continue;

    CODEREP *rhs = stmt->Rhs();
    CODEREP *lhs = stmt->Lhs();

#ifdef TARG_NVISA
    if ((rhs != NULL) && Collect_expression(rhs, iv, bb, FALSE)) {
#else
    if ((rhs != NULL) && Collect_expression(rhs, iv, bb)) {
#endif
      Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - bad occur):\n"));
      Is_Trace_cmd(Tracing(), stmt->Print_node(TFile));
      return FALSE;
    }
    
#ifdef TARG_NVISA
    if ((lhs != NULL) && Collect_expression(lhs, iv, bb, FALSE)) {
#else
    if ((lhs != NULL) && Collect_expression(lhs, iv, bb)) {
#endif
      Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - bad occur):\n"));
      Is_Trace_cmd(Tracing(), stmt->Print_node(TFile));
      return FALSE;
    }
  }  

  return TRUE;
}

#ifdef TARG_NVISA

// Check if cr1 and cr2 belong to the same group of expressions.
//
BOOL
Same_expr_pattern (const CODEREP *iv_cr, const CODEREP *expr_cr)
{
  if (iv_cr->Kind() != expr_cr->Kind()) {
    return FALSE;
  }

  if (iv_cr->Kind() == CK_LDA || iv_cr->Kind() == CK_CONST || iv_cr->Kind() == CK_RCONST) {
    return (iv_cr == expr_cr);
  }

  if (iv_cr->Kind() == CK_VAR) {
    return (iv_cr->Aux_id() == expr_cr->Aux_id());
  }

  if (iv_cr->Kind() == CK_IVAR) {
    if (iv_cr->Opr() != expr_cr->Opr() || iv_cr->Opr() != OPR_ILOAD) {
      return FALSE;
    }

    if (iv_cr->Ilod_base() == NULL || expr_cr->Ilod_base() == NULL) {
        return FALSE;
    }

    return Same_expr_pattern (iv_cr->Ilod_base(), expr_cr->Ilod_base());
  }

  if (iv_cr->Opr() != expr_cr->Opr() || iv_cr->Kid_count() != expr_cr->Kid_count()) {
    return FALSE;
  }

  for (int i = 0;  i < iv_cr->Kid_count(); i++) {
    if (!Same_expr_pattern(iv_cr->Opnd(i), expr_cr->Opnd(i))) {
      return FALSE;
    }
  }

  return TRUE;
}

typedef enum {
   BB_VISITED_INIT     = 0,
   BB_VISITED_MARKED   = 1,
   BB_VISITED_EXPR_OCC = 2
} BB_VISITED_KIND;


// Check if there is a path from bb to loop exit without
// hit any expression occurance
BOOL
Find_path_to_loop_exit_without_hit_expr_occ (BB_LOOP *loop, BB_NODE *bb)
{
  if (bb->Get_visited() == BB_VISITED_EXPR_OCC) {
    // Hit one of the expression occurance
    return FALSE;
  }

  if (bb->Get_visited() == BB_VISITED_MARKED) {
    // already visited
    return FALSE;
  }

  // Mark BB visited
  bb->Set_visited (BB_VISITED_MARKED);

  BB_LIST_ITER bb_list_iter(bb->Succ());
  BB_NODE *succ_bb;
  FOR_ALL_ELEM (succ_bb, bb_list_iter, Init()) {
    if (succ_bb->In_loop(loop)) {
      if (Find_path_to_loop_exit_without_hit_expr_occ (loop, succ_bb)) {
        return TRUE;
      }
    } else {
      return TRUE;
    }
  }

  return FALSE;
}

// Check if all expression occurances inside a loop form a general 
// post-dominator to the loop header. This is to find
// if there is a path from the loop header to exit without passing
// any blocks with expression occurances.
//
BOOL
STR_RED::Check_expr_occs_post_dom_loop_header(BB_LOOP *loop, const CODEREP *expr_cr)
{
  // Initialize visited field of all blocks in a loop
  BB_NODE_SET_ITER loop_body_iter;
  BB_NODE *bb;
  FOR_ALL_ELEM (bb, loop_body_iter, Init(loop->True_body_set())) {
    bb->Set_visited (BB_VISITED_INIT);
  }

  // Mark blocks with expression occurance
  for (int i = 0; i < Iv_expression_occurs_count(); i++) {

    EXPRESSION_SR_OCCURS occurs = Get_iv_expression_occurs(i);

    if (Same_expr_pattern (occurs.cr, expr_cr)) {
      occurs.bb->Set_visited (BB_VISITED_EXPR_OCC);
    }
  }

  // check if from loop header to exit, there is a path
  // which does not pass any expression occurance
  return !Find_path_to_loop_exit_without_hit_expr_occ (loop, loop->Header());
}
#endif

//======================================================================
// Return TRUE if all occurences of iv in this loop are expected
// to be strength reduced away
//======================================================================

BOOL
STR_RED::Iv_can_disappear(const CODEREP *iv,
                          BB_NODE *use_bb,
                          const CODEREP *cr)
{
  // paranoia 1
  if (cr == NULL || iv == NULL || use_bb == NULL) {
    return FALSE;
  }

  BB_LOOP *loop = use_bb->Innermost();

  // paranoia 2
  if ((loop == NULL) || Is_const_or_loop_invar(iv, use_bb)) {
    return FALSE;
  }

  // first use of iv
#ifdef TARG_NVISA
  Add_iv_expression_occurs((CODEREP *)cr, use_bb, FALSE);
#else
  Add_iv_expression_occurs((CODEREP *)cr, use_bb);
#endif

  // collect tther uses of iv in loop
  BB_NODE_SET_ITER loop_body_iter;
  BB_NODE *bb;
  FOR_ALL_ELEM (bb, loop_body_iter, Init(loop->True_body_set())) {
    if (!Collect_iv_occurs(iv, bb)) {
      return FALSE;
    }
  }

  INT32 aux_id_iv = iv->Aux_id();

#ifdef TARG_NVISA
  // Information about if an expression occurance post-dominate loop header
  // if one or all expression occs post-dominate the loop header, the computation
  // of the expression could be moved out of the loop, and IV could be
  // eliminated.
  BOOL found_down_safety_expr_occ = FALSE;
  INT32 expr_occ_count = 0;     // count number of expression occurance, and if it is more than 1,
                                // general post-dominator will be checked if none of the expr occ
                                // post-dominate the loop header

  // Consider the following heuristics here,
  //   When an IV occurance in the form of "IVAR iv" is seen, keep a counter for it,
  //   and other IV occurance in the form of IVAR iv +/- loop-invar are also
  //   counted, and at the end, if there are cases IVAR iv +/- loop-invar 
  //   the IV will be considered. The reason is that simply reject "IVAR iv"
  //   may reject more strength reduction cases involving the same IV, and
  //   keep one IV around for multiple strength reduction cases should be good.
  //
  INT32 ivar_ref_count = 0;     // count occurance in the form of  "IVAR iv"
  INT32 in_ivar_count = 0;      // count number of iv occurance in the form of "IVAR iv +/- var"
#endif

  // Go over all uses to see if they are good
  for (int i = 0; i < Iv_expression_occurs_count(); i++) {

    EXPRESSION_SR_OCCURS occurs = Get_iv_expression_occurs(i);
    CODEREP *iv_cr = occurs.cr;

#ifdef TARG_NVISA
    in_ivar_count += occurs.in_ivar;

    // check if the iv_cr is an expression occurance
    BOOL is_expr_occ = Same_expr_pattern (iv_cr, cr);

#endif

    // this block should be one from where we can hoist
    if (!(Control_dependence_implies_down_safety(occurs.bb,
                                                 loop,
                                                 Pool()))) {
#ifdef TARG_NVISA
      if (is_expr_occ) {
        expr_occ_count++;
        // consider all expression occurs together later
        continue;
      }
#endif
      Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - control relation):\n"));
      Is_Trace_cmd(Tracing(), iv_cr->Print_node(0,TFile));
      return FALSE;
    }
    
#ifdef TARG_NVISA
    if (is_expr_occ) {
      found_down_safety_expr_occ = TRUE;
      continue;
    }

    // Consider the iv occurance which is not an expression occurance 

    if (iv_cr->Kind() == CK_IVAR) {
      // keep info about "IVAR iv" occs
      ivar_ref_count ++;
      continue;
    }
#endif

    // all operators should be add/sub
    if ((iv_cr->Kind() != CK_OP) ||
        ( (iv_cr->Opr() != OPR_ADD)
          && (iv_cr->Opr() != OPR_SUB) )) {
      Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - operator):\n"));
      Is_Trace_cmd(Tracing(), iv_cr->Print_node(0,TFile));
      return FALSE;
    }

    for (INT32 i = 0; i < iv_cr->Kid_count(); i++) {

      CODEREP *child = iv_cr->Opnd(i);

      // skip this child, it's the iv
      if ((child->Kind() == CK_VAR) && 
          (child->Aux_id() == aux_id_iv))
        continue;

      // for the tree of the other child,

      // it should be invariant
      if (!Is_const_or_loop_invar(child, use_bb)) {
        Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - invariance):\n"));
        Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
        return FALSE;
      }
      // it should not be a constant
      if (child->Kind() == CK_CONST) {
        Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - immediate const):\n"));
        Is_Trace_cmd(Tracing(), iv_cr->Print_node(0,TFile));
        return FALSE;
      }
    }
  }

#ifdef TARG_NVISA

  // if there are IV occs in the form of "IVAR iv", but there
  // are no other IV occs in the form of "IVAR iv +/- loop-invar",
  // reject the IV
  if (WOPT_Enable_Estr_Iload) {
    if (ivar_ref_count != 0 && in_ivar_count == 0) {
      return FALSE;
    }
  } else {
    if (ivar_ref_count != 0) {
      return FALSE;
    }
  }

  if (found_down_safety_expr_occ == FALSE) {
    // if none of the expression occurance post-dominate the loop header
    // check if all expression occurances post-dominate the loop header
    return expr_occ_count > 1 && Check_expr_occs_post_dom_loop_header (loop, cr);
  } 

#endif

  return TRUE;
}

//======================================================================
// Return TRUE if this transformation can reduce register pressure
//
// Right now this only evaluates SR of add/sub operators
// If all expressions of the form iv+k can be strength reduced
// the original iv can be dead coded away reducing register pressure
// So this is the case we are trying to recoganize
//
// To determine profitability of (iv+k) is an SR candidate, we collect
// all expressions for which iv is a direct child. Then we evaluate if 
// all of these occurences can be strength reduced. If so, we return
// true, and this one is reduced and the result cached for next time
//
// The recorded occurences will be used in next stage (of implementation)
// to rewrite remaining occurence of this iv in terms of new one
//
//======================================================================

BOOL 
STR_RED::Is_profitable(const CODEREP *iv, 
                       BB_NODE *use_bb,
                       const CODEREP *cr)
{
  // restricted check
  if ((cr == NULL) || 
      (cr->Kind() != CK_OP) ||
      (iv->Kind() != CK_VAR)) {
    Is_Trace(Tracing(), (TFile, "Yes estr candidate(reason - unhandled):\n"));
    Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
    return TRUE;
  }

  const OPERATOR opr = cr->Opr();

  if (WOPT_Enable_Estr_Less_Adds &&
      ( (opr == OPR_ADD) || (opr == OPR_SUB) ) ) {

    switch(Get_recorded_status(iv)) {

    case IV_SR_STATUS_ACCEPT:
      Is_Trace(Tracing(), (TFile, "Yes estr candidate(reason - recorded iv):\n"));
      Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
      return TRUE;

    case IV_SR_STATUS_REJECT:
      Is_Trace(Tracing(), (TFile, "Non estr candidate(reason - recorded iv):\n"));
      Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
      return FALSE;

    case IV_SR_STATUS_UNKNOWN:
      // clear any previous stored values
      Clear_iv_expression_occurs();

      if (Iv_can_disappear(iv, use_bb, cr)) {
        Add_iv_record(iv, IV_SR_STATUS_ACCEPT);
        return TRUE;
      } else {
        Add_iv_record(iv, IV_SR_STATUS_REJECT);
        return FALSE;
      }
    default:
      return FALSE;
    }
  }

  Is_Trace(Tracing(), (TFile, "Yes estr candidate(reason - default):\n"));
  Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
  return TRUE;
}

//======================================================================
// Determine if this occurrence is a strength-reduction candidate
//
// Expression must be in form:  i*k+c
// but because we're only dealing with expressions whose kids are
// terminals, we look for patterns:
//   i*k  or  k*i	(i*k+0)
//   i+k  or  k+i	(i*1+k)
//   i-k		(i*1+k)
//   -i			(i*-1+0)
//
//   *** NOTE ***
//    When change is made to STR_RED::Candidate, remember to check
//    if the same change should be in STR_RED::Candidate_phi_res.
//
//======================================================================

BOOL 
STR_RED::Candidate( const CODEREP *cr,
		    const CODEREP *def_opnd0, const CODEREP *def_opnd1,
		    BB_NODE *def_bb,
		    CODEREP *use_opnd0, CODEREP *use_opnd1,
		    BB_NODE *use_bb )
{
#ifdef TARG_NVISA
  BB_LOOP *bb_loop = NULL;
  Is_Trace(Tracing(), (TFile, "estr candidate Operation: "));
  Is_Trace_cmd(Tracing(), cr->Print_node(0,TFile));
  Is_Trace(Tracing(), (TFile, "\n"));
  if ( def_bb != NULL ) {
      bb_loop = def_bb->Innermost();
      Is_Trace(Tracing(), (TFile, 
	"Non-Null BB Node information and Loop Depth=%d\n", 
	def_bb->Loopdepth()));
  }
  else {
      Is_Trace(Tracing(), (TFile, "Null BB information\n"));
  }

  if ( bb_loop != NULL ) {
      Is_Trace(Tracing(), (TFile, 
	"Non-Null Loop information, Loop Depth=%d and Max Depth=%d\n", 
	bb_loop->Depth(), bb_loop->Max_depth() ));
      if ( ! WOPT_Enable_Estr_Early_Exit
	&& bb_loop->Exit_early() && ! WOPT_Enable_Aggressive_Code_Motion) 
      {
	// If we don't do aggressive code motion, but do strength reduce
	// a loop with an early exit, then we end up with worse code
	// because it goes to the expense of extra regs for the address,
	// but doesn't hoist the address computation out of the loop.
	// For some (unknown) reason, the code for self-contained ifs 
	// inside loops is good, it is only a problem with early exit loops.
        DevWarn("early exit loop and not aggcm, so don't strength reduce?");
	return FALSE;
      }
  } else {
      Is_Trace(Tracing(), (TFile, "Null Loop information\n"));
  }

  if ( ! WOPT_Enable_Estr_Outer_Loop
    && bb_loop != NULL && bb_loop->Depth() <= 1 && bb_loop->Max_depth() >=2 ) 
  {
      // Dont strength reduce the outermost loop in deeply nested loops
      return FALSE; 
  }
#endif

  const OPERATOR opr = cr->Opr();
  switch ( opr ) {
    case OPR_ADD:
    case OPR_SUB:
#ifdef TARG_NVISA
	Is_Trace(Tracing(), (TFile, 
	  "Number of uses of this add/sub operation=%d\n", cr->Usecnt()));
	if ( ! WOPT_Enable_Estr_Const_Opnds
	  && ( cr->Get_opnd(0)->Kind() == CK_CONST 
	    || cr->Get_opnd(1)->Kind() == CK_CONST ) )
	{
	    // Don't Consider for strength reduction, if one operand is constant
	    Is_Trace(Tracing(), (TFile, "At least one constant operand\n"));
	    break; 
	}
	if ( ! WOPT_Enable_Estr_Used_Once && !(cr->Usecnt() > 1) ) {
	    // Don't consider for strength reduction, 
	    // if SR variable is not used in more than one place
       	    break;
	}
	// else fallthru
#endif
    case OPR_MPY:
      // i*k
      // i+k
      // i-k
      if ( Defined_by_iv_update(use_opnd0, def_opnd0,
      				use_opnd1, use_bb, cr, 
				opr == OPR_ADD || opr == OPR_SUB))
      {
	if (Is_cvt_linear(use_opnd0) &&
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd0->Dtyp())) {
          return Is_profitable(use_opnd0, use_bb, cr);
        }
      }
      else 
      // k*i
      // k+i
      // k-i
      if ( Defined_by_iv_update(use_opnd1, def_opnd1,
				use_opnd0, use_bb, cr,
				opr == OPR_ADD || opr == OPR_SUB))
      {
        if (Is_cvt_linear(use_opnd1) &&
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd1->Dtyp())) {
          return Is_profitable(use_opnd1, use_bb, cr);
        }
      }
      break;

    case OPR_NEG:
	Is_Trace(Tracing(), (TFile, 
          "Number of uses of this neg operation=%d\n", cr->Usecnt()));
#ifdef TARG_NVISA
	if ( ! WOPT_Enable_Estr_Used_Once && !(cr->Usecnt() > 1) ) {
	    // Don't consider for strength reduction, 
	    // if SR variable is not used in more than one place
	    break;
	}
	// else fallthru
#endif
#ifdef TARG_X8664 // avoid U4 due to zero-extension to high-order 32 bits
      if (cr->Dtyp() == MTYPE_U4)
	return FALSE;
#endif
      // i*-1
      if (Defined_by_iv_update( use_opnd0, def_opnd0, NULL, use_bb, cr)) {
	if (Is_cvt_linear(use_opnd0) && 
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd0->Dtyp())) {
	  return TRUE;
	}
      }
      break;

    case OPR_CVT:
      // CVT(i)
      if (Defined_by_iv_update( use_opnd0, def_opnd0, NULL, use_bb, cr)) {
	if (Is_cvt_linear(cr) && 
	    Is_cvt_linear(use_opnd0) &&
	    Is_implicit_cvt_linear(cr->Dsctyp(), use_opnd0->Dtyp())) {
	  return TRUE;
	}
      }
      break;
  }

  // catch fall-thru cases
  return FALSE;
}

//======================================================================
// Determine if the use occurrence is a strength-reduction candidate
// but assume that the def occurrence is in a phi result in the
// def_bb.
//======================================================================

BOOL 
STR_RED::Candidate_phi_res( const CODEREP *cr,
		    	    BB_NODE *def_bb,
			    CODEREP *use_opnd0, CODEREP *use_opnd1,
			    BB_NODE *use_bb )
{
  CODEREP *def_opnd;
  const OPERATOR opr = cr->Opr();
  switch ( opr ) {
    case OPR_ADD:
    case OPR_SUB:
    case OPR_MPY:
      // i*k
      // i+k
      // i-k
      if ( Defined_by_iv_update_no_def(use_opnd0, def_bb, &def_opnd,
				       use_opnd1, use_bb, cr,
				       opr == OPR_ADD || opr == OPR_SUB))
      {
	if (Is_cvt_linear(use_opnd0) &&
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd0->Dtyp())) {
          return Is_profitable(use_opnd0, use_bb, cr);
	}
      }
      else 
      // k*i
      // k+i
      // k-i
      if ( Defined_by_iv_update_no_def(use_opnd1, def_bb, &def_opnd,
				       use_opnd0, use_bb, cr,
				       opr == OPR_ADD || opr == OPR_SUB))
      {
	if (Is_cvt_linear(use_opnd1) &&
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd1->Dtyp())) {
          return Is_profitable(use_opnd1, use_bb, cr);
	}
      }
      break;

    case OPR_NEG:
#ifdef TARG_X8664 // avoid U4 due to zero-extension to high-order 32 bits
      if (cr->Dtyp() == MTYPE_U4)
	return FALSE;
#endif
      // i*-1
      if ( Defined_by_iv_update_no_def(use_opnd0, def_bb, &def_opnd,
				       NULL/*invar*/, use_bb, cr))
      {
	if (Is_cvt_linear(use_opnd0) && 
	    Is_implicit_cvt_linear(cr->Dtyp(), use_opnd0->Dtyp())) {
	  return TRUE;
	}
      }
      break;

    case OPR_CVT:
      if ( Defined_by_iv_update_no_def(use_opnd0, def_bb, &def_opnd,
				       NULL/*invar*/, use_bb, cr))
      {
	if (Is_cvt_linear(cr) && 
	    Is_cvt_linear(use_opnd0) &&
	    Is_implicit_cvt_linear(cr->Dsctyp(), use_opnd0->Dtyp())) {
	  return TRUE;
	}
      }
      break;
  }

  // catch fall-thru cases
  return FALSE;
}

//======================================================================
// Find the induction variable that we want to strength reduce in the
// "use" occurrence.  Find the matching variable in the "def" occur.
//======================================================================

void
STR_RED::Find_iv_and_mult( const EXP_OCCURS *def, CODEREP **iv_def,
			   const EXP_OCCURS *use, CODEREP **iv_use,
			   CODEREP **multiplier ) const
{
  Is_True(use->Occ_kind() == EXP_OCCURS::OCC_REAL_OCCUR ||
	  use->Occ_kind() == EXP_OCCURS::OCC_PHI_PRED_OCCUR,
	  ("STR_RED::Find_iv_and_mult: use is not real"));

  // phi results are a different breed
  if ( def->Occ_kind() == EXP_OCCURS::OCC_PHI_OCCUR ) {
    Find_iv_and_mult_phi_res( def, iv_def, use, iv_use, multiplier );
    return;
  }

  const CODEREP *use_cr = use->Occurrence();
  const CODEREP *def_cr = def->Occurrence();
  Is_True( use_cr->Kind() == CK_OP,
	   ("STR_RED::Find_iv_and_mult: use is not op") );

  MTYPE n1_mtype = OPCODE_rtype(use_cr->Op());
#if defined(TARG_NVISA)
  // Should probably do this for all targets?
  // For now, just do our target (even pathscale didn't change these uses,
  // but did change other uses).
  // -1 needs a signed type, else U4 -1 wil be large int 
  // which could get cvt'ed to U8 and then do wrong arithmetic on.
  n1_mtype = Mtype_TransferSign(MTYPE_I4, n1_mtype);
#endif

  // determine which operand is the iv we wish to work on
  if ( use_cr->Kid_count() == 2 ) {
    const OPERATOR opr = use_cr->Opr();
    Is_True( opr == OPR_MPY || opr == OPR_ADD || opr == OPR_SUB,
	   ("STR_RED::Find_iv_and_mult: bad op: %s",
	   OPCODE_name(use_cr->Op())) );

    if ( Defined_by_iv_update(use_cr->Opnd(0), def_cr->Opnd(0),
			      use_cr->Opnd(1), use->Bb(), use_cr))
    {
      *iv_use = use_cr->Opnd(0);
      *iv_def = def_cr->Opnd(0);

      if ( opr == OPR_MPY ) {
	*multiplier = Str_red_get_fixed_operand(use_cr,1);
      }
      else {
	// multiplier is 1
	*multiplier = NULL;
      }
    }
    else 
    if ( Defined_by_iv_update(use_cr->Opnd(1), def_cr->Opnd(1),
			      use_cr->Opnd(0), use->Bb(), use_cr))
    {
      *iv_use = use_cr->Opnd(1);
      *iv_def = def_cr->Opnd(1);

      if ( opr == OPR_MPY ) {
	*multiplier = Str_red_get_fixed_operand(use_cr,0);
      }
      else if ( opr == OPR_SUB) {
        *multiplier = Htable()->Add_const(n1_mtype,-1);
      }
      else { // opr == OPR_ADD
	// multiplier is 1
	*multiplier = NULL;
      }
    }
    else {
      FmtAssert( FALSE,
	("STR_RED::Find_iv_and_mult: not a candidate") );
    }
  }
  else if ( use_cr->Kid_count() == 1 ) {
    Is_True( use_cr->Opr() == OPR_NEG || use_cr->Opr() == OPR_CVT,
      ("STR_RED::Find_iv_and_mult: single kid op is not neg") );
      
    BOOL found_iv_update = Defined_by_iv_update(use_cr->Opnd(0),
						def_cr->Opnd(0),
						NULL/*invar*/,
						use->Bb(), use_cr);
    
    Is_True(found_iv_update, 
	("STR_RED::Find_iv_and_mult: single kid is not iv") );
    
    *iv_use = use_cr->Opnd(0);
    *iv_def = def_cr->Opnd(0);
    if ( use_cr->Opr() == OPR_NEG ) {
      CODEREP *multi = Htable()->Add_const(n1_mtype,-1);
      *multiplier = multi;
    }
    else {
      // multiplier is 1
      *multiplier = NULL;
    }
  }
  else {
    FmtAssert( FALSE,
      ("STR_RED::Find_iv_and_mult: invalid sr candidate") );
  }
}

//======================================================================
// Find the induction variable that we want to strength reduce in the
// "use" occurrence.  Find the matching variable in the "def" occur
// when it is a phi result.
//======================================================================

void
STR_RED::Find_iv_and_mult_phi_res( const EXP_OCCURS *def, CODEREP **iv_def,
			   const EXP_OCCURS *use, CODEREP **iv_use,
			   CODEREP **multiplier ) const
{
  Is_True( def->Occ_kind() == EXP_OCCURS::OCC_PHI_OCCUR,
	   ("STR_RED::Find_iv_and_mult_phi_res: def is not phi") );
  Is_True(use->Occ_kind() == EXP_OCCURS::OCC_REAL_OCCUR ||
	  use->Occ_kind() == EXP_OCCURS::OCC_PHI_PRED_OCCUR,
	  ("STR_RED::Find_iv_and_mult_phi_res: use is not real"));

  const CODEREP *use_cr = use->Occurrence();
  Is_True( use_cr->Kind() == CK_OP,
	   ("STR_RED::Find_iv_and_mult_phi_res: use is not op") );

  Is_Trace(Tracing(), (TFile, "def: "));
  Is_Trace_cmd(Tracing(), def->Print(TFile));
  Is_Trace(Tracing(), (TFile, "use: "));
  Is_Trace_cmd(Tracing(), use->Print(TFile));

  MTYPE n1_mtype = OPCODE_rtype(use_cr->Op());
#if defined(TARG_X8664) || defined(TARG_NVISA)
  // pathscale bug 4518
  // Should probably do this for all targets?
  // -1 needs a signed type, else U4 -1 wil be large int 
  // which could get cvt'ed to U8 and then do wrong arithmetic on.
  n1_mtype = Mtype_TransferSign(MTYPE_I4, n1_mtype);
#endif

  // determine which operand is the iv we wish to work on
  if ( use_cr->Kid_count() == 2 ) {
    const OPERATOR opr = use_cr->Opr();
    Is_True( opr == OPR_MPY || opr == OPR_ADD || opr == OPR_SUB,
	   ("STR_RED::Find_iv_and_mult_phi_res: bad op: %s",
	   OPCODE_name(use_cr->Op())) );

    if ( Defined_by_iv_update_no_def(use_cr->Opnd(0), def->Bb(), iv_def,
				     use_cr->Opnd(1), use->Bb(), use_cr))
    {
      *iv_use = use_cr->Opnd(0);
      // *iv_def already set

      if ( opr == OPR_MPY ) {
        *multiplier = use_cr->Opnd(1);
      }
      else {
	// multiplier is 1
	*multiplier = NULL;
      }
    }
    else 
    if ( Defined_by_iv_update_no_def(use_cr->Opnd(1), def->Bb(), iv_def,
				     use_cr->Opnd(0), use->Bb(), use_cr))
    {
      *iv_use = use_cr->Opnd(1);
      // *iv_def already set

      if ( opr == OPR_MPY ) {
        *multiplier = use_cr->Opnd(0);
      }
      else if ( opr == OPR_SUB) {
        *multiplier = Htable()->Add_const(n1_mtype,-1);
      }
      else { // opr == OPR_ADD
	// multiplier is 1
	*multiplier = NULL;
      }
    }
    else {
      FmtAssert( FALSE,
	("STR_RED::Find_iv_and_mult_phi_res: not a candidate") );
    }
  }
  else if ( use_cr->Kid_count() == 1 ) {
    const OPERATOR opr = use_cr->Opr();
    Is_True( opr == OPR_NEG || opr == OPR_CVT,
      ("STR_RED::Find_iv_and_mult_phi_res: single kid op is not neg") );
      
    BOOL found_iv_update = Defined_by_iv_update_no_def(use_cr->Opnd(0),
						       def->Bb(),
				      	               iv_def,
						       NULL/*invar*/,
						       use->Bb(),
						       use_cr);
    Is_True( found_iv_update, 
      ("STR_RED::Find_iv_and_mult_phi_res: single kid is not iv") );
    
    *iv_use = use_cr->Opnd(0);
    // *iv_def already set
    if ( opr == OPR_NEG ) {
      CODEREP *multi = Htable()->Add_const(n1_mtype,-1);
      *multiplier = multi;
    }
    else if ( opr == OPR_CVT ) {
      // multiplier is 1
      *multiplier = NULL;
    }
    else {
      FmtAssert( FALSE,
        ("STR_RED::Find_iv_and_mult_phi_res: wrong op") );
    }
  }
  else {
    FmtAssert( FALSE,
      ("STR_RED::Find_iv_and_mult_phi_res: invalid sr candidate") );
  }

}

//======================================================================
// See if this expression ever shows up on the rhs of an iv-update,
// which means we will exclude it from being a strength-reduction
// candidate
//======================================================================
// Unused?
#if 0
void
EXP_WORKLST::Exclude_strength_reduction_cands(ETABLE *etable)
{
  EXP_OCCURS        *exp_occ;
  EXP_OCCURS_ITER    exp_occ_iter;

  // if we're not enabled, this everything is excluded
  if ( ! WOPT_Enable_New_SR ) {
    Set_exclude_sr_cand();
    return;
  }

#ifdef EXCLUDE_IV_RHS_FROM_SR
  // is it even something we would accept as an iv update?
  //

  // if it's not an op, we don't consider it as an rhs of iv-update
  if ( Exp()->Kind() != CK_OP ) {
    return;
  }

  // if it's not an add/sub, we don't consider it as an rhs of iv-update
  const OPERATOR exp_opr = Exp()->Opr();
  if ( exp_opr != OPR_ADD && exp_opr != OPR_SUB ) {
    return;
  }

  // we now have an expression that *may* show up on the rhs of an
  // iv update, so see if it ever does
  FOR_ALL_NODE (exp_occ, exp_occ_iter, Init(Real_occurs().Head())) {
    STMTREP *occ_stmt = exp_occ->Enclosed_in_stmt();
    if ( occ_stmt->Rhs() == exp_occ->Occurrence() &&
         etable->Str_red()->Determine_iv_update(occ_stmt,NULL) )
    {
      // we'll have to exclude this expression from strength-reduction
      Set_exclude_sr_cand();

      Is_Trace(etable->Tracing(),
	(TFile,"Exclude_strength_reduction_cands: excluded iv rhs\n"));

      break;
    }
  }
#endif // EXCLUDE_IV_RHS_FROM_SR

}
#endif

void
STR_RED::Perform_per_expr_cleanup(void)
{
  while (!_repaired_statements.Is_Empty()) {
    STMTREP *stmt = _repaired_statements.Pop();
    Is_True(stmt->Repaired(), ("STR_RED::Perform_per_expr_cleanup: "
			       "statement must be repaired. Duplicate?"));
    stmt->Reset_repaired();
  }
}

void
STR_RED::Set_repaired(STMTREP *stmt)
{
  stmt->Set_repaired();
  _repaired_statements.Push(stmt);
}
