/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#include "rp_sched.h"
#include "cg_dep_graph.h"

//assumption: no inst has more than 10 operands.
const INT MaxCost = 10;

Rp_Scheduler::Rp_Scheduler(BB *bb, Bb_Linfo *info, MEM_POOL *pool)
{
  //initialize the pool
  _pool = pool;

  //initialize _live set with live-outs
  _live = TN_SET_Copy(info->Set(BB_LIVE_OUT), _pool);

  //initialize sched to empty vector
  _sched = VECTOR_Init(BB_length(bb), _pool);

  //initialize _ready with leaf ops
  _ready = VECTOR_Init(BB_length(bb), _pool);

  OP *op;
  FOR_ALL_BB_OPs_REV(bb, op) {
    if (OP_succs(op) == NULL) {
      VECTOR_Add_Element(_ready, op);
    }
  }
}

OP*
Rp_Scheduler::Get_Next_Element()
{
  INT i, j;
  INT mincost = MaxCost;
  OP *best_op = NULL;

  //find best op
  for (i = 0 ; i < VECTOR_count(_ready) ; i++) {
    OP *op = (OP *)VECTOR_element(_ready, i);
    INT cost = 0;

    //an opnd that is not live increases the cost by 1
    for (j = 0 ; j < OP_opnds(op) ; j++) {
      TN *tn = OP_opnd(op, j);
      if (TN_is_register(tn) && !TN_is_dedicated(tn) && 
          !TN_SET_MemberP(_live, tn))
        cost++;
    }

    //a result that is live decreases the cost by 1
    for (j = 0 ; j < OP_results(op) ; j++) {
      TN *tn = OP_result(op, j);
      if (TN_is_register(tn) && TN_SET_MemberP(_live, tn))
        cost--;
    }

    if (cost < mincost) {
      mincost = cost;
      best_op = op;
    }
  }

  return best_op;
}

void
Rp_Scheduler::Schedule(OP *op)
{
  INT i;

  //schedule op
  VECTOR_Add_Element(_sched, op);

  //remove op from _ready
  VECTOR_Delete_Element(_ready, op);

  //update _live
  //_live -= op->defs
  for (i = 0 ; i < OP_results(op) ; i++) {
    TN *tn = OP_result(op, i);
    if (TN_is_register(tn)) {
      _live = TN_SET_Difference1D(_live, tn);
    }
  }

  //_live += op->uses
  for (i = 0 ; i < OP_opnds(op) ; i++) {
    TN *tn = OP_opnd(op, i);
    if (TN_is_register(tn)) {
      _live = TN_SET_Union1D(_live, tn, _pool);
    }
  }

  //add new ops to _ready
  ARC_LIST *pred_arcs, *succ_arcs;
  for (pred_arcs = OP_preds(op) ; pred_arcs != NULL ; 
      pred_arcs = ARC_LIST_rest(pred_arcs)) {
    ARC *pred_arc = ARC_LIST_first(pred_arcs);
    OP *pred_op = ARC_pred(pred_arc);

    for (succ_arcs = OP_succs(pred_op) ; succ_arcs != NULL ; 
        succ_arcs = ARC_LIST_rest(succ_arcs)) {
      ARC *succ_arc = ARC_LIST_first(succ_arcs);
      OP *succ_op = ARC_succ(succ_arc);

      //if (succ_op is not scheduled yet) break;
      if (!VECTOR_Member_Element(_sched, (void *)succ_op)) break;
    }

    //if all succs are scheduled then add to _ready
    if (succ_arcs == NULL && !VECTOR_Member_Element(_ready, pred_op)) {
      VECTOR_Add_Element(_ready, pred_op);
    }
  }
}

void
Rp_Scheduler::Get_Schedule(BB *bb)
{
  INT i;

  FmtAssert(BB_length(bb) == VECTOR_count(_sched), ("BB length is different from schedule"));

  BB_Remove_All(bb);

  for (i = VECTOR_count(_sched) - 1 ; i >= 0 ; i--) {
    OP *op = (OP *)VECTOR_element(_sched, i);
    BB_Append_Op(bb, op);
  }
}
