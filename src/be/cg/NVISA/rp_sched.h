/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef RP_SCHED
#define RP_SCHED

#include "bb.h"
#include "rp_live.h"
#include "mempool.h"
#include "op.h"
#include "tn.h"
#include "tn_set.h"
#include "cg_vector.h"
#include "op_map.h"

class Rp_Scheduler
{
  public:
    Rp_Scheduler(BB *bb, Bb_Linfo *info, MEM_POOL *pool); //constructor
    OP* Get_Next_Element();                               //returns next op to
                                                          //be scheduled
    void Schedule(OP *op);                                //add op to the schedule
    void Get_Schedule(BB *bb);                            //get new schedule

  private:
    MEM_POOL *_pool;
    TN_SET *_live;
    VECTOR _sched;
    VECTOR _ready;
};

#endif //RP_SCHED
