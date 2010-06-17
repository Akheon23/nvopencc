/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/*

  This program is free software; you can redistribute it and/or modify it
  under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it would be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

  Further, this software is distributed without any warranty that it is
  free of the rightful claim of any third person regarding infringement 
  or the like.  Any license provided herein, whether implied or 
  otherwise, applies only to this software file.  Patent licenses, if 
  any, provided herein do not apply to combinations of this program with 
  other software, or any other product whatsoever.  

  You should have received a copy of the GNU General Public License along
  with this program; if not, write the Free Software Foundation, Inc., 59
  Temple Place - Suite 330, Boston MA 02111-1307, USA.

*/

#include "defs.h"
#include "rp_core.h"
#include "rp_live.h"
#include "tracing.h"
#include "cg_dep_graph.h"
#include "rp_sched.h"

//==========================================================
//  Initialize and compute liveness information
//==========================================================

void
Rp_Core::Compute_Liveness()
{
  _liveness = (Rp_Liveness*) CXX_NEW(Rp_Liveness, Pool());
  Liveness()->Init(Pool());
  Liveness()->Build();
  Liveness()->Compute_Max_Lives();
}

//==========================================================
//  Schedule instructions in blocks with high register
//  pressure
//==========================================================

void
Rp_Core::Do_Local_Scheduling()
{
  BB *bb;

  Compute_Liveness();

  for (bb = REGION_First_BB ; bb != NULL ; bb = BB_next(bb)) {
    //compute data dep graph
    CG_DEP_Compute_Graph(bb,
        INCLUDE_ASSIGNED_REG_DEPS,
        NON_CYCLIC,
        NO_MEMREAD_ARCS,
        INCLUDE_MEMIN_ARCS,
        INCLUDE_CONTROL_ARCS,
        NULL);

    //get liveness info for bb
    Bb_Linfo *info = Liveness()->Live_Info(bb);

    //initialize scheduler
    Rp_Scheduler sched(bb, info, Pool());

    //start scheduling!
    OP *op;
    while((op = sched.Get_Next_Element()) != NULL) {
      sched.Schedule(op);
    }

    //get the new schedule in bb
    sched.Get_Schedule(bb);

    //delete data dep graph
    CG_DEP_Delete_Graph(bb);
  }

  if (Get_Trace(TP_RP, RP_TRACE_LIVENESS)) {
    Liveness()->Print(TFile);
  }
}

//==========================================================
//  Initialize function for register pressure optimizations
//==========================================================

Rp_Core* Rp_Init()
{
  MEM_POOL_Initialize(&Rp_Pool, "CG_Rp_Pool", TRUE);
  MEM_POOL_Push(&Rp_Pool);

  Rp_Core * core = (Rp_Core*) CXX_NEW(Rp_Core, &Rp_Pool);
  core->Init(&Rp_Pool);
  return core;  
}

//==========================================================
//  Finalize function for register pressure optimizations
//==========================================================

void Rp_Fini()
{
  MEM_POOL_Pop(&Rp_Pool);
  MEM_POOL_Delete(&Rp_Pool);
}
