/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

//-*-c++-*-
// ====================================================================
// ====================================================================
//
// Module: optimizer.h
// $Revision: 1.2 $
// $Date: 02/11/07 23:41:57-00:00 $
// $Author: fchow@keyresearch.com $
// $Source: /scratch/mee/Patch0002-taketwo/kpro64-pending/be/opt/SCCS/s.optimizer.h $
//
// Revision history:
//  14-SEP-94 - Original Version
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
//
// Description:
//
// The external interface for the optimizer.
//
// ====================================================================
// ====================================================================


#ifndef optimizer_INCLUDED
#define optimizer_INCLUDED      "optimizer.h"
#ifdef _KEEP_RCS_ID
static char *optimizerrcs_id =	optimizer_INCLUDED"$Revision: 1.2 $";
#endif /* _KEEP_RCS_ID */

#include "opt_alias_interface.h"

/*  The phases of PREOPT */
typedef enum {
  PREOPT_PHASE,		// used for -PHASE:p
  PREOPT_LNO_PHASE,	// used for -PHASE:l
  PREOPT_DUONLY_PHASE,  // called by LNO, but will disable optimization */
#ifdef TARG_NVISA
  PREOPT_CMC_PHASE,
  PREOPT_CMC_PHASE2,
#endif
  MAINOPT_PHASE,	// used for -PHASE:w
  PREOPT_IPA0_PHASE,	// called by IPL
  PREOPT_IPA1_PHASE,	// called by main IPA
} PREOPT_PHASES;

typedef PREOPT_PHASES OPT_PHASE;

#ifdef __cplusplus
extern "C" {
#endif


/* Clients of the optimizer pass a WHIRL tree for the function, and
 * receive back a possibly optimized version of the tree.
 */
struct DU_MANAGER;

extern WN *Pre_Optimizer( INT32 /* PREOPT_PHASES */,  WN *, DU_MANAGER * , ALIAS_MANAGER *);

DU_MANAGER* Create_Du_Manager(MEM_POOL *);
void               Delete_Du_Manager(DU_MANAGER *, MEM_POOL *);


#ifdef __cplusplus
}
#endif
#endif /* optimizer_INCLUDED */

