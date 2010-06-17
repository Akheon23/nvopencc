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

#ifndef RP_CORE_INCLUDED
#define RP_CORE_INCLUDED
#include "mempool.h"

class Rp_Liveness;

#define RP_TRACE_LIVENESS 0x001

//==========================================================
// Rp_Core is the top level class for register pressure
// relater optimizations. It holds all data and uses
// multiple slave optimizations to get the required effect
//==========================================================
//==========================================================

class Rp_Core {
 private:
  BOOL          _init;
  MEM_POOL    * _local_pool;
  Rp_Liveness * _liveness;

 public:
  Rp_Core() { _init = FALSE;  }

  void Init(MEM_POOL *pool) {
    _local_pool = pool; 
    _liveness = NULL;
    _init = TRUE;
  }

  // Accessors
  MEM_POOL*      Pool()     { return _local_pool; }
  Rp_Liveness *  Liveness() { return _liveness; }

  // Interface
  void Compute_Liveness();
  void Compute_Dependence_Graph();
  void Do_Local_Scheduling();
  
};

//==========================================================
// Global functions to initialize and finish this set of
// optimizations
//==========================================================

static MEM_POOL Rp_Pool;

Rp_Core* Rp_Init();
void Rp_Fini();


#endif // RP_CORE_INCLUDED
