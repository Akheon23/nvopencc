/*

  Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.

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

  Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
  Mountain View, CA 94043, or:

  http://www.sgi.com

  For further information regarding this notice, see:

  http://oss.sgi.com/projects/GenInfo/NoticeExplan

*/


//
// Generate instruction decoding information.
/////////////////////////////////////
/////////////////////////////////////
//
//  $Revision: 1.1 $
//  $Date: 03/03/20 23:08:49-00:00 $
//  $Author: jliu@keyresearch.com $
//  $Source: /scratch/mee/Patch0002-taketwo/kpro64-pending/common/targ_info/isa/x8664/SCCS/s.isa_decode.cxx $

#include "topcode.h"
#include "isa_decode_gen.h"
#include "targ_isa_bundle.h"
 
main()
{
  ISA_Decode_Begin("x8664");

  STATE unit = Create_Unit_State("unit", 0, 3);
  Initial_State(unit);

  ISA_Decode_End();
  return 0;
}
