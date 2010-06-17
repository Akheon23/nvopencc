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


//  
//  Generate ABI information
///////////////////////////////////////


//  $Revision: 1.22 $
//  $Date: 2001/03/10 01:15:59 $
//  $Author: mtibuild $
//  $Source: /osprey.src/osprey1.0/common/targ_info/abi/ia64/RCS/abi_properties.cxx,v $

#include <stddef.h>
#include "abi_properties_gen.h"
#include "targ_isa_registers.h"

static ABI_PROPERTY
	allocatable,
	func_arg,	// params
	func_val,	// retval
	unused,		// unused dummy result when only using cc
        // Define special regs here; these are really part of isa, not abi, 
        // but here is where we have ability to assign a dedicated reg to them.
        // tid/ctaid are 16bit in ptx1, 32bit in ptx2,
        // so define two variants which cg will have to choose between.
	tid1,		// thread id
	ntid1,		// number of thread ids
	ctaid1,		// cta id
	nctaid1,	// number of cta ids
	tid,		// thread id
	ntid,		// number of thread ids
	ctaid,		// cta id
	nctaid,		// number of cta ids
        laneid,         // lane id
        warpid,         // warp id
        nwarpid,        // number of warp ids
        smid,           // sm id
        nsmid,          // number of sm ids
        gridid,         // grid id
        lanemask,       // mask with bit set in position relative to lane number
        pm,             // perf monitor counter
	clock,		// cycle counter
        clock64;        // cycle counter 64b

#define ISA_REGISTER_CLASS_Last_Reg(x) \
	ISA_REGISTER_CLASS_INFO_Last_Reg(ISA_REGISTER_CLASS_Info(x))

#define MAX_PARAM_REGS 32 
#define MAX_RETURN_REGS 12

static const char *i32param_names[MAX_PARAM_REGS] = {
  "%ra1", "%ra2", "%ra3", "%ra4", "%ra5", "%ra6", "%ra7", "%ra8", 
  "%ra9", "%ra10", "%ra11", "%ra12", "%ra13", "%ra14", "%ra15", "%ra16",
  "%ra17", "%ra18", "%ra19", "%ra20", "%ra21", "%ra22", "%ra23", "%ra24",
  "%ra25", "%ra26", "%ra27", "%ra28", "%ra29", "%ra30", "%ra31", "%ra32"};
static const char *i32ret_names[MAX_RETURN_REGS] = {
  "%rv1", "%rv2", "%rv3", "%rv4", "%rv5", "%rv6", "%rv7", "%rv8",
  "%rv9", "%rv10", "%rv11", "%rv12"};
static const char *i64param_names[MAX_PARAM_REGS] = {
  "%rda1", "%rda2", "%rda3", "%rda4", "%rda5", "%rda6", "%rda7","%rda8",
  "%rda9", "%rda10", "%rda11", "%rda12", "%rda13", "%rda14", "%rda15","%rda16",
  "%rda17", "%rda18", "%rda19", "%rda20", "%rda21", "%rda22", "%rda23","%rda24",
  "%rda25", "%rda26", "%rda27", "%rda28", "%rda29", "%rda30", "%rda31","%rda32"};
static const char *i64ret_names[MAX_RETURN_REGS] = {
  "%rdv1", "%rdv2", "%rdv3", "%rdv4", "%rdv5", "%rdv6", "%rdv7", "%rdv8",
  "%rdv9", "%rdv10", "%rdv11", "%rdv12"};
static const char *f32param_names[MAX_PARAM_REGS] = {
  "%fa1", "%fa2", "%fa3", "%fa4", "%fa5", "%fa6", "%fa7", "%fa8", 
  "%fa9", "%fa10", "%fa11", "%fa12", "%fa13", "%fa14", "%fa15", "%fa16",
  "%fa17", "%fa18", "%fa19", "%fa20", "%fa21", "%fa22", "%fa23", "%fa24",
  "%fa25", "%fa26", "%fa27", "%fa28", "%fa29", "%fa30", "%fa31", "%fa32"};
static const char *f32ret_names[MAX_RETURN_REGS] = {
  "%fv1", "%fv2", "%fv3", "%fv4", "%fv5", "%fv6", "%fv7", "%fv8",
  "%fv9", "%fv10", "%fv11", "%fv12"};
static const char *f64param_names[MAX_PARAM_REGS] = {
  "%fda1", "%fda2", "%fda3", "%fda4", "%fda5", "%fda6", "%fda7","%fda8",
  "%fda9", "%fda10", "%fda11", "%fda12", "%fda13", "%fda14", "%fda15", "%fda16",
  "%fda17", "%fda18", "%fda19", "%fda20", "%fda21", "%fda22", "%fda23","%fda24",
  "%fda25", "%fda26", "%fda27", "%fda28", "%fda29", "%fda30", "%fda31","%fda32"};
static const char *f64ret_names[MAX_RETURN_REGS] = {
  "%fdv1", "%fdv2", "%fdv3", "%fdv4", "%fdv5", "%fdv6", "%fdv7", "%fdv8",
  "%fdv9", "%fdv10", "%fdv11", "%fdv12"};

int main()
{
  ABI_Properties_Begin("nvisa");

  allocatable = Create_Reg_Property("allocatable");
  func_arg = Create_Reg_Property("func_arg");
  func_val = Create_Reg_Property("func_val");
  unused = Create_Reg_Property("unused");
  tid1 = Create_Reg_Property("tid1");
  ntid1 = Create_Reg_Property("ntid1");
  ctaid1 = Create_Reg_Property("ctaid1");
  nctaid1 = Create_Reg_Property("nctaid1");
  tid = Create_Reg_Property("tid");
  ntid = Create_Reg_Property("ntid");
  ctaid = Create_Reg_Property("ctaid");
  nctaid = Create_Reg_Property("nctaid");
  laneid = Create_Reg_Property("laneid");
  warpid = Create_Reg_Property("warpid");
  nwarpid = Create_Reg_Property("nwarpid");
  smid = Create_Reg_Property("smid");
  nsmid = Create_Reg_Property("nsmid");
  gridid = Create_Reg_Property("gridid");
  lanemask = Create_Reg_Property("lanemask");
  pm = Create_Reg_Property("pm");
  clock = Create_Reg_Property("clock");
  clock64 = Create_Reg_Property("clock64");

#if 0 // initially, nvisa has no calling convention
  callee = Create_Reg_Property("callee");
  caller = Create_Reg_Property("caller");
#endif 
#if 0 // initially, nvisa has no stack
  frame_ptr = Create_Reg_Property("frame_ptr");
  stack_ptr = Create_Reg_Property("stack_ptr");
#endif

  ///////////////////////////////////////
  Begin_ABI("nvisa");

  // Would be simpler to put these special regs at front,
  // but more user-friendly to put at end so user sees r1,r2,etc
  INT tid1_index = ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_integer16) 
	- 2; // includes last index
  INT ntid1_index = tid1_index - 3;
  INT ctaid1_index = ntid1_index - 3;
  INT nctaid1_index = ctaid1_index - 3;

  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, tid1_index, "%tid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, tid1_index+1, "%tid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, tid1_index+2, "%tid.z");
  Reg_Property_Range (tid1, ISA_REGISTER_CLASS_integer16, 
	tid1_index, tid1_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ntid1_index, "%ntid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ntid1_index+1, "%ntid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ntid1_index+2, "%ntid.z");
  Reg_Property_Range (ntid1, ISA_REGISTER_CLASS_integer16, 
	ntid1_index, ntid1_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ctaid1_index, "%ctaid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ctaid1_index+1, "%ctaid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, ctaid1_index+2, "%ctaid.z");
  Reg_Property_Range (ctaid1, ISA_REGISTER_CLASS_integer16, 
	ctaid1_index, ctaid1_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, nctaid1_index, "%nctaid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, nctaid1_index+1, "%nctaid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, nctaid1_index+2, "%nctaid.z");
  Reg_Property_Range (nctaid1, ISA_REGISTER_CLASS_integer16, 
	nctaid1_index, nctaid1_index+2);

  INT tid_index = ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_integer) 
	- 2; // includes last index
  INT ntid_index = tid_index - 3;
  INT ctaid_index = ntid_index - 3;
  INT nctaid_index = ctaid_index - 3;
  INT laneid_index = nctaid_index - 1;
  INT warpid_index = laneid_index - 1;
  INT nwarpid_index = warpid_index - 1;
  INT smid_index = nwarpid_index - 1;
  INT nsmid_index = smid_index - 1;
  INT gridid_index = nsmid_index - 1;
  INT lanemask_index = gridid_index - 5;
  INT pm_index = lanemask_index - 4;
  INT clock_index = pm_index - 1;
  INT i32param_index = clock_index - MAX_PARAM_REGS;
  INT i32ret_index = i32param_index - MAX_RETURN_REGS;

  Set_Reg_Name(ISA_REGISTER_CLASS_integer, tid_index, "%tid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, tid_index+1, "%tid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, tid_index+2, "%tid.z");
  Reg_Property_Range (tid, ISA_REGISTER_CLASS_integer, 
	tid_index, tid_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ntid_index, "%ntid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ntid_index+1, "%ntid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ntid_index+2, "%ntid.z");
  Reg_Property_Range (ntid, ISA_REGISTER_CLASS_integer, 
	ntid_index, ntid_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ctaid_index, "%ctaid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ctaid_index+1, "%ctaid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, ctaid_index+2, "%ctaid.z");
  Reg_Property_Range (ctaid, ISA_REGISTER_CLASS_integer, 
	ctaid_index, ctaid_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, nctaid_index, "%nctaid.x");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, nctaid_index+1, "%nctaid.y");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, nctaid_index+2, "%nctaid.z");
  Reg_Property_Range (nctaid, ISA_REGISTER_CLASS_integer, 
	nctaid_index, nctaid_index+2);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, laneid_index, "%laneid");
  Reg_Property (laneid, ISA_REGISTER_CLASS_integer, laneid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, warpid_index, "%warpid");
  Reg_Property (warpid, ISA_REGISTER_CLASS_integer, warpid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, nwarpid_index, "%nwarpid");
  Reg_Property (nwarpid, ISA_REGISTER_CLASS_integer, nwarpid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, smid_index, "%smid");
  Reg_Property (smid, ISA_REGISTER_CLASS_integer, smid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, nsmid_index, "%nsmid");
  Reg_Property (nsmid, ISA_REGISTER_CLASS_integer, nsmid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, gridid_index, "%gridid");
  Reg_Property (gridid, ISA_REGISTER_CLASS_integer, gridid_index, -1);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, lanemask_index, "%lanemask_eq");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, lanemask_index+1, "%lanemask_le");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, lanemask_index+2, "%lanemask_lt");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, lanemask_index+3, "%lanemask_ge");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, lanemask_index+4, "%lanemask_gt");
  Reg_Property_Range (lanemask, ISA_REGISTER_CLASS_integer, 
	lanemask_index, lanemask_index+4);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, pm_index, "%pm0");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, pm_index+1, "%pm1");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, pm_index+2, "%pm2");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, pm_index+3, "%pm3");
  Reg_Property_Range (pm, ISA_REGISTER_CLASS_integer, 
	pm_index, pm_index+3);
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, clock_index, "%clock");
  Reg_Property (clock, ISA_REGISTER_CLASS_integer, clock_index, -1);
  Reg_Names(ISA_REGISTER_CLASS_integer, i32param_index, 
	i32param_index+MAX_PARAM_REGS-1, i32param_names);
  Reg_Property_Range (func_arg, ISA_REGISTER_CLASS_integer, 
	i32param_index, i32param_index+MAX_PARAM_REGS-1);
  Reg_Names(ISA_REGISTER_CLASS_integer, i32ret_index, 
	i32ret_index+MAX_RETURN_REGS-1, i32ret_names);
  Reg_Property_Range (func_val, ISA_REGISTER_CLASS_integer, 
	i32ret_index, i32ret_index+MAX_RETURN_REGS-1);

  INT clock64_index = ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_integer64);
  INT i64param_index = clock64_index - MAX_PARAM_REGS;


  Set_Reg_Name(ISA_REGISTER_CLASS_integer64, clock64_index, "%clock64");
  Reg_Property (clock64, ISA_REGISTER_CLASS_integer64, clock64_index, -1);

  INT i64ret_index = i64param_index - MAX_RETURN_REGS;
  Reg_Names(ISA_REGISTER_CLASS_integer64, i64param_index, 
	i64param_index+MAX_PARAM_REGS-1, i64param_names);
  Reg_Property_Range (func_arg, ISA_REGISTER_CLASS_integer64, 
	i64param_index, i64param_index+MAX_PARAM_REGS-1);
  Reg_Names(ISA_REGISTER_CLASS_integer64, i64ret_index, 
	i64ret_index+MAX_RETURN_REGS-1, i64ret_names);
  Reg_Property_Range (func_val, ISA_REGISTER_CLASS_integer64, 
	i64ret_index, i64ret_index+MAX_RETURN_REGS-1);

  INT f32param_index = 
    ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_float) - MAX_PARAM_REGS;
  INT f32ret_index = f32param_index - MAX_RETURN_REGS;
  Reg_Names(ISA_REGISTER_CLASS_float, f32param_index, 
	f32param_index+MAX_PARAM_REGS-1, f32param_names);
  Reg_Property_Range (func_arg, ISA_REGISTER_CLASS_float, 
	f32param_index, f32param_index+MAX_PARAM_REGS-1);
  Reg_Names(ISA_REGISTER_CLASS_float, f32ret_index, 
	f32ret_index+MAX_RETURN_REGS-1, f32ret_names);
  Reg_Property_Range (func_val, ISA_REGISTER_CLASS_float, 
	f32ret_index, f32ret_index+MAX_RETURN_REGS-1);

  INT f64param_index = 
    ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_float64) - MAX_PARAM_REGS;
  INT f64ret_index = f64param_index - MAX_RETURN_REGS;
  Reg_Names(ISA_REGISTER_CLASS_float64, f64param_index, 
	f64param_index+MAX_PARAM_REGS-1, f64param_names);
  Reg_Property_Range (func_arg, ISA_REGISTER_CLASS_float64, 
	f64param_index, f64param_index+MAX_PARAM_REGS-1);
  Reg_Names(ISA_REGISTER_CLASS_float64, f64ret_index, 
	f64ret_index+MAX_RETURN_REGS-1, f64ret_names);
  Reg_Property_Range (func_val, ISA_REGISTER_CLASS_float64, 
	f64ret_index, f64ret_index+MAX_RETURN_REGS-1);

  // for now, no abi so don't need anything defined
  // except allocatable
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_integer16, 1,
	nctaid1_index - 1); // ends before nctaid1 
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_integer, 1,
	i32ret_index - 1); // ends before retval 
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_integer64, 1,
	i64ret_index - 1);
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_float, 1,
	f32ret_index - 1);
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_float64, 1,
	f64ret_index - 1);
  Reg_Property_Range (allocatable, ISA_REGISTER_CLASS_predicate, 1, 
	ISA_REGISTER_CLASS_Last_Reg(ISA_REGISTER_CLASS_predicate));

  Set_Reg_Name(ISA_REGISTER_CLASS_integer16, 0, "_");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer, 0, "_");
  Set_Reg_Name(ISA_REGISTER_CLASS_integer64, 0, "_");
  Set_Reg_Name(ISA_REGISTER_CLASS_float, 0, "_");
  Set_Reg_Name(ISA_REGISTER_CLASS_float64, 0, "_");
  Reg_Property(unused, ISA_REGISTER_CLASS_integer, 0, -1);
  Reg_Property(unused, ISA_REGISTER_CLASS_integer16, 0, -1);
  Reg_Property(unused, ISA_REGISTER_CLASS_integer64, 0, -1);
  Reg_Property(unused, ISA_REGISTER_CLASS_float, 0, -1);
  Reg_Property(unused, ISA_REGISTER_CLASS_float64, 0, -1);

  ABI_Properties_End();

  return 0;
}
