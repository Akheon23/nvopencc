/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

/*
 * Copyright 2003, 2004, 2005 PathScale, Inc.  All Rights Reserved.
 */

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


/*
 * This defines the ABI subprogram interface,
 * and is used to determine how parameters and results are passed.
 * We have an array of tables, where each table describes the info
 * for one abi.  The array is indexed by the TARGET_ABI enumeration.
 * The register values are the PREG offsets, so these values can be
 * used in WHIRL.
 */

#define TRACE_ENTRY(x)
#define TRACE_EXIT(x)
#define TRACE_EXIT_i(x,i)

#define __STDC_LIMIT_MACROS
#include <stdint.h>
#include <limits.h>
#include "defs.h"
#include "mtypes.h"
#include "errors.h"
#include "erglob.h"
#include "stab.h"
#include "config_targ.h"
#include "targ_sim.h"

#include "targ_sim_body.h"

#define IP0 First_Int32_Preg_Param_Offset
#define LP0 First_Int64_Preg_Param_Offset
#define FP0 First_Float32_Preg_Param_Offset
#define DP0 First_Float64_Preg_Param_Offset
#define IR0 First_Int32_Preg_Return_Offset
#define LR0 First_Int64_Preg_Return_Offset
#define FR0 First_Float32_Preg_Return_Offset
#define DR0 First_Float64_Preg_Return_Offset
#define NP MAX_NUMBER_OF_REGISTER_PARAMETERS
#define NR MAX_NUMBER_OF_REGISTERS_FOR_RETURN

#if (__GNUC__ == 2)
static
#endif /* _LP64 */
SIM SIM_Info[] = {
  /* flags */
  /* int args, int64 args, flt args, dbl args */
  /* int res , int64 res, flt res, dbl res */
  /* int type, int64 type, flt type, dbl type */
  /* save area, formal-area, var ofst */
  /* struct arg, struct res, slink, pic */
  {/* ABI_UNDEF */
    0,
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    0, 0, 0, 0,
    0, 0, 0, 
    0, 0, 0, 0
  },

  { /* ABI_n32 */
    SIM_FLT_AFTER_INT | SIM_COORD_MEM_REG 
    | SIM_REG_STRUCTS | SIM_FLT_RTN_COMPLEX 
    | SIM_FLT_REG_FIELDS | SIM_DBL_REG_FIELDS | SIM_VARARGS_FLOATS ,
    {IP0,IP0+NP-1,1}, {LP0,LP0+NP-1,1}, {FP0,FP0+NP-1,1}, {DP0,DP0+NP-1,1},
    {IR0,IR0+NR-1,1}, {LR0,LR0+NR-1,1}, {FR0,FR0+NR-1,1}, {DR0,DR0+NR-1,1},
    MTYPE_I4, MTYPE_I8, MTYPE_F4, MTYPE_F8,
    0, 0, -64/*TODO*/, 
    NP*64, NR*64, 0, 0
  },

  { /* ABI_n64 */
    SIM_FLT_AFTER_INT | SIM_COORD_MEM_REG 
    | SIM_REG_STRUCTS | SIM_FLT_RTN_COMPLEX 
    | SIM_FLT_REG_FIELDS | SIM_DBL_REG_FIELDS | SIM_VARARGS_FLOATS ,
    {IP0,IP0+NP-1,1}, {LP0,LP0+NP-1,1}, {FP0,FP0+NP-1,1}, {DP0,DP0+NP-1,1},
    {IR0,IR0+NR-1,1}, {LR0,LR0+NR-1,1}, {FR0,FR0+NR-1,1}, {DR0,DR0+NR-1,1},
    MTYPE_I4, MTYPE_I8, MTYPE_F4, MTYPE_F8,
    0, 64/*TODO*/, -64/*TODO*/, 
    NP*64, NR*64, 0, 0
  },
  { /* ABI_w64 */
    SIM_FLT_AFTER_INT | SIM_COORD_MEM_REG 
    | SIM_REG_STRUCTS | SIM_FLT_RTN_COMPLEX 
    | SIM_FLT_REG_FIELDS | SIM_DBL_REG_FIELDS | SIM_VARARGS_FLOATS ,
    {IP0,IP0+NP-1,1}, {LP0,LP0+NP-1,1}, {FP0,FP0+NP-1,1}, {DP0,DP0+NP-1,1},
    {IR0,IR0+NR-1,1}, {LR0,LR0+NR-1,1}, {FR0,FR0+NR-1,1}, {DR0,DR0+NR-1,1},
    MTYPE_I4, MTYPE_I8, MTYPE_F4, MTYPE_F8,
    0, 64/*TODO*/, -64/*TODO*/, 
    NP*64, NR*64, 0, 0
  }
};

/* return whether preg is a return preg */
extern BOOL 
Is_Return_Preg (PREG_NUM preg)
{
	return (preg >= First_Int32_Preg_Return_Offset &&
	        preg <= Last_Int32_Preg_Return_Offset) || 
	       (preg >= First_Float32_Preg_Return_Offset && 
		preg <= Last_Float32_Preg_Return_Offset) ||
	       (preg >= First_Int64_Preg_Return_Offset && 
		preg <= Last_Int64_Preg_Return_Offset) ||
	       (preg >= First_Float64_Preg_Return_Offset && 
		preg <= Last_Float64_Preg_Return_Offset);
}

/* return whether preg is an output preg */
extern BOOL 
Is_Int_Output_Preg (PREG_NUM preg)
{
  Fail_FmtAssertion (
    ("Is_Int_Output_Preg not applicable to x8664 targets"));
  return FALSE;
}

/* return whether preg is an input preg */
extern BOOL
Is_Formal_Preg (PREG_NUM preg)
{
	return (preg >= First_Int32_Preg_Param_Offset && 
		preg <= Last_Int32_Preg_Param_Offset) || 
	       (preg >= First_Float32_Preg_Param_Offset && 
		preg <= Last_Float32_Preg_Param_Offset) ||
	       (preg >= First_Int64_Preg_Param_Offset && 
		preg <= Last_Int64_Preg_Param_Offset) || 
	       (preg >= First_Float64_Preg_Param_Offset && 
		preg <= Last_Float64_Preg_Param_Offset);
}


/* This routine figures out the mtypes of the return registers that are 
 * used for returning an object of the given type.
 * This returns the mtypes to use for the CALL opcode in high-level whirl.
 * This means that returns of simulated objects, like FQ, are just shown
 * as returning FQ, which will later be split into F8F8.
 * However, structures that return in registers are specified explicitly.
 */
/*ARGSUSED*/
extern void
Get_Return_Mtypes (
  TY_IDX rtype,		/* The result type */
  Mtype_Return_Level level,	/* whether to lower the mtypes */
  TYPE_ID *mreg1,	/* out: mtype for result register 1 */
  TYPE_ID *mreg2)	/* out: mtype for result register 2 */
{
  Fail_FmtAssertion (
    ("Get_Return_Mtypes should not be invoked; invoke Get_Return_Info instead"));
}

/* This routine figures out which return registers are to be used
 * for returning an object with the given mtypes.
 * It is assumed that the mtypes will be determined by calling
 * Get_Return_Mtypes.
 */
/*ARGSUSED*/
extern void
Get_Return_Pregs (
  TYPE_ID mreg1,	/* in:  mtype for result register 1 */
  TYPE_ID mreg2,	/* in:  mtype for result register 2 */
  PREG_NUM *rreg1,	/* out: result register 1 */
  PREG_NUM *rreg2)	/* out: result register 2 */
{
  Fail_FmtAssertion (
    ("Get_Return_Pregs should not be invoked; invoke Get_Return_Info instead"));
}

// Input is struct ty, field_id, and array index;
// Output is mtype and offset of each element.
// Return mtype_v if no more fields
static void
Get_Field_Element (TY_IDX struct_ty, UINT field_id, UINT index, 
                   TYPE_ID *mtype, UINT *offset)
{
      UINT cfi = 0;
      FLD_HANDLE fld;
      fld = FLD_get_to_field (struct_ty, field_id, cfi);
      if (fld.Is_Null()) {
        // no more fields
        *mtype = MTYPE_V;
        *offset = 0;
      }
      else if (Is_Simple_Type(FLD_type(fld))) {
        *mtype = TY_mtype(FLD_type(fld));
        *offset = FLD_ofst(fld);
      }
      else if (TY_kind(FLD_type(fld)) == KIND_ARRAY) {
        // one sub-field for each element in nested array
        TY_IDX ety = TY_etype(FLD_type(fld));
        if ( ! Is_Simple_Type(ety)) {
          // nested struct inside array, need to find each sub-field
          // could call get_to_field on this struct, 
          // but then have to keep fieldid for every nested struct that 
          // is inside an array (unlimited possibilities).
          // Instead we avoid this situation earlier when count fields.
          DevWarn("NYI");
        }
        *mtype = TY_mtype(ety);
        *offset = FLD_ofst(fld) + ((index) * TY_size(ety));
      }
      else {
        FmtAssert(FALSE, ("unexpected kind"));
      }
}

// Get field_id for first field in struct,
// skipping field_ids that are for nested structs.
static void
Get_First_Field_Element (TY_IDX struct_ty, UINT *field_id, UINT *index)
{
  UINT cfi = 0;
  FLD_HANDLE fld;
  *field_id = 1;
  *index = 0;
  fld = FLD_get_to_field (struct_ty, *field_id, cfi);
  while (!fld.Is_Null() && TY_kind(FLD_type(fld)) == KIND_STRUCT) {
        // recurse to sub-struct
        ++(*field_id);
        cfi = 0;
        fld = FLD_get_to_field (struct_ty, *field_id, cfi);
  }
}

// We want to iterate through each field in a struct, 
// for which we can use field_id, 
// but want to skip field_ids that are for nested structs.
// Also need to handle nested arrays (sub-fields).
static void
Get_Next_Field_Element (TY_IDX struct_ty, UINT *field_id, UINT *index, 
                        UINT *nest_offset)
{
      UINT cfi = 0;
      FLD_HANDLE fld;
      fld = FLD_get_to_field (struct_ty, *field_id, cfi);
      if (TY_kind(FLD_type(fld)) == KIND_ARRAY) {
        // one sub-field for each element in nested array
        TY_IDX ety = TY_etype(FLD_type(fld));
        UINT asize = TY_size(FLD_type(fld)) / TY_size(ety);
        ++(*index);
        if (*index < asize) {
          return;
        } 
        else {
          // reached end of array, so go to next field
          *index = 0;
        }
      }
      if (FLD_last_field(fld)) {
        *nest_offset = 0; // pop the nest_offset
      }
      ++(*field_id);
      cfi = 0;
      fld = FLD_get_to_field (struct_ty, *field_id, cfi);
      while (!fld.Is_Null() && TY_kind(FLD_type(fld)) == KIND_STRUCT) {
        // recurse to sub-struct.
        // new sub-field will be offset from sub-struct, 
        // so add in offset of sub-struct.
        // Earlier num_fields check avoids case of struct in struct in struct,
        // as we don't have easy way of popping such a stack.
        FmtAssert(*nest_offset == 0, ("can't handle structs nested more than 1 deep"));
        *nest_offset = FLD_ofst(fld);
        ++(*field_id);
        cfi = 0;
        fld = FLD_get_to_field (struct_ty, *field_id, cfi);
      }
}

static UINT
Num_Fields_In_Struct (TY_IDX struct_ty, BOOL nested_struct = FALSE)
{
  UINT count = 0;
  UINT tcount;
  FLD_ITER fld_iter = Make_fld_iter(TY_fld(struct_ty));
  while (TRUE) {
    FLD_HANDLE fld (fld_iter);
    switch (TY_kind(FLD_type(fld))) {
    case KIND_STRUCT:
      if (nested_struct) {
        // give up if struct nested in struct nested in struct;
        // else have to figure out how to pop offsets
        return UINT_MAX;
      }
      tcount = Num_Fields_In_Struct(FLD_type(fld), TRUE); 
      if (tcount == UINT_MAX) 
        return UINT_MAX;
      else
        count += tcount;
      break;
    case KIND_ARRAY:
      // one sub-field for each element in nested array
      if ( ! Is_Simple_Type(TY_etype(FLD_type(fld)))) {
        // give up if struct nested in array in struct;
        // else have to figure out how to recurse through fields in Get_Field.
        return UINT_MAX;
      }
      count += TY_size(FLD_type(fld)) / TY_size(TY_etype(FLD_type(fld)));
      break;
    default:
      ++count;
    }
    if (FLD_last_field(fld))
      return count;
    else
      ++fld_iter;
  }
}

static Preg_Range
Get_Return_Mtype_Preg_Range (TYPE_ID mtype)
{
  Preg_Range prange;
  switch (mtype) {
  case MTYPE_F4:
    return SIM_INFO.flt_results;
  case MTYPE_F8:
    return SIM_INFO.dbl_results;
  case MTYPE_I4:
  case MTYPE_U4:
  case MTYPE_I2:
  case MTYPE_U2:
  case MTYPE_I1:
  case MTYPE_U1:
    return SIM_INFO.int_results;
  case MTYPE_I8:
  case MTYPE_U8:
    return SIM_INFO.int64_results;
  default:
    return SIM_Info[0].int_results; // empty
  }
}

RETURN_INFO
Get_Return_Info(TY_IDX rtype, Mtype_Return_Level level)
{
  TYPE_ID mtype = TY_mtype (rtype);
  RETURN_INFO info;
  INT32 i; 
  INT64 size;

  info.return_via_first_arg = FALSE;
  info.offset[0] = 0;

  switch (mtype) {

    case MTYPE_UNKNOWN:

      // FORTRAN character array
      info.count = 0;
      // f90 already has made visible the arg for arrays
      // info.return_via_first_arg = TRUE;
      break;

    case MTYPE_FQ:

      info.count = 1;
      info.mtype[0] = mtype;
      info.preg[0] = PR_first_reg(SIM_INFO.flt_results);
      break;

    case MTYPE_V:

      info.count = 0;
      break;

    case MTYPE_I8:
    case MTYPE_U8:
    case MTYPE_A8:
      info.count = 1;
      info.mtype [0] = mtype;
      info.preg  [0] = PR_first_reg(SIM_INFO.int64_results);
      break;

    case MTYPE_I1:
    case MTYPE_I2:
    case MTYPE_I4:
    case MTYPE_U1:
    case MTYPE_U2:
    case MTYPE_U4:
    case MTYPE_A4:

      info.count = 1;
      info.mtype [0] = mtype;
      info.preg  [0] = PR_first_reg(SIM_INFO.int_results);
      break;

    case MTYPE_F4:
#ifdef TARG_SUPPORTS_VECTORS
    case MTYPE_V16F4:
#endif
      info.count = 1;
      info.mtype [0] = mtype;
      info.preg  [0] = PR_first_reg(SIM_INFO.flt_results);
      break;
    case MTYPE_F8:
#ifdef TARG_SUPPORTS_VECTORS
    case MTYPE_V16F8:
#endif
      info.count = 1;
      info.mtype [0] = mtype;
      info.preg  [0] = PR_first_reg(SIM_INFO.dbl_results);
      break;

    case MTYPE_C4:
      if (Is_Target_32bit()) {
	/* Under -m32 for C and C++, if type "float _Complex" is passed as argument,
	   there is no need to introduce a fake first parameter; and if it is a return
	   value, the real part and the imaginary part are set aside in
	   %eax and %edx, respectively.    (bug#2707)
	 */
	if( PU_c_lang(Get_Current_PU()) ||
	    PU_cxx_lang(Get_Current_PU()) ){

	  if( level == Use_Simulated ){
	    info.count = 1;
	    info.mtype[0] = mtype;
	    info.preg[0] = PR_first_reg(SIM_INFO.flt_results);

	  } else {
	    info.count = 2;
	    info.mtype[0] = info.mtype[1] = SIM_INFO.int_type;
	    info.preg[0] = PR_first_reg(SIM_INFO.int_results);
	    info.preg[1] = info.preg[0] + PR_skip_value(SIM_INFO.int_results);
	  }

	} else {
	  info.count = 0;
	  info.return_via_first_arg = TRUE;
	}
      } else if( level == Use_Simulated ){
	info.count     = 1;
	info.mtype [0] = mtype;
	info.preg  [0] = PR_first_reg(SIM_INFO.flt_results);

      } else {
	// For bug:143
	
        info.count     = 2;
        info.mtype [0] = Mtype_complex_to_real(mtype);
        info.mtype [1] = Mtype_complex_to_real(mtype);
        info.preg  [0] = PR_first_reg(SIM_INFO.flt_results);
        info.preg  [1] =   PR_first_reg(SIM_INFO.flt_results)
                         + PR_skip_value(SIM_INFO.flt_results);
      }

      break;

    case MTYPE_C8:
      if (Is_Target_32bit()) {
        info.count = 0;
        info.return_via_first_arg = TRUE;
      } else if (level == Use_Simulated) {

        info.count     = 1;
        info.mtype [0] = mtype;
        info.preg  [0] = PR_first_reg(SIM_INFO.dbl_results);
      }

      else {

        info.count     = 2;
        info.mtype [0] = Mtype_complex_to_real(mtype);
        info.mtype [1] = Mtype_complex_to_real(mtype);
        info.preg  [0] = PR_first_reg(SIM_INFO.dbl_results);
        info.preg  [1] =   PR_first_reg(SIM_INFO.dbl_results)
                         + PR_skip_value(SIM_INFO.dbl_results);
      }
      break;

    case MTYPE_CQ:
      if (Is_Target_32bit()) {
        info.count = 0;
        info.return_via_first_arg = TRUE;
      }
      else if (level == Use_Simulated) {

        info.count     = 1;
        info.mtype [0] = mtype;
        info.preg  [0] = PR_first_reg(SIM_INFO.flt_results);
      }

      else {

        info.count     = 2;
        info.mtype [0] = Mtype_complex_to_real(mtype);
        info.mtype [1] = Mtype_complex_to_real(mtype);
        info.preg  [0] = PR_first_reg(SIM_INFO.flt_results);
        info.preg  [1] =   PR_first_reg(SIM_INFO.flt_results)
                         + PR_skip_value(SIM_INFO.flt_results);
      }
      break;

    case MTYPE_M:

      info.count = 0;

      size = TY_size(Ty_Table[rtype]);
      if (size == 0)
	break;

      if (size*8 > SIM_INFO.max_struct_result)
        info.return_via_first_arg = TRUE;
      else if (Num_Fields_In_Struct(rtype) > MAX_NUMBER_OF_REGISTERS_FOR_RETURN)
        info.return_via_first_arg = TRUE;
      else if (TY_can_be_vector(rtype)) 
      {
        Preg_Range prange;
        TYPE_ID etype = TY_mtype(TY_vector_elem_ty(rtype));
        prange = Get_Return_Mtype_Preg_Range (etype);
        FmtAssert(PR_skip_value(prange) != 0, ("unexpected vector mtype"));
        PREG_NUM preg = PR_first_reg(prange);
        info.count = TY_vector_count(rtype);
        for (i = 0; i < info.count; i++) {
          info.mtype [i] = etype;
          info.preg  [i] = preg;
          preg += PR_skip_value(prange);
          info.offset[i] = i*(MTYPE_RegisterSize(etype));
        }
      }
      else if (Target_ISA >= TARGET_ISA_compute_20) {
        // handle arbitrary structures
        Preg_Range prange;
        TYPE_ID ftype;
        UINT offset;
        UINT struct_offset = 0;
        UINT fid;
        UINT index;
        info.count = Num_Fields_In_Struct(rtype);
        Get_First_Field_Element(rtype, &fid, &index);
        for (i = 0; i < info.count; i++) {
          Get_Field_Element (rtype, fid, index, &ftype, &offset);
          prange = Get_Return_Mtype_Preg_Range (ftype);
          FmtAssert(PR_skip_value(prange) != 0, ("unexpected field mtype"));
          info.mtype [i] = ftype;
          info.preg  [i] = PR_first_reg(prange) + i;
          info.offset[i] = (INT32) (struct_offset + offset);
          Get_Next_Field_Element(rtype, &fid, &index, &struct_offset);
        }
      }
      else {
        info.return_via_first_arg = TRUE;
      }
      break;

    default:

      info.count = 0;
      Fail_FmtAssertion ("Invalid return mtype %s encountered",
                         (MTYPE_name(mtype)));
      break;
  } /* switch (mtype) */

  for (i = info.count; i < MAX_NUMBER_OF_REGISTERS_FOR_RETURN; i++) {

    info.mtype [i] = MTYPE_V;
    info.preg  [i] = 0;
    info.offset[i] = 0;
  }

  return info;
} /* Get_Return_Info */

static Preg_Range
Get_Arg_Mtype_Preg_Range (TYPE_ID mtype)
{
  Preg_Range prange;
  switch (mtype) {
  case MTYPE_F4:
    return SIM_INFO.flt_args;
  case MTYPE_F8:
    return SIM_INFO.dbl_args;
  case MTYPE_I4:
  case MTYPE_U4:
  case MTYPE_I2:
  case MTYPE_U2:
  case MTYPE_I1:
  case MTYPE_U1:
    return SIM_INFO.int_args;
  case MTYPE_I8:
  case MTYPE_U8:
    return SIM_INFO.int64_args;
  default:
    return SIM_Info[0].int_args; // empty
  }
}

static PLOC
Setup_Parameter_Locations (TY_IDX pu_type)
{
    static PLOC plocNULL;

    TY_IDX ret_type = (TY_kind(pu_type) == KIND_FUNCTION ? TY_ret_type(pu_type)
			: pu_type);
    RETURN_INFO info = Get_Return_Info (ret_type, No_Simulated);
    if (TY_is_varargs (pu_type)) {
	// find last fixed parameter
	TYLIST_IDX idx = TY_tylist (pu_type);
	Last_Fixed_Param = -1;
	for (++idx; Tylist_Table[idx] != 0; ++idx)
	    ++Last_Fixed_Param;
	// old style varargs is counting va_alist and should not
	if ( ! TY_has_prototype(pu_type))
	    --Last_Fixed_Param;
	// account for functions returning to first parameter
	if (TY_return_to_param (pu_type))
	    ++Last_Fixed_Param;
    } else
	Last_Fixed_Param = INT_MAX;

    Current_Param_Num = -1;		// count all parameters
    Last_Param_Offset = 0;
    return plocNULL;
} // Setup_Parameter_Locations


static PLOC
Get_Parameter_Location (TY_IDX ty, BOOL is_output)
{
    PLOC ploc;				// return location

    ploc.reg = 0;
    ploc.start_offset = Last_Param_Offset;
    ploc.size = 0;
    ploc.vararg_reg = 0;               // to silence purify
    if (TY_kind (ty) == KIND_VOID) {
	return ploc;
    }

    /* check for array case where fe doesn't fill in right btype */
    TYPE_ID pmtype = Fix_TY_mtype (ty);	/* Target type */
    ploc.size = MTYPE_RegisterSize(pmtype);

    ++Current_Param_Num;

    INT rpad = 0;			/* padding to right of object */

    // do need to pad when start offset not already aligned
    rpad = (ploc.start_offset % TY_align(ty));

    switch (pmtype) {
	
    case MTYPE_I1:
    case MTYPE_U1:
    case MTYPE_I2:
    case MTYPE_U2:
    case MTYPE_I4:
    case MTYPE_U4:
    case MTYPE_A4:
	ploc.reg = PR_first_reg(SIM_INFO.int_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.int_args)) 
	  ploc.reg = 0;
	break;

    case MTYPE_I8:
    case MTYPE_U8:
    case MTYPE_A8:
	ploc.reg = PR_first_reg(SIM_INFO.int64_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.int64_args)) 
	  ploc.reg = 0;
	break;
	
#ifdef TARG_SUPPORTS_VECTORS
    case MTYPE_V16I4:
    case MTYPE_V16F4:
#endif
    case MTYPE_F4:
	ploc.reg = PR_first_reg(SIM_INFO.flt_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.flt_args)) {
	  ploc.reg = 0;
	  /* ptx param space has no padding */
	  // if( Is_Target_64bit() )
	  //   rpad = MTYPE_RegisterSize(SIM_INFO.flt_type) - ploc.size;
	}
	break;
#ifdef TARG_SUPPORTS_VECTORS
    case MTYPE_V16F8:
#endif
    case MTYPE_F8:
	ploc.reg = PR_first_reg(SIM_INFO.dbl_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.dbl_args)) {
	  ploc.reg = 0;
	}
	break;

#ifdef TARG_SUPPORTS_VECTORS
    case MTYPE_V8I1:
    case MTYPE_V8I2:
    case MTYPE_V8I4:
      ploc.reg = 0; // pass in memory
      break;
#endif

    case MTYPE_CQ:
    case MTYPE_FQ:
      ploc.reg = 0;  /* pass in memory */
      break;

    case MTYPE_C4:
        ++Current_Param_Num;
	ploc.reg = PR_first_reg(SIM_INFO.flt_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.flt_args)) 
	  ploc.reg = 0;
	break;
	
    case MTYPE_C8:
        ++Current_Param_Num;
	ploc.reg = PR_first_reg(SIM_INFO.dbl_args) + Current_Param_Num;
	if (ploc.reg > PR_last_reg(SIM_INFO.dbl_args)) {
          --Current_Param_Num;
	  ploc.reg = 0;	/* pass in memory */
	}
	else {
          ++Current_Param_Num;
	}
	break;

    case MTYPE_M:
	ploc.size = TY_size (ty);
	/* default is to pass whole struct in memory */
	ploc.reg = 0;
        if (ploc.size*8 > SIM_INFO.max_struct_size)
          break; // too big

        if (TY_can_be_vector(ty) 
          && (Current_Param_Num + TY_vector_count(ty)) 
            <= MAX_NUMBER_OF_REGISTER_PARAMETERS)
        {
          Preg_Range prange = 
            Get_Arg_Mtype_Preg_Range (TY_mtype(TY_vector_elem_ty(ty)));
          FmtAssert(PR_skip_value(prange) != 0, ("unexpected vector mtype"));
          ploc.reg = PR_first_reg(prange) + Current_Param_Num;
          Current_Param_Num += TY_vector_count(ty) - 1;
        }
        else if (Target_ISA >= TARGET_ISA_compute_20) {
          // pass any struct in regs
          Preg_Range prange;
          UINT psize = Num_Fields_In_Struct (ty);
          UINT fid;
          UINT index;
          TYPE_ID ftype;
          UINT offset;
          if (psize == UINT_MAX 
           || Current_Param_Num + psize > MAX_NUMBER_OF_REGISTER_PARAMETERS)
          {
            break; // too big
          }
          Get_First_Field_Element (ty, &fid, &index);
          Get_Field_Element (ty, fid, index, &ftype, &offset);
          prange = Get_Arg_Mtype_Preg_Range (ftype);
          FmtAssert(PR_skip_value(prange) != 0, ("unexpected field mtype"));
          ploc.reg = PR_first_reg(prange) + Current_Param_Num;
          Current_Param_Num += psize - 1;
        }
	break;
	
    default:
	FmtAssert (FALSE, ("Get_Parameter_Location:  mtype %s",
			   MTYPE_name(pmtype)));
    }
    if (ploc.reg == 0)
      Last_Param_Offset = ploc.start_offset + ploc.size + rpad;
    return ploc;
} // Get_Parameter_Location

struct PSTRUCT {
  BOOL	first_call;
  BOOL	is_vector;
  TYPE_ID fldtype;
  UINT field_id;
  UINT eindex;
  TY_IDX struct_ty;
  UINT	nest_offset; // offset of nested struct
  INT64	size;
  INT64	offset;
};

struct PSTRUCT pstruct;

static void
Setup_Struct_Parameter_Locations (TY_IDX struct_ty)
{
  pstruct.first_call = TRUE;
  pstruct.field_id = 0;
  pstruct.eindex = 0;
  pstruct.nest_offset = 0;
  pstruct.struct_ty = struct_ty;
  pstruct.size = TY_size(struct_ty);
  if (TY_can_be_vector(struct_ty)) {
    pstruct.is_vector = TRUE;
    pstruct.fldtype = TY_mtype(TY_vector_elem_ty(struct_ty));
  }
  else {
    UINT offset;
    pstruct.is_vector = FALSE;
    Get_First_Field_Element (struct_ty, &pstruct.field_id, &pstruct.eindex);
    Get_Field_Element (struct_ty, pstruct.field_id, pstruct.eindex, 
      &pstruct.fldtype, &offset);
  }
}

static PLOC 
Get_Struct_Parameter_Location (PLOC prev)
{
  PLOC next;
  if (pstruct.first_call) {
    pstruct.first_call = FALSE;
    PLOC_offset(next) = PLOC_offset(prev);
    pstruct.offset = PLOC_offset(prev);
    PLOC_reg(next) = PLOC_reg(prev);
    if (PLOC_reg(prev) == 0) // all on stack
      PLOC_size(next) = PLOC_size(prev);
    else
      PLOC_size(next) = MTYPE_RegisterSize(pstruct.fldtype);
  }
  else {
    if (pstruct.is_vector) {
      PLOC_reg(next) = PLOC_reg(prev) + 1;
      PLOC_size(next) = MTYPE_RegisterSize(pstruct.fldtype);
      PLOC_offset(next) = PLOC_offset(prev) + PLOC_size(prev);
    }
    else {
      // should be next numbered param for type of field
      Preg_Range prange;
      INT pnum;
      UINT offset;
      Get_Field_Element (pstruct.struct_ty, pstruct.field_id, pstruct.eindex, 
        &pstruct.fldtype, &offset);
      prange = Get_Arg_Mtype_Preg_Range (pstruct.fldtype);
      pnum = PLOC_reg(prev) - PR_first_reg(prange); // num of prev param
      Get_Next_Field_Element (pstruct.struct_ty, 
        &pstruct.field_id, &pstruct.eindex, &pstruct.nest_offset);
      Get_Field_Element (pstruct.struct_ty, pstruct.field_id, pstruct.eindex, 
        &pstruct.fldtype, &offset);
      if (pstruct.fldtype == MTYPE_V) {
        // no more fields
        PLOC_reg(next) = 0;
        PLOC_size(next) = 0;
      }
      else {
        prange = Get_Arg_Mtype_Preg_Range (pstruct.fldtype);
        PLOC_reg(next) = PR_first_reg(prange) + pnum + 1;
        PLOC_size(next) = MTYPE_RegisterSize(pstruct.fldtype);
        PLOC_offset(next) = pstruct.offset + pstruct.nest_offset + offset;
      }
    }
  }
  if (PLOC_reg(prev) == 0)
    PLOC_reg(next) = 0;
  if (PLOC_offset(next) >= pstruct.offset + pstruct.size) {
    PLOC_size(next) = 0;
    return next;
  }

  return next;
} // Get_Struct_Parameter_Location


/* Iterate over vararg non-fixed parameters */
static PLOC
Get_Vararg_Parameter_Location (PLOC prev)
{
  FmtAssert(FALSE, ("varargs not supported"));
}

BOOL Is_Caller_Save_GP;  /* whether GP is caller-save */

INT Formal_Save_Area_Size = 0;
INT Stack_Offset_Adjustment = 0;

extern void 
Init_Targ_Sim (void)
{
	Is_Caller_Save_GP = SIM_caller_save_gp;
}

