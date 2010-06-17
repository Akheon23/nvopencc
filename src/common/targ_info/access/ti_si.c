/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
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

/* Move the definitions of functions from ti_si.h to ti_si.c
 * to avoid multiple definitions when static build. */

#include "ti_si.h"

inline SI_BAD_II_SET SI_BAD_II_SET_Union( SI_BAD_II_SET s1, SI_BAD_II_SET s2 )
{
  SI_BAD_II_SET the_union;

  the_union.dw[0] = s1.dw[0] | s2.dw[0];
  the_union.dw[1] = s1.dw[1] | s2.dw[1];

  return the_union;
}

inline INT SI_BAD_II_SET_MemberP( SI_BAD_II_SET s, UINT i )
{
  UINT bitnum = i - 1;

  if ( bitnum > (UINT)SI_BAD_II_SET_MAX ) return 0;

  return (s.dw[bitnum / 64]  & (1ULL << (bitnum % 64))) != 0;
}

inline SI_BAD_II_SET SI_BAD_II_SET_Empty( void )
{
  const SI_BAD_II_SET empty_set = {{0,0}};

  return empty_set;
}

/****************************************************************************
 ****************************************************************************/

inline const char* SI_RESOURCE_Name( SI_RESOURCE* res )
{
  return res->name;
}

inline UINT SI_RESOURCE_Id( SI_RESOURCE* res )
{
  return res->id;
}

inline UINT SI_RESOURCE_Avail_Per_Cycle( SI_RESOURCE* res )
{
  return res->avail_per_cycle;
}

inline UINT SI_RESOURCE_Word_Index( SI_RESOURCE* res )
{
  return res->word_index;
}

inline UINT SI_RESOURCE_Bit_Index( SI_RESOURCE* res )
{
  return res->bit_index;
}

extern const INT SI_resource_count;

extern SI_RESOURCE* const SI_resources[];

inline const char* SI_RESOURCE_ID_Name( SI_RESOURCE_ID id )
{
  return SI_RESOURCE_Name(SI_resources[id]);
}

inline UINT SI_RESOURCE_ID_Avail_Per_Cycle( SI_RESOURCE_ID id )
{
  return SI_RESOURCE_Avail_Per_Cycle(SI_resources[id]);
}

/****************************************************************************
 ****************************************************************************/

inline SI_RESOURCE_ID_SET SI_RESOURCE_ID_SET_Universe(void)
{
  return    (SI_RESOURCE_ID_SET)-1
	 >> (sizeof(SI_RESOURCE_ID_SET) * 8 - SI_resource_count);
}

inline SI_RESOURCE_ID_SET SI_RESOURCE_ID_SET_Empty(void)
{
  return (SI_RESOURCE_ID_SET)0;
}

inline SI_RESOURCE_ID_SET
SI_RESOURCE_ID_SET_Intersection( SI_RESOURCE_ID_SET s0,
                                 SI_RESOURCE_ID_SET s1 )
{
  return s0 & s1;
}

inline INT
SI_RESOURCE_ID_SET_Intersection_Non_Empty( SI_RESOURCE_ID_SET s0,
                                           SI_RESOURCE_ID_SET s1 )
{
  return (s0 & s1) != (SI_RESOURCE_ID_SET)0;
}

inline INT
SI_RESOURCE_ID_SET_Intersection4_Non_Empty( SI_RESOURCE_ID_SET s0,
                                            SI_RESOURCE_ID_SET s1,
                                            SI_RESOURCE_ID_SET s2,
                                            SI_RESOURCE_ID_SET s3 )
{
  return (s0 & s1 & s2 & s3) != (SI_RESOURCE_ID_SET)0;
}

inline SI_RESOURCE_ID_SET
SI_RESOURCE_ID_SET_Complement( SI_RESOURCE_ID_SET s )
{
  return (~s) & SI_RESOURCE_ID_SET_Universe();
}

/****************************************************************************
 ****************************************************************************/

inline SI_RRW SI_RRW_Initial(void)
{
  return SI_RRW_initializer;
}

inline SI_RRW SI_RRW_Reserve( SI_RRW table, SI_RRW requirement )
{
  return table + requirement;
}

inline SI_RRW SI_RRW_Has_Overuse( SI_RRW word_with_reservations )
{
  return (word_with_reservations & SI_RRW_overuse_mask) != 0;
}

inline SI_RRW SI_RRW_Unreserve( SI_RRW table, SI_RRW requirement )
{
  return table - requirement;
}

/****************************************************************************
 ****************************************************************************/

inline const char* SI_ISSUE_SLOT_Name( SI_ISSUE_SLOT* slot )
{
  return slot->name;
}

inline INT SI_ISSUE_SLOT_Skew( SI_ISSUE_SLOT* slot )
{
  return slot->skew;
}

inline INT SI_ISSUE_SLOT_Avail_Per_Cycle( SI_ISSUE_SLOT* slot )
{
  return slot->avail_per_cycle;
}

inline INT SI_ISSUE_SLOT_Count(void)
{
  return SI_issue_slot_count;
}

inline SI_ISSUE_SLOT* SI_Ith_Issue_Slot( UINT i )
{
  return SI_issue_slots[i];
}

/****************************************************************************
 ****************************************************************************/

inline SI_RESOURCE*
SI_RESOURCE_TOTAL_Resource( SI_RESOURCE_TOTAL* pair )
{
  return pair->resource;
}

inline SI_RESOURCE_ID SI_RESOURCE_TOTAL_Resource_Id( SI_RESOURCE_TOTAL* pair )
{
  return SI_RESOURCE_Id(SI_RESOURCE_TOTAL_Resource(pair));
}

inline UINT SI_RESOURCE_TOTAL_Avail_Per_Cycle(SI_RESOURCE_TOTAL* pair)
{
  return SI_RESOURCE_Avail_Per_Cycle(SI_RESOURCE_TOTAL_Resource(pair));
}

inline INT SI_RESOURCE_TOTAL_Total_Used( SI_RESOURCE_TOTAL* pair )
{
  return pair->total_used;
}

/****************************************************************************
 ****************************************************************************/
inline UINT SI_RR_Length( SI_RR req )
{
  return (INT) req[0];
}

inline SI_RRW SI_RR_Cycle_RRW( SI_RR req, UINT cycle )
{
  /* TODO: make this compilable with and without defs.h 
  assert(cycle <= req[0]);
  */
  return req[cycle+1];
}

/****************************************************************************
 ****************************************************************************/
inline const char* TSI_Name( TOP top )
{
  return SI_top_si[(INT) top]->name;
}

inline SI_ID TSI_Id( TOP top )
{
  return SI_top_si[top]->id;
}

inline INT
TSI_Operand_Access_Time( TOP top, INT operand_index )
{
  return SI_top_si[(INT) top]->operand_access_times[operand_index];
}

inline INT
TSI_Result_Available_Time( TOP top, INT result_index )
{
  return SI_top_si[(INT) top]->result_available_times[result_index];
}

inline INT
TSI_Load_Access_Time( TOP top )
{
  return SI_top_si[(INT) top]->load_access_time;
}

inline INT
TSI_Last_Issue_Cycle( TOP top )
{
  return SI_top_si[(INT) top]->last_issue_cycle;
}

inline INT
TSI_Store_Available_Time( TOP top )
{
  return SI_top_si[(INT) top]->store_available_time;
}

inline SI_RR TSI_Resource_Requirement( TOP top )
{
  return SI_top_si[(INT) top]->rr;
}

inline SI_BAD_II_SET TSI_Bad_IIs( TOP top )
{
  return SI_top_si[(INT) top]->bad_iis;
}

inline SI_RR TSI_II_Resource_Requirement( TOP top, INT ii )
{
  SI* const info = SI_top_si[(INT) top];

  if ( ii > info->ii_info_size ) return info->rr;

  return info->ii_rr[ii - 1];
}

inline const SI_RESOURCE_ID_SET*
TSI_II_Cycle_Resource_Ids_Used( TOP opcode, INT ii )
{
  SI* const info = SI_top_si[(INT)opcode];
  if ( ii > info->ii_info_size ) return info->resources_used;

  return info->ii_resources_used[ii - 1];
}

inline UINT TSI_Valid_Issue_Slot_Count( TOP top )
{
  return SI_top_si[(INT) top]->valid_issue_slot_count;
}

inline SI_ISSUE_SLOT* TSI_Valid_Issue_Slots( TOP top, UINT i )
{
  return SI_top_si[(INT) top]->valid_issue_slots[i];
}

inline UINT TSI_Resource_Total_Vector_Size( TOP top )
{
  return SI_top_si[(INT) top]->resource_total_vector_size;
}

inline SI_RESOURCE_TOTAL* TSI_Resource_Total_Vector( TOP top )
{
  return SI_top_si[(INT) top]->resource_total_vector;
}

inline INT TSI_Write_Write_Interlock( TOP top )
{
  return SI_top_si[(INT) top]->write_write_interlock;
}

/****************************************************************************
 ****************************************************************************/
inline const SI_RESOURCE_ID_SET*
SI_ID_II_Cycle_Resource_Ids_Used( SI_ID id, INT ii )
{
  SI* const info = SI_ID_si[id];
  if ( ii > info->ii_info_size ) return info->resources_used;

  return info->ii_resources_used[ii - 1];
}
