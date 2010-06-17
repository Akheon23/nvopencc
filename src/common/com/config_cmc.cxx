/*
 * Copyright 2008-2010 NVIDIA Corporation.  All rights reserved.
 */

// This file is included in config.cxx, so it doesn't need its own set of
// standard includes -- only the following:
#include "config_cmc.h"

/* ====================================================================
 * List of global variables that are set by the -CMC option group
 * ====================================================================
 */
BOOL CMC_Use_Fallback = FALSE;   /* Force use of fall-back implementation */

static OPTION_DESC Options_CMC[] = {
    { OVK_BOOL,	OV_VISIBLE,	FALSE, "fallback",	"f",
	  0, 0, 0,		&CMC_Use_Fallback,	NULL,
	  "Force use of fall-back cmc strategy" },
    { OVK_COUNT }	    /* List terminator -- must be last */
};
