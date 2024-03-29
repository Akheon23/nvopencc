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


#include "basic.h"
#include "lang_defs.h"

extern char *source_file;
extern boolean_t multiple_source_files;

extern char *outfile;		/* outfile for -o */
extern char *prof_file;        /*  executable file for prof to work upon */
extern char *fb_file;		/* feedback file for -fb_create */
extern char *internal_fb_file;	/* feedback file for -fb */
extern char *opt_file;		/* feedback file for -fb_opt */
extern char *fb_xdir;		/* dir where pixie emits dso's */
extern char *fb_cdir;		/* dir where pixie emits count files */
extern char *fb_phase;         /* phase for -fb_phase */
extern char *fb_type;          /* type for -fb_type */
#ifdef KEY
extern char *coco_setfile;  /* setfile for -fcoco */
#endif

extern char *global_toolroot;   /* TOOLROOT, get from $TOOLROOT or guess from arg[0] */
extern char *ldpath_for_pixie;  /* Tell pixie where to find ld */
extern int saved_argc;		/* original argc */
extern char **saved_argv;	/* original argv */
extern char *command_line;	/* original command line */

extern boolean_t keep_mp;		/* keep pfa/pca file */
extern boolean_t keep_list;	/* keep pfa/pca listing */
extern boolean_t keep_listing;	/* keep listing file */
extern boolean_t keep_cif;	/* keep cif file */
extern boolean_t auto_parallelize;	/* invoke lno:ap=1 */
extern boolean_t Gen_feedback;	/* Generate pixified binary for spec */
extern boolean_t Use_feedback;	/* Use feedback for spec */
extern boolean_t Disable_open_mp; /* Disable the recognition of open mp */
extern boolean_t Disable_old_mp;  /* Disable the recognition old style mp */
extern char roundoff;		/* roundoff level for pfa */
extern boolean_t O3_flag;		/* -O3 has been specified */
extern boolean_t nocpp_flag;	/* don't invoke cpp */
extern boolean_t use_cpp;		/* Use cpp instead of ftpp for F90 */
extern boolean_t expand_ftpp_macros; /* fully macro-expand in ftpp */
extern int     fortran_line_length;	/* Line length for fixed form fortran */

extern char *ld_library_path;	/* env. variable LD_LIBRARY_PATH */
extern char *ld_libraryn32_path;   /* env. variable LD_LIBRARYN32_PATH */

extern char *orig_program_name; /* actual name passed in to argv[0] */

boolean_t dump_outfile_to_stdout;	// for "-o -"


/* call once before running compiler */
extern void init_phase_info (void);

/* run dsm_prelink */
extern void run_dsm_prelink(void);

/* run ld only */
phases_t determine_ld_phase (boolean_t run_ipa);
extern void run_ld (void);
extern void run_ar (void);
extern void run_pixie (void);
extern void run_prof (void);

/* run whole compiler */
extern void run_compiler ();

/* return fixed-up file name */
extern char *fix_name_by_lang (char *name);

/* save original command line */
extern void save_command_line(int, char **);
extern void set_current_arg_pos(int n);
extern int check_for_saved_option(char *s);
extern void cancel_saved_arg(int count);
extern void add_minus_c_option(void);

/* save user options for ipl */
extern void save_ipl_commands (void);
extern char *dirname(char *const s);

int quote_shell_arg (char *p, char *buf);

// Change the phase names based on run-time info.
extern void init_phase_names (void);

#ifdef KEY
// Get the system GCC's major version number.
extern int get_gcc_major_version(void);
extern void init_frontend_phase_names(int, int);
#endif

#define PASS1 0
#define PASS2 1



