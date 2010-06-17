/*
 * Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
 */

#ifndef __OPT_DUMP_IR__
#define __OPT_DUMP_IR__

extern void Set_current_comp_unit (COMP_UNIT *unit);
extern COMP_UNIT *Get_current_comp_unit(void);

extern int
get_block_number_on_label(int label_no);

extern void
print_aux_extra (FILE *f, WN *wn);

extern void
print_cr_extra (FILE *f, CODEREP *cr);

extern void
Set_ssa_print_rout (void);

extern void
Reset_ssa_print_rout (void);

extern void
Set_cr_print_rout (void);

extern void
Reset_cr_print_rout (void);

extern void 
print_cfg_cr (void);

extern void
print_cfg_aux (void);

#endif // __OPT_DUMP_IR__

