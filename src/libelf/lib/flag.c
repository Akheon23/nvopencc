/*
 * Copyright 2005 PathScale, Inc.  All Rights Reserved.
 */

/*
flag.c - implementation of the elf_flag*(3) functions.
Copyright (C) 1995 - 1998 Michael Riepe <michael@stud.uni-hannover.de>

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

#include <private.h>

#ifndef lint
static const char rcsid[] = "@(#) $Id: flag.c 1.4 05/05/06 08:20:34-07:00 bos@eng-25.pathscale.com $";
#endif /* lint */

static unsigned
_elf_flag(unsigned *f, Elf_Cmd cmd, unsigned flags) {
    if (cmd == ELF_C_SET) {
	return *f |= flags;
    }
    if (cmd == ELF_C_CLR) {
	return *f &= ~flags;
    }
    seterr(ERROR_INVALID_CMD);
    return 0;
}

unsigned
elf_flagdata(Elf_Data *data, Elf_Cmd cmd, unsigned flags) {
    Scn_Data *sd = (Scn_Data*)data;

    if (!sd) {
	return 0;
    }
    elf_assert(sd->sd_magic == DATA_MAGIC);
    return _elf_flag(&sd->sd_data_flags, cmd, flags);
}

unsigned
elf_flagehdr(Elf *elf, Elf_Cmd cmd, unsigned flags) {
    if (!elf) {
	return 0;
    }
    elf_assert(elf->e_magic == ELF_MAGIC);
    return _elf_flag(&elf->e_ehdr_flags, cmd, flags);
}

unsigned
elf_flagelf(Elf *elf, Elf_Cmd cmd, unsigned flags) {
    if (!elf) {
	return 0;
    }
    elf_assert(elf->e_magic == ELF_MAGIC);
    return _elf_flag(&elf->e_elf_flags, cmd, flags);
}

unsigned
elf_flagphdr(Elf *elf, Elf_Cmd cmd, unsigned flags) {
    if (!elf) {
	return 0;
    }
    elf_assert(elf->e_magic == ELF_MAGIC);
    return _elf_flag(&elf->e_phdr_flags, cmd, flags);
}

unsigned
elf_flagscn(Elf_Scn *scn, Elf_Cmd cmd, unsigned flags) {
    if (!scn) {
	return 0;
    }
    elf_assert(scn->s_magic == SCN_MAGIC);
    return _elf_flag(&scn->s_scn_flags, cmd, flags);
}

unsigned
elf_flagshdr(Elf_Scn *scn, Elf_Cmd cmd, unsigned flags) {
    if (!scn) {
	return 0;
    }
    elf_assert(scn->s_magic == SCN_MAGIC);
    return _elf_flag(&scn->s_shdr_flags, cmd, flags);
}
