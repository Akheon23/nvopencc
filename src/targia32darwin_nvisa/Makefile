#
# Copyright 2005-2010 NVIDIA Corporation.  All rights reserved.
#

#
#
#  Copyright (C) 2000, 2001 Silicon Graphics, Inc.  All Rights Reserved.
#
#  This program is free software; you can redistribute it and/or modify it
#  under the terms of version 2 of the GNU General Public License as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it would be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
#  Further, this software is distributed without any warranty that it is
#  free of the rightful claim of any third person regarding infringement
#  or the like.  Any license provided herein, whether implied or
#  otherwise, applies only to this software file.  Patent licenses, if
#  any, provided herein do not apply to combinations of this program with
#  other software, or any other product whatsoever.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write the Free Software Foundation, Inc., 59
#  Temple Place - Suite 330, Boston MA 02111-1307, USA.
#
#  Contact information:  Silicon Graphics, Inc., 1600 Amphitheatre Pky,
#  Mountain View, CA 94043, or:
#
#  http://www.sgi.com
#
#  For further information regarding this notice, see:
#
#  http://oss.sgi.com/projects/GenInfo/NoticeExplan
#
#

#
#  Makefile for targia32_nvisa
#

#
#  all of the libraries are built because the tools depend on them.
#
LIBRARY_SUBDIRS = \
        include \
        libelf \
        libelfutil \
        libdwarf \
        libiberty \
        targ_info \
        libbe \
        libcomutil \
        libcg \
        libwopt\
        libwhirl2c

#
#  Finally, the compiler tools (components) are built.
#
TOOL_SUBDIRS = \
        driver \
        gccfe \
        ir_tools \
        inliner \
        backend

# comic:
TOOL_SUBDIRS += bec

SUBDIRS=$(LIBRARY_SUBDIRS) $(TOOL_SUBDIRS)

all: build install

clean clobber:
	@for d in $(SUBDIRS); do \
	    if [ -d $$d ] ; then \
		echo "===> cd $$d; $(MAKE) $@"; \
		cd $$d; $(MAKE) $@  || exit 1; cd ..; \
	    fi; \
	done
	rm -rf bin lib

build:
	@for d in $(SUBDIRS); do \
	    if [ -d $$d ] ; then \
		echo "===> cd $$d; $(MAKE)"; \
		cd $$d; $(MAKE) SKIP_DEP_BUILD=1 || exit 1; cd ..; \
	    fi; \
	done

install:
	mkdir -p bin
	mkdir -p lib
	cp -f driver/driver bin/nvopencc
	cp -f gccfe/gfec lib
	cp -f inliner/inline lib
	cp -f backend/be lib
	cp -f bec/bec lib
