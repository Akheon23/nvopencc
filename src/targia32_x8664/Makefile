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
        libcmplrs \
        libcomutil \
        libelf \
        libelfutil \
        libdwarf \
        libiberty \
	libspin_4_2_0 \
        targ_info


#
#  Finally, the compiler tools (components) are built.
#
#  Note that be_driver must come last since it depends on other components
#  being built first.
#
TOOL_SUBDIRS = \
        driver \
        gccfe \
	wgen_4_2_0 \
        ir_tools \
        lw_inline \
        be \
        cg \
        wopt \
	ipl \
        lno \
        ipa \
	whirl2c


SUBDIRS=$(LIBRARY_SUBDIRS) $(TOOL_SUBDIRS)

all: build install

clean clobber:
	@for d in $(SUBDIRS); do \
	    if [ -d $$d ] ; then \
		echo "===> cd $$d; $(MAKE) $@"; \
		cd $$d; $(MAKE) $@ || exit 1; cd ..; \
	    fi; \
	done
	rm -rf bin lib
	pushd ../../osprey-gcc-4.2.0/targia32_x8664; $(MAKE) $@ || exit 1; popd
	pushd ../targcygnus_ia32_x8664; $(MAKE) $@ || exit 1; popd

build:
	@for d in $(SUBDIRS); do \
	    if [ -d $$d ] ; then \
		echo "===> cd $$d; $(MAKE)"; \
		cd $$d; $(MAKE) || exit 1; cd ..; \
	    fi; \
	done
	echo "===> cd ../../osprey-gcc-4.2.0/targia32_x8664; $(MAKE)"; \
	pushd ../../osprey-gcc-4.2.0/targia32_x8664; $(MAKE) || exit 1; popd
	echo "===> cd ../targcygnus_ia32_x8664; $(MAKE)"; \
	pushd ../targcygnus_ia32_x8664; $(MAKE) || exit 1; popd

# install locally in targ*/bin
install:
	mkdir -p bin
	mkdir -p lib
	cp -u driver/driver bin/opencc
	cp -u driver/driver bin/openCC
	cp -u ir_tools/ir_b2a bin/ir_b2a
	cp -u ../../osprey-gcc-4.2.0/targia32_x8664/gcc/cc1plus42 lib
	cp -u wgen_4_2_0/wgen42 lib
	cp -u gccfe/gfec lib
	cp -u lw_inline/lw_inline lib/inline
	cp -u be/be lib
	cp -u be/be.so lib
	cp -u cg/cg.so lib
	cp -u wopt/wopt.so lib
	cp -u ipl/ipl.so lib
	cp -u lno/lno.so lib
	cp -u ipa/ipa.so lib
	cp -u targ_info/opteron.so lib
	cp -u targ_info/core.so lib
	cp -u ../targcygnus_ia32_x8664/ld/ld-new lib/ipa_link
	pushd lib; ln -s be ipl; popd
	# add dummy files to help ipa compile
	mkdir -p lib/32
	touch lib/32/libmv.a
	cp -u driver/driver bin/nvopencc

