
################################################################
#
# Make version check:
#
################################################################

ifndef MAKEFILE_LIST
define gmakevermsg

************************************************************

   Invalid version of gmake (>=3.80 required). 
   Are you using the one from the sw/tools? 

************************************************************
TARGET := $(error $(call gmakevermsg))

endef
endif


################################################################
#
# Determine top of various directory trees:
#
################################################################

ROOT_DIR         := $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST)))/..
NV_TOOLS         := $(ROOT_DIR)/../../tools

CYGWIN_INSTALL   := $(NV_TOOLS)/win32/cygnus/Apr2008
MINGW_INSTALL    := $(NV_TOOLS)/msys/1.0

################################################################
#
# Heuristics to determine current OS:
#
################################################################

# Cygwin sets WINDIR instead of windir
ifneq ($(WINDIR),)
  windir := $(WINDIR)
endif

ifeq ($(windir),)
        ifeq ($(wildcard /cygdrive/c/WINDOWS*),)
          ifeq ($(shell uname), Darwin)
            export OS = Darwin
          else
            export OS = Linux
          endif
        else
            export OS = win32
        endif
else
	ifeq ($(shell uname), Darwin)
	  export OS = Darwin
	else
	  export OS = win32
	endif
endif

################################################################
#
# Define current build environment
#
################################################################

ifeq ($(OS), Linux)
    export HOST_ARCH        := $(shell uname -m)
    export HOST_OS          := $(shell uname   )
    USE_NATIVE              := 1
endif

ifeq ($(OS), Darwin)
    export HOST_ARCH        := $(shell uname -m)
    export HOST_OS          := $(shell uname   )
    USE_NATIVE              := 1
endif

ifeq ($(OS), win32)
    ifeq ($(PROCESSOR_ARCHITECTURE),x86)
        ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
            export HOST_ARCH := x86_64
        else
            export HOST_ARCH := i686
        endif
    else
        export HOST_ARCH := x86_64
    endif

    export HOST_OS          := CygWin

    ifdef USE_NATIVE
        TOOL_PREFIX         := 
    else
     ifdef USE_CYGWIN
        TOOL_PREFIX          = $(CYGWIN_INSTALL)/bin/
        SHELL               := $(CYGWIN_INSTALL)/bin/sh.exe
     else
        USE_MINGW           := 1
        export DRIVE_PREFIX := 
        TOOL_PREFIX          = $(MINGW_INSTALL)/bin/
        SHELL               := $(ROOT_DIR)/build/shell.exe
     endif
     
     __to_dos__  = | $(TOOL_PREFIX)sed "s@/cygdrive/\(.\)/@\1\:/@" \
                   | $(TOOL_PREFIX)sed "s@/\(.\)/@\1\:/@"
    endif
endif


export ARCH       := $(HOST_ARCH)
export OS_VARIANT := $(OS)
    
ifeq ($(OS), win32)
    ifeq ($(findstring ProgramData,"$(ALLUSERSPROFILE)"),ProgramData)
        export OS_VARIANT := vista
    endif
endif



MSVC_INSTALL_BIN := $(NV_TOOLS)/win32/msvc80sp1/VC/bin


################################################################
#
# Convert to absolute paths
#
################################################################

export ROOT_DIR             := $(shell echo `cd $(ROOT_DIR);         pwd ` $(__to_dos__) )

ifeq ($(OS), win32)
    export MSVC_INSTALL_BIN := $(shell echo `cd $(MSVC_INSTALL_BIN); pwd ` $(__to_dos__) )
endif

ifneq ($(TOOL_PREFIX),)
    export MINGW_INSTALL    := $(shell echo `cd $(MINGW_INSTALL);    pwd ` $(__to_dos__) )/
    export CYGWIN_INSTALL   := $(shell echo `cd $(CYGWIN_INSTALL);   pwd ` $(__to_dos__) )/
    export TOOL_PREFIX      := $(shell echo `cd $(TOOL_PREFIX);      pwd ` $(__to_dos__) )/
    export NV_TOOLS         := $(shell echo `cd $(NV_TOOLS);         pwd ` $(__to_dos__) )/
endif

ifdef USE_MINGW
    export PATH             := $(TOOL_PREFIX);$(PATH)
    export SHELL            := $(ROOT_DIR)/build/shell.exe
endif

ifdef USE_CYGWIN
    export SHELL            := $(CYGWIN_INSTALL)/bin/sh.exe
endif


################################################################
#
# Definition of commands that are allowed in the build:
#
################################################################

export AR                 := $(TOOL_PREFIX)ar
export AWK                := $(TOOL_PREFIX)awk
export BASENAME           := $(TOOL_PREFIX)basename
export CAL                := $(TOOL_PREFIX)cal
export CAT                := $(TOOL_PREFIX)cat
export CHMOD              := $(TOOL_PREFIX)chmod
export CHOWN              := $(TOOL_PREFIX)chown
export CKSUM              := $(TOOL_PREFIX)cksum
export CMP                := $(TOOL_PREFIX)cmp
export COMM               := $(TOOL_PREFIX)comm
export CP                 := $(TOOL_PREFIX)cp -f
export CUT                := $(TOOL_PREFIX)cut
export DIFF               := $(TOOL_PREFIX)diff
export DIRNAME            := $(TOOL_PREFIX)dirname
export DOS2UNIX           := $(TOOL_PREFIX)dos2unix
export DU                 := $(TOOL_PREFIX)du
export EGREP              := $(TOOL_PREFIX)egrep
export ENV                := $(TOOL_PREFIX)env
export EXPR               := $(TOOL_PREFIX)expr
export FALSE              := $(TOOL_PREFIX)false
export FGREP              := $(TOOL_PREFIX)fgrep
export FIND               := $(TOOL_PREFIX)find
export GREP               := $(TOOL_PREFIX)grep
export GUNZIP             := $(TOOL_PREFIX)gunzip
export GZIP               := $(TOOL_PREFIX)gzip
export HEAD               := $(TOOL_PREFIX)head
export HOSTID             := $(TOOL_PREFIX)hostid
export INSTALL            := $(TOOL_PREFIX)install
export JOIN               := $(TOOL_PREFIX)join
export LEX                := $(TOOL_PREFIX)lex
export LN                 := $(TOOL_PREFIX)ln
export LOGNAME            := $(TOOL_PREFIX)logname
export LS                 := $(TOOL_PREFIX)ls
export MD5SUM             := $(TOOL_PREFIX)md5sum
export MKDIR              := $(TOOL_PREFIX)mkdir
export M4                 := $(TOOL_PREFIX)m4
export MV                 := $(TOOL_PREFIX)mv
export OD                 := $(TOOL_PREFIX)od
export PERL               := $(TOOL_PREFIX)perl
export PYTHON             := $(TOOL_PREFIX)python
export PRINTENV           := $(TOOL_PREFIX)printenv
export PRINTF             := $(TOOL_PREFIX)printf
export REALPATH           := $(TOOL_PREFIX)realpath
export RM                 := $(TOOL_PREFIX)rm -rf
export RMDIR              := $(TOOL_PREFIX)rm -rf
export SED                := $(TOOL_PREFIX)sed
export SEQ                := $(TOOL_PREFIX)seq
export SHA1SUM            := $(TOOL_PREFIX)sha1sum
export SLEEP              := $(TOOL_PREFIX)sleep
export SORT               := $(TOOL_PREFIX)sort
export STAT               := $(TOOL_PREFIX)stat
export STRINGS            := $(TOOL_PREFIX)strings
export SUM                := $(TOOL_PREFIX)sum
export TAIL               := $(TOOL_PREFIX)tail
export TAR                := $(TOOL_PREFIX)tar
export TEE                := $(TOOL_PREFIX)tee
export TEST               := $(TOOL_PREFIX)test
export TOUCH              := $(TOOL_PREFIX)touch
export TR                 := $(TOOL_PREFIX)tr
export TRUE               := $(TOOL_PREFIX)true
export UNAME              := $(TOOL_PREFIX)uname
export UNIQ               := $(TOOL_PREFIX)uniq
export UNIX2DOS           := $(TOOL_PREFIX)unix2dos
export UNZIP              := $(TOOL_PREFIX)unzip
export USLEEP             := $(TOOL_PREFIX)usleep
export WC                 := $(TOOL_PREFIX)wc
export WHICH              := $(TOOL_PREFIX)which
export XARGS              := $(TOOL_PREFIX)xargs
export YACC               := $(TOOL_PREFIX)bison
export YES                := $(TOOL_PREFIX)yes
export ZCAT               := $(TOOL_PREFIX)zcat
export ZIP                := $(TOOL_PREFIX)zip

ifneq ($(OS), Darwin)
    CP += -u
endif

ifeq ($(OS),win32)
      export ZIP	         := $(NV_TOOLS)/win32/infozip/zip
endif

# Shell built-in commands:
export CD                := cd
export ECHO              := echo
export _PWD              := pwd        #Note underscore to distinguish from PWD (shell macro)

ifdef USE_MINGW
  export AWK    = awk
  export EGREP  = egrep
  export FGREP  = fgrep
  export GUNZIP = gunzip
  export PRINTF = printf
  export WHICH  = which
endif



################################################################
#
# Redefinition of commands in CYGWIN or MINGW mode:
#
################################################################

ifeq ($(USE_NATIVE),1)
  LEX := flex
else

  #
  # i.e. USE_CYGWIN=1 or USE_MINGW=1.
  #
  export LEX      := $(NV_TOOLS)/win32/MiscBuildTools/flex_2_5_4
  export YACC     := $(NV_TOOLS)/win32/MiscBuildTools/bison_1_25
  export PERL     := $(NV_TOOLS)/win32/ActivePerl/584/bin/perl.exe
  export PYTHON   := $(NV_TOOLS)/win32/python/254/python.exe
  export GCC_BIN  := $(CYGWIN_INSTALL)/bin/
  export AR       := $(CYGWIN_INSTALL)/bin/ar

ifeq ($(OS), win32)
  #
  # bison uses 'm4' from the current exec path:
  #
  export M4                  := $(MINGW_INSTALL)/bin/m4
  export BISON_INSTALLDIR    := $(NV_TOOLS)/win32/bison/bison_2_3
  export BISON_PKGDATADIR    := $(BISON_INSTALLDIR)/share/bison
  YACC                       := $(BISON_INSTALLDIR)/bin/bison
endif

endif

  export BISON $(YACC)




ifeq ($(OS), win32)
  # Current working directory macro
  export _PWD       := pwd $(__to_dos__)
  export _MAKESHELL := $(SHELL)

  ifndef PWD
    export PWD := $(shell $(_PWD))
  endif
endif

# Note in the following: on Linux, nvcc is prebuilt, 
# while on Windows it is checked in:
ifeq ($(OS), win32)
    OBJ_SUFFIX = obj
    NVCC := $(ROOT_DIR)/build/$(HOST_ARCH)_win32/nvcc -ccbin $(MSVC_INSTALL_BIN)
else
    OBJ_SUFFIX = o
    NVCC := $(ROOT_DIR)/built/nvcc
endif

#don't use PWD in build scripts, since
#that macro is not guaranteed to be in DOS
#format on Windows:
export CURDIR := $(shell $(_PWD))

################################################################
#
# Defining secondary macros:
#
################################################################

ifdef RELEASE
    export RELEASE    = 1
    export BUILD_MODE = release
else
    export DEBUG      = 1
    export BUILD_MODE = debug
endif
