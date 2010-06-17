#!/bin/bash
source $(dirname $0)/__common

__LIB="$__DIR/bin/lib.exe"
__BSCMAKE="$__DIR/bin/bscmake.exe"

__check_exec "$__LIB" "$__BSCMAKE"

__OUT=""
__LIB_ARGS=""
__BROWSE_FILES=""
__LIB_VERSION="/nologo"
__BSCMAKE_VERSION="/nologo"

while [[ $# -ge 1 ]]; do
  __ARG="$1"
  shift
  case $__ARG in
    cru | rcs | rc )
      __OUT="$1"
      shift
      __LIB_ARGS="$__LIB_ARGS /OUT:$__OUT"
      rm -f ${__OUT%.*}.bsc.listing
      ;;
    --version | -V )
      __LIB_VERSION=""
      ;;
    --help )
      __LIB_ARGS="$__LIB_ARGS /?"
      ;;
    -v )
      VERBOSE=1
      ;;
    *.o )
      __LIB_ARGS="$__LIB_ARGS $__ARG"
      if [[ -f ${__ARG%.*}.sbr ]]; then
        __BROWSE_FILES="$__BROWSE_FILES ${__ARG%.*}.sbr"
        echo -n " $(cygpath -am ${__ARG%.*}.sbr)" >>${__OUT%.*}.bsc.listing
      fi  
      ;;  
    * ) 
      # pass it thru by default 
      __LIB_ARGS="$__LIB_ARGS $__ARG"
      ;;
  esac
done

if [[ "$VERBOSE" ]]; then
  __LIB_VERSION=""
  __BSCMAKE_VERSION=""
fi

__CMD="$__LIB $__LIB_ARGS $__LIB_VERSION"
[[ "$VERBOSE" ]] && __CMD="$__CMD /VERBOSE"
__invoke_cmd "$__CMD"
__RET=$?

# If object files were involved, also gather corresponding browse info
if [[ $__RET -eq 0 && "$__BROWSE_FILES" ]]; then
  __build_browse_info
fi 

exit $__RET
