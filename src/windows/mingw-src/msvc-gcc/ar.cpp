#include "__common.h"

int main (int argc, char **argv) {
  if (__common (&argv)) {
    return 1;
  }  
  
  __OUT = __LIB_ARGS = __BSCMAKE_ARGS = __BROWSE_FILES = "";
  __LIB_VERSION = __BSCMAKE_VERSION = "/nologo";
  char *out_listing = 0;
  
  while (argv[1]) {
    const char *__ARG = argv[1];
    ++argv;
    
    if (__matches (__ARG, "cru") || __matches (__ARG, "rcs") || __matches (__ARG, "rc")) {
      __OUT = __concat (argv[1]);
      __LIB_ARGS = __concat (__LIB_ARGS, " /OUT:", __OUT);
      out_listing = __concat (__chop_off_ext (__OUT), ".bsc.listing");
      _unlink (out_listing);
      ++argv;
      continue;
    }
    if (__matches (__ARG, "--version") || __matches (__ARG, "-V")) {
      __LIB_VERSION = "";
      continue;
    }
    if (__matches (__ARG, "--help")) {
      __LIB_ARGS = __concat (__LIB_ARGS, " /?");
      continue;
    }
    if (__matches (__ARG, "-v") || __matches (__ARG, "--verbose")) {
      __LIB_ARGS = __concat (__LIB_ARGS, " /VERBOSE");
      __BSCMAKE_ARGS = __concat (__BSCMAKE_ARGS, "/v");
      __VERBOSE = 1;
      continue;
    }
    if (__ends_with (__ARG, ".o")) {
      __LIB_ARGS = __concat (__LIB_ARGS, " ", __ARG);
      char *sbr = __concat (__chop_off_last (__ARG, '.'), ".sbr");
      if (out_listing && __file_exists (sbr)) {
        __BROWSE_FILES = __concat (__BROWSE_FILES, " ", sbr);
        sbr = __concat (" ", __realpath (sbr));
        __append_file (out_listing, sbr);
      }
      continue;
    }
    /* Pass it through by default */
    __LIB_ARGS = __concat (__LIB_ARGS, " ", __ARG);
  }
  
  if (__VERBOSE) {
    __LIB_VERSION = __BSCMAKE_VERSION = "";
  }
  
  __CMD = __concat (__LIB, " ", __LIB_VERSION, " ", __LIB_ARGS);

  /* Run command */
  int __RET = __invoke_cmd (__CMD);  
    
  /* If object files were involved, also gather corresponding browse info. */
  if (!__RET && !__empty_string (__BROWSE_FILES)) {
    __RET = __build_browse_info ();
  }
  
  return __RET;  
}
