#include <vector>
#include <algorithm>
#include <hash_set>

#include "__common.h"

struct path_info {
  const char *win_path;
  const char *unix_path;
  bool operator < (const path_info &pi2) const {
    /* We sort in order of _decreasing_ windows path length.  */
    return strlen (win_path) > strlen (pi2.win_path);
  }  
};

struct string_comp: stdext::hash_compare <char *> {
  size_t operator () (const char *s) const {
    size_t h = 17;
    while (*s) {
      h += (*s++ ^ 71);
    }  
    return h;
  }
  bool operator () (const char *s1, const char *s2) const {
    return strcmp (s1, s2) < 0;
  }
};

typedef std::vector<path_info> pathvec;
typedef stdext::hash_set<char *, string_comp> stringset;

static pathvec usr_paths, sys_paths, usr_sources;

static void add_path (pathvec &vec, const char *path) {
  path = __replace_char (path, '\\', '/');
  for (size_t i = 0; i < vec.size (); ++i) {
    if (!strcmp (vec[i].unix_path, path)) {
      return;
    }
  }
  struct path_info pi;
  pi.unix_path = path;
  pi.win_path = __realpath (path);
  vec.push_back (pi);
}

enum line_stat {
  lineEOF,
  lineSource,
  lineInclude
};
  
static char *retrieve_useful_line (const char **buf, line_stat *st, int ret = 0) {
  char *l = __retrieve_line (buf);
  while (l) {
    if (__starts_with (l, "Note: including file:")) {
      if (__SHOW_INCLUDES) {
        printf ("%s\n", l);
      }
      if (!ret) {
        *st = lineInclude;
        return l;
      }
      goto next_line;
    }
    if (__ends_with (l, ".cpp") || __ends_with (l, ".cc")
        || __ends_with (l, ".cxx") || __ends_with (l, ".c")) {
      if (!ret) {  
        *st = lineSource;
        return l;
      }
      goto next_line;
    }
    printf ("%s\n", l);
   next_line: 
    l = __retrieve_line (buf);
  }
  *st = lineEOF;
  return 0;  
}

static void write_out_dependencies (int ret) {
  std::sort (usr_paths.begin (), usr_paths.end ());
  const char *membuf = __STDOUT_MAP;
  line_stat st;
  char *line = retrieve_useful_line (&membuf, &st, ret);
  while (st == lineSource) {
    char *dep = __concat (__chop_off_last (line, '.'), ".d");
    const char *obj_targ = __concat (__chop_off_ext (line), ".o"), *src_dep = 0;
    FILE *d;
    stringset paths;
    
    fopen_s (&d, dep, "w");
    for (size_t i = 0; i < usr_sources.size (); ++i) {
      if (__ends_with (usr_sources[i].unix_path, line)) {
        src_dep = usr_sources[i].unix_path;
        break;
      }
    }
    fprintf (d, "%s: %s", obj_targ, src_dep);
    line = retrieve_useful_line (&membuf, &st);
    while (st == lineInclude) {
      char *fname = 0, *fpath = line + sizeof ("Note: including file:");
      while (strchr (" \t", *fpath)) {
        ++fpath;
      }
      fpath = __realpath (fpath);
      if (paths.insert (fpath).second) {
        for (size_t i = 0; i < usr_paths.size (); ++i) {
          if (__starts_with (fpath, usr_paths[i].win_path)) {
            fname = __replace_char (fpath + strlen (usr_paths[i].win_path), '\\', '/');
            fname = __concat (usr_paths[i].unix_path, fname);
            break;
          }
        }
        if (fname) {
          fprintf (d, " \\\n  %s", fname);
        }
      }  
      line = retrieve_useful_line (&membuf, &st);
    }
    fprintf (d, "\n\n");
    fclose (d);      
  }
}

static int __gcc (char **argv) {
  if (__common (&argv)) {
    return 1;
  }  

  __OUT = __BROWSE_FILES = __BSCMAKE_ARGS = __LINK_DEBUG = "";
  __CL_ARGS = "/D__MINGW32__ /DWIN32 /D__STDC__ /D__STDC_VERSION__=199409L /D__func__=__FUNCTION__ /Dinline=__inline"
              " /D__inline__=__inline /D__alignof__=__alignof /D_CRT_SECURE_NO_WARNINGS /MTd /bigobj";
  __LINK_ARGS = "/INCREMENTAL:NO /MANIFEST:NO";
  __NO_CL = __NO_LINK = 1;
  __PREPROCESS_ONLY = __SHOW_INCLUDES = 0;
  __CL_VERSION = __LINK_VERSION = __BSCMAKE_VERSION = "/nologo";
  __CL_DEBUG = "/DNDEBUG";

  while (argv[1]) {
    const char *__ARG = argv[1];
    ++argv;

    if (__starts_with (__ARG, "-D")) {
      __CL_ARGS = __concat (__CL_ARGS, " /D", __preserve_quotes (__ARG + 2));
      continue;
    }    
    if (__starts_with (__ARG, "-U")) {
      __CL_ARGS = __concat (__CL_ARGS, " /U", __ARG + 2);
      continue;
    }    
    if (__starts_with (__ARG, "-I")) {
      const char *p;
      if (*(__ARG + 2)) {
        p = __ARG + 2;
      } else {
        p = argv[1];
        ++argv;
      }
      __CL_ARGS = __concat (__CL_ARGS, " /I", __replace_char (p, '/', '\\'));
      add_path (usr_paths, p);
      continue;
    }    
    if (__matches (__ARG, "-Wformat") || __matches (__ARG, "-Wno-format") 
        || __starts_with (__ARG, "-std=") || __matches (__ARG, "-mno-cygwin") 
        || __starts_with (__ARG, "-mno-sse") || __matches (__ARG, "-m32") 
        || __matches (__ARG, "-fno-strict-aliasing")) {
      continue;
    } 
    if (__matches (__ARG, "-MMD")) {
      __DEPENDENCY_MODE = __concat (__ARG);
      __CL_ARGS = __concat (__CL_ARGS, " /showIncludes");
      continue;
    }
    if (__matches (__ARG, "-include")) {
      __CL_ARGS = __concat (__CL_ARGS, " /FI", __replace_char (argv[1], '/', '\\'));
      ++argv;
      continue;
    }
    if (__matches (__ARG, "-Werror")) {
      __CL_ARGS = __concat (__CL_ARGS, " /WX /wd4068 /wd4005 /wd4274 /wd4142 /wd4114 /wd4716 /wd4144 /wd4553"
                                       " /wd4715 /wd4113 /wd4530 /wd4700 /wd4145 /wd4090 /wd4293 /wd4355 /wd4305"
                                       " /wd4805 /wd4190");
      continue;
    }
    if (__matches (__ARG, "-O0")) {
      __CL_ARGS = __concat (__CL_ARGS, " /Od");
      continue;
    }
    if (__matches (__ARG, "-O2")) {
      __CL_ARGS = __concat (__CL_ARGS, " /O2");
      __CL_DEBUG = "/D_DEBUG /Zi /FR"; // temporary!!!
      __LINK_DEBUG = "/DEBUG";
      continue;
    }
    if (__matches (__ARG, "-c")) {
      __CL_ARGS = __concat (__CL_ARGS, " /c");
      __NO_LINK = 1;
      __NO_CL = 0;
      continue;
    }                                    
    if (__matches (__ARG, "-H")) {
      __CL_ARGS = __concat (__CL_ARGS, " /showIncludes");
      __SHOW_INCLUDES = 1;
      continue;
    }                                    
    if (__matches (__ARG, "--help") || __matches (__ARG, "--target-help")) {
      __CL_ARGS = __concat (__CL_ARGS, " /?");
      continue;
    }
    if (__matches (__ARG, "--verbose") || __matches (__ARG, "-v")) {
      __CL_ARGS = __concat (__CL_ARGS, " /FC /showIncludes");
      __LINK_ARGS = __concat (__LINK_ARGS, " /VERBOSE");
      __BSCMAKE_ARGS = __concat (__BSCMAKE_ARGS, "/v");
      __VERBOSE = 1;
      continue;
    }
    if (__matches (__ARG, "-E")) {
      if (__matches (argv[1], "-P")) {
        __CL_ARGS = __concat (__CL_ARGS, " /EP");
        ++argv;
      } else {
        __CL_ARGS = __concat (__CL_ARGS, " /E");
      }  
      __NO_LINK = __PREPROCESS_ONLY = 1;
      __NO_CL = 0;
      continue;
    }  
    if (__matches (__ARG, "-o")) {
      __OUT = __replace_char (argv[1], '/', '\\');
      ++argv;
      continue;
    }
    if (__matches (__ARG, "-funsigned-char")) {
      __CL_ARGS = __concat (__CL_ARGS, " /J");
      continue;
    }                                    
    if (__matches (__ARG, "-fno-exceptions")) {
      __CL_ARGS = __concat (__CL_ARGS, " /EHs-c-");
      continue;
    }                                    
    if (__starts_with (__ARG, "-msse")) {
      __CL_ARGS = __concat (__CL_ARGS, " /arch:", __ARG + 2);
      continue;
    }                                    
    if (__matches (__ARG, "-g")) {
      __CL_DEBUG = "/D_DEBUG /Zi /FR";
      __LINK_DEBUG = "/DEBUG";
      continue;
    }
    if (__matches (__ARG, "-xc")) {
      __CL_ARGS = __concat (__CL_ARGS, " /TC");
      continue;
    }                                    
    if (__matches (__ARG, "-xc++")) {
      __CL_ARGS = __concat (__CL_ARGS, " /TP");
      continue;
    }                                    
    if (__matches (__ARG, "-x")) {
      if (__matches (argv[1], "c")) {
        __CL_ARGS = __concat (__CL_ARGS, " /TC");
      } else if (__matches (argv[1], "c++")) {
        __CL_ARGS = __concat (__CL_ARGS, " /TP");
      } else {
        /* This will almost certainly explode */
        __ARG = __concat (__ARG, " ", argv[1]);
        goto pass_through_and_warn;
      }
      ++argv;
      continue;
    } 
    if (__starts_with (__ARG, "-L")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " /LIBPATH:", __replace_char (__ARG + 2, '/', '\\'));
      continue;
    }
    if (__matches (__ARG, "-lm")) {
      continue;
    }
    if (__matches (__ARG, "-Wl,--version") || __matches (__ARG, "-Wl,-v") || __matches (__ARG, "-Wl,-V")) {
      __LINK_VERSION = "";
      continue;
    }      
    if (__matches (__ARG, "-Wl,--verbose")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " /VERBOSE");
      __BSCMAKE_ARGS = __concat (__BSCMAKE_ARGS, "/v");
      continue;
    }      
    if (__matches (__ARG, "-Wl,--help") || __matches (__ARG, "-Wl,--target-help")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " /?");
      continue;
    }
    if (__matches (__ARG, "-Wl,--export-dynamic")) {
      continue;
    }
    if (__starts_with (__ARG, "-l")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " lib", __ARG + 2, ".a");
      __NO_LINK = 0;
      continue;
    }
    if (__ends_with (__ARG, ".a")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " ", __replace_char (__ARG, '/', '\\'));
      const char *bsc = __concat (__chop_off_last (__ARG, '.'), ".bsc.listing");
      if (__file_exists (bsc)) {
        __BROWSE_FILES = __concat (__BROWSE_FILES, " ", __read_file (bsc));
      }
      __NO_LINK = 0;
      continue;
    }  
    if (__ends_with (__ARG, ".o")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " ", __replace_char (__ARG, '/', '\\'));
      add_path (usr_sources, __ARG);
      const char *base = __chop_off_last (__basename (__ARG), '.');
      const char *sbr = __concat (__chop_off_last (__ARG, '.'), ".sbr");
      if (__file_exists (sbr)) {
        __BROWSE_FILES = __concat (__BROWSE_FILES, " ", sbr);
      }
      __NO_LINK = 0;
      continue;
    }
    if (__ends_with (__ARG, ".cpp") || __ends_with (__ARG, ".cc") || __ends_with (__ARG, ".cxx") || __ends_with (__ARG, ".c")) {
      __CL_ARGS = __concat (__CL_ARGS, " ", __replace_char (__ARG, '/', '\\'));
      add_path (usr_sources, __ARG);
      const char *base = __chop_off_last (__basename (__ARG), '.');
      __BROWSE_FILES = __concat (__BROWSE_FILES, " ", base, ".sbr");
      __NO_CL = 0;
      continue;
    }
    if (__starts_with (__ARG, "-Wl,")) {
      __LINK_ARGS = __concat (__LINK_ARGS, " ", __ARG + 4);
      continue;
    }
    if (__matches (__ARG, "--version")) {
      __CL_VERSION = "";
      continue;
    }
    /* Pass anything else through, but warn first */
   pass_through_and_warn: 
    __ARG = __replace_char (__ARG, '/', '\\');
    fprintf (stderr, "WARNING: Passing unrecognized option '%s' to MSVC\n", __ARG);
    __CL_ARGS = __concat (__CL_ARGS, " ", __ARG);
  }
  
  if (__VERBOSE) {
    __CL_VERSION = __LINK_VERSION = __BSCMAKE_VERSION = "";
  }
  
  /* Determine output file name if not explicitly specified */
  if (!__PREPROCESS_ONLY) {
    if (__empty_string (__OUT)) {
      __OUT = usr_sources.size ()
              ? __chop_off_ext (__basename (usr_sources[0].unix_path))
              : "a";
      __OUT = __concat (__OUT, __NO_LINK? ".o": ".exe");
    } else if (!__NO_LINK && !__ends_with (__OUT, ".exe")) {
      __OUT = __concat (__OUT, ".exe");
    }  
  }
  
  /* Decide whether to run CL.EXE or LINK.EXE */
  if (!__NO_CL) {
    __CMD = __concat (__CL, " ", __CL_VERSION, " ", __CL_DEBUG);
    if (!__PREPROCESS_ONLY) {
      __CMD = __concat (__CMD, " /Fo", __OUT);
    }
    __CMD = __concat (__CMD, " ", __CL_ARGS);
    if (!__NO_LINK) {
      __CMD = __concat (__CMD, " /link ", __LINK_VERSION, " ", __LINK_DEBUG, " ", __LINK_ARGS);
    } else if (__PREPROCESS_ONLY && !__empty_string (__OUT)) {
      __STDOUT_FILE = __OUT;
    }    
  } else {
    __CMD = __concat (__LINK, " ", __LINK_VERSION, " ", __LINK_DEBUG);
    __CMD = __concat (__CMD, " /OUT:", __OUT);
    __CMD = __concat (__CMD, " ", __LINK_ARGS);
    __DEPENDENCY_MODE = 0;  /* since no compiling is being done */
  }
  
  /* Run command */
  int __RET = __invoke_cmd (__CMD);  
  
  /* If compiling was performed, create dependency files. */
  if (!__NO_CL && __DEPENDENCY_MODE) {
    write_out_dependencies (__RET);
  }
  
  /* If linking was performed and was successful, build browse info. */
  if (!__RET && !__empty_string (__BROWSE_FILES) && !__NO_LINK) {
    __RET = __build_browse_info ();
  }
  
  return __RET;  
}