
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include <_mingw.h>

int main (int argc, char **argv) {

  char *source = 0, *suffix = 0;
  int ret = 0;
  
  for (char **a = argv + 1; *a; ++a) {
    if (**a == '-') {
      continue;
    }
    if (!source) {
      source = *a;
    } else if (!suffix) {
      suffix = *a;
    } else {
      _mingw_error ("too many arguments");
      ret = 4;
    }  
  }
  
  if (!source) {
    _mingw_error ("too few arguments");
    ret = 4;
  }
    
  while (!ret) {
    char source_drive[_MAX_DRIVE], source_dir[_MAX_DIR], source_fname[_MAX_FNAME], source_ext[_MAX_EXT];
    char source_file[_MAX_FNAME + _MAX_EXT];
    
    if (_splitpath_s (source, source_drive, source_dir, source_fname, source_ext)) {
      _mingw_error ("invalid source");
      ret = 1;
      break;
    }

    sprintf_s (source_file, sizeof (source_file), "%s%s", source_fname, source_ext);

    // remove trailing suffix, if present and found
    if (suffix) {
      size_t source_file_len = strlen (source_file), suffix_len = strlen (suffix);
      if (source_file_len > suffix_len && ! strcmp (source_file + source_file_len - suffix_len, suffix)) {
        source_file[source_file_len - suffix_len] = 0;
      }
    }
        
    printf ("%s\n", source_file);
    break;    
  }
        
  return ret;
}
