
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include <_mingw.h>

int main (int argc, char **argv) {

  char *source = 0;
  int ret = 0;
  
  for (char **a = argv + 1; *a; ++a) {
    if (**a == '-') {
      continue;
    }
    if (!source) {
      source = *a;
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
    char source_path[_MAX_DRIVE + _MAX_DIR];
    
    if (_splitpath_s (source, source_drive, source_dir, source_fname, source_ext)) {
      _mingw_error ("invalid source");
      ret = 1;
      break;
    }

    // remove trailing (back)slash from dir, use "." if dir not given
    size_t source_len = strlen (source_dir);
    if (source_len > 1 && strchr ("/\\", source_dir[source_len - 1])) {
      source_dir[source_len - 1] = 0;
    } else if (!source_len) {
      strcpy_s (source_dir, _MAX_DIR, ".");
    }
     
    sprintf_s (source_path, sizeof (source_path), "%s%s", source_drive, source_dir);
    _mingw_sanitize_path (source_path);

    printf ("%s\n", source_path);
    break;    
  }
        
  return ret;
}
