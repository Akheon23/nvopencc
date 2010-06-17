
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include <_mingw.h>

int main (int argc, char **argv) {

  char *source = 0, *target = 0;
  BOOL force = FALSE;
  int ret = 0;
  
  for (char **a = argv + 1; *a; ++a) {
    if (**a == '-') {
      if (strchr (*a, 'f')) {
        force = TRUE;
      }
      continue;
    }
    if (!source) {
      source = *a;
    } else if (!target) {
      target = *a;
    } else {
      _mingw_error ("too many arguments");
      ret = 4;
    }  
  }
  
  if (!source || !target) {
    _mingw_error ("too few arguments");
    ret = 4;
  }
    
  while (!ret) {
    char *pwd = _getcwd (NULL, 0);
    char target_drive[_MAX_DRIVE], target_dir[_MAX_DIR], target_fname[_MAX_FNAME], target_ext[_MAX_EXT];
    char target_path[_MAX_DRIVE + _MAX_DIR];
    
    target = _fullpath (NULL, target, 0);
    if (_splitpath_s (target, target_drive, target_dir, target_fname, target_ext)) {
      _mingw_error ("invalid target");
      ret = 1;
      break;
    }

    sprintf_s (target_path, sizeof (target_path), "%s%s", target_drive, target_dir);
    // we evaluate the source path relative to the target directory
    ret = _chdir (target_path);
    source = _fullpath (NULL, source, 0);
    _chdir (pwd);
    if (ret) {
      _mingw_error ("invalid target directory");
      ret = 1;
      break;
    }
 
    if (force) {   
      DWORD target_attribs = GetFileAttributes (target);
      SetFileAttributes (target, target_attribs & ~FILE_ATTRIBUTE_READONLY);
    }
    if (!CopyFile (source, target, !force)) {
      _mingw_error ("failure");
      ret = 1;
      break;
    }
    
    break;
  }
        
  return ret;
}
