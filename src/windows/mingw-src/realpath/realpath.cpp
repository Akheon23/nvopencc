
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
    source = _fullpath (NULL, source, 0);
    _mingw_sanitize_path (source);
    
    printf ("%s\n", source);  
    break;
  }
        
  return ret;
}
