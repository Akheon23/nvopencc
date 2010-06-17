
#include <direct.h>
#include <stdlib.h>
#include <stdio.h>

#include <_mingw.h>

int main (void) {
  char *pwd = _getcwd (NULL, 0);

  if (pwd) {
    _mingw_sanitize_path (pwd);
    printf ("%s\n", pwd);
    free (pwd);
  } else {
    _mingw_error ("could not retrieve current directory");
  }    

  return (pwd == 0);
}