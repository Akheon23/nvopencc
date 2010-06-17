#ifndef _MINGW_H
#define _MINGW_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void _mingw_sanitize_path (char *p) {
  while (*p) {
    if (*p == '\\') {
      *p = '/';
    }
    ++p;
  }    
}

void _mingw_error (const char *msg) {
  char *app = _strdup (__argv[0]);
  _mingw_sanitize_path (app);
  fprintf (stderr, "%s: %s\n", app, msg);
  free (app);
}

#endif  // #ifndef _MINGW_H