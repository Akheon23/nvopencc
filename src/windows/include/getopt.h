/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_GETOPT_H
#define __MINGW32_MSVC_GETOPT_H

#ifdef __cplusplus
extern "C" {
#endif

static char *optarg = NULL;
static int optind = 1, opterr = 1, optopt = '?';

static int getopt (int argc, char * const *argv, const char *optstring) {
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif // #ifndef __MINGW32_MSVC_GETOPT_H
