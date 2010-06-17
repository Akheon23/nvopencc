/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STDIO_H
#define __MINGW32_MSVC_STDIO_H

#include "include/stdio.h"

#define snprintf _snprintf
#define popen _popen
#define pclose _pclose
#define fdopen _fdopen

#define setvbuf(F, B, M, S) setvbuf(F, B, M, (S < 2? BUFSIZ: S))

#endif // #ifndef __MINGW32_MSVC_STDIO_H
