/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STRING_H
#define __MINGW32_MSVC_STRING_H

#include "include/string.h"

#define strdup _strdup

/* The new, correct place for the following is <strings.h>. */
#define strcasecmp _stricmp
#define strncasecmp _strnicmp

#define bzero(S, N) memset(S, 0, N)

#endif // #ifndef __MINGW32_MSVC_STRING_H
