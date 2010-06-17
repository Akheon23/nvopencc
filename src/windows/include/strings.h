/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STRINGS_H
#define __MINGW32_MSVC_STRINGS_H

#include "include/string.h"

#define strcasecmp _stricmp
#define strncasecmp _strnicmp

#define bzero(S, N) memset(S, 0, N)

#endif // #ifndef __MINGW32_MSVC_STRINGS_H
