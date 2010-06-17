/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STDLIB_H
#define __MINGW32_MSVC_STDLIB_H

#include "include/stdlib.h"
#include "include/io.h"

#define mktemp _mktemp
#define alloca _alloca
#define atoll _atoi64

#endif // #ifndef __MINGW32_MSVC_STDLIB_H
