/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_MATH_H
#define __MINGW32_MSVC_MATH_H

#include "include/math.h"

#define hypot _hypot
#define hypotf _hypotf

#define trunc(V) ((V) < 0.0? ceil(V): floor(V))

#endif // #ifndef __MINGW32_MSVC_MATH_H
