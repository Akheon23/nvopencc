/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_WINDOWS_H
#define __MINGW32_MSVC_WINDOWS_H

#define INT8     __msvc_INT8
#define INT16    __msvc_INT16
#define UINT8    __msvc_UINT8
#define UINT16   __msvc_UINT16

#define NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "PlatformSDK/Include/Windows.h"

#undef INT8
#undef INT16
#undef UINT8
#undef UINT16

#endif // #ifndef __MINGW32_MSVC_WINDOWS_H
