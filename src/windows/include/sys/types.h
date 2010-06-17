/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_SYS_TYPES_H
#define __MINGW32_MSVC_SYS_TYPES_H

#include "include/sys/types.h"

typedef unsigned int size_t;
typedef long ssize_t;
typedef long off_t;
typedef unsigned short ino_t;
typedef unsigned int dev_t;

#endif // #ifndef __MINGW32_MSVC_SYS_TYPES_H
