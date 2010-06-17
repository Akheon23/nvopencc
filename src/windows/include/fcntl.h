/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_FCNTL_H
#define __MINGW32_MSVC_FCNTL_H

#include "include/fcntl.h"

#define open  _open
#define close _close

#define O_RDONLY _O_RDONLY
#define O_BINARY _O_BINARY
#define O_RDWR   _O_RDWR
#define O_CREAT  _O_CREAT
#define O_EXCL   _O_EXCL
#define O_WRONLY _O_WRONLY
#define O_TRUNC  _O_TRUNC

#endif // #ifndef __MINGW32_MSVC_FCNTL_H
