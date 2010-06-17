/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_UNISTD_H
#define __MINGW32_MSVC_UNISTD_H

#include "include/process.h"
#include "include/direct.h"
#include "include/io.h"

#define getcwd _getcwd
#define access _access
#define unlink _unlink
#define write _write
#define read _read
#define ftruncate _chsize

#define F_OK 0
#define X_OK 1
#define W_OK 2
#define R_OK 4

#endif // #ifndef __MINGW32_MSVC_UNISTD_H
