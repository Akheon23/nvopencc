/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_SYS_STAT_H
#define __MINGW32_MSVC_SYS_STAT_H

#include "include/sys/stat.h"

#define _S_IFBLK        0x3000

#define S_IRWXU         (_S_IREAD | _S_IWRITE | _S_IEXEC)
#define S_IXUSR         _S_IEXEC
#define S_IWUSR         _S_IWRITE
#define S_IRUSR         _S_IREAD

#define S_ISCHR(MODE)   (((MODE) & _S_IFMT) == _S_IFCHR)
#define S_ISDIR(MODE)   (((MODE) & _S_IFMT) == _S_IFDIR)
#define S_ISFIFO(MODE)  (((MODE) & _S_IFMT) == _S_IFIFO)
#define S_ISREG(MODE)   (((MODE) & _S_IFMT) == _S_IFREG)
#define S_ISBLK(MODE)   (((MODE) & _S_IFMT) == _S_IFBLK)

#define S_IFCHR _S_IFCHR
#define S_IFDIR _S_IFDIR
#define S_IFREG _S_IFREG

#define stat _stat
#define fstat _fstat

#endif // #ifndef __MINGW32_MSVC_SYS_STAT_H
