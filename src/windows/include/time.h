/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_TIME_H
#define __MINGW32_MSVC_TIME_H

#include "include/time.h"

struct timespec
{
  time_t tv_sec;
  long tv_nsec;
};

#endif // #ifndef __MINGW32_MSVC_TIME_H
