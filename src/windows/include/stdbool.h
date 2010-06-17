/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STDBOOL_H
#define __MINGW32_MSVC_STDBOOL_H

#ifndef __cplusplus

typedef char _Bool;
#define bool _Bool
#define false 0
#define true 1

#endif

#endif // #ifndef __MINGW32_MSVC_STDBOOL_H
