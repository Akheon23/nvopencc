/* Replacement file for Visual Studio. */
#ifndef __MINGW32_MSVC_STDINT_H
#define __MINGW32_MSVC_STDINT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int pid_t;
typedef long off_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long uint32_t;
typedef unsigned long long uint64_t;

typedef signed char int8_t;
typedef signed short int16_t;
typedef signed long int32_t;
typedef signed long long int64_t;

#define INT8_MIN (-128)
#define INT16_MIN (-32768)
#define INT32_MIN (-2147483647 - 1)
#define INT64_MIN (-9223372036854775807LL - 1LL)

#define INT8_MAX (127)
#define INT16_MAX (32767)
#define INT32_MAX (2147483647)
#define INT64_MAX (9223372036854775807LL)

#define UINT8_MAX (255)
#define UINT16_MAX (65535)
#define UINT32_MAX (4294967295UL)
#define UINT64_MAX (18446744073709551615ULL)

#ifdef __cplusplus
}
#endif

#endif // #ifndef __MINGW32_MSVC_STDINT_H
