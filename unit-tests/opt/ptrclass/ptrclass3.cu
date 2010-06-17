#include <stdio.h>
#include <cuda_runtime.h>
__device__ char xx;

__global__ void cuCopyTest( char *s1, char *s2)
{
    char out;
    char * dest;
    char * src;
    int n;

    dest = &xx;
    src = s1;
    n = 1;
    while(n-- > 0)
        *dest++ = *src++;

    dest = &out;
    src = s2;
    n = 1;
    while(n-- > 0)
        *dest++ = *src++;

}


