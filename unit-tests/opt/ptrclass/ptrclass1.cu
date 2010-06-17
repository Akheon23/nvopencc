#include <stdio.h>
#include <cuda_runtime.h>
__device__ char xx[23];
__shared__ char s2[23];
__global__ void cuCopyTest( char *s1, int start, int end)
{
    char out[23];
    char * dest;
    char * src;
    int n = end;

    // initialize shared memory s2 from xxx;
    dest = &s2[start];
    n = end;
    src = &xx[start];
    while(n-- > 0)
        *dest++ = *src++;

    __syncthreads();
    
    dest = (&xx[0]+start)+1;
    n = end;
    src = s1;
    while(n-- > 0)
        *dest++ = *src++;

    dest = (&out[0])+start;
    n = end;
    src = &s2[start];
    while(n-- > 0)
        *dest++ = *src++;

}




