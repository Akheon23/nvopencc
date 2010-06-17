//
//  This is the simple test kernel
//

#include <stdio.h>
#include <cuda_runtime.h>

// This crashes the compiler!
//__device__ static char *s1="a", *s2="b";

__device__ static char *copyOneChar(char *dest, const char *src)
{
    // The while is needed, as is the "src++", because it forces the
    // compiler to load "src" into a register. Without the while,
    // the load never happens because the loop is irrelevant.
    int n = 1;
    while(n-- > 0)
        *dest++ = *src++;

    return dest;
}

// The indirection is required because it makes the compiler think that
// "src" is global, not const. Dunno why.
__device__ static int copyIndirect(char *dest, const char *src)
{
    copyOneChar(dest, src);
    return 1;
}

__global__ void cuCopyTest()
{
    const char *s1="a", *s2="b";      // These string constants are causing the trouble
    char out;

    // Look into the .ptx code and you see that "s1" is taken
    // as global (ld.global.s8), while "s2" is taken as const
    // (ld.const.s8). If we add an s3, that's taken as const too.
    copyOneChar((char *)&out, s1);
    copyOneChar((char *)&out, s2);     // <-- change this to ", s1" and the sync works
}


///////////////////////////////////////////////////////////////////////////////
// HOST SIDE

int main(int argc, char *argv[])
{
    // Launch the kernel...
    cuCopyTest<<< 1, 1 >>>();
  
    // Now try to synchronise with everything
    if(cudaThreadSynchronize() != cudaSuccess)
        printf("Failed to synchronize!\n");
    else
        printf("Synchronization OK.\n");

    return 0;
}
