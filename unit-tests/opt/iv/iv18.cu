/*
 * a simple test
 */

__device__ float data1[1024];
__device__ float data2[1024];
__device__ float data3[1024];

__device__ void mult(float d1[1024],
                     float d2[1024],
                     float d3[1024])
{
  int i;
  if (threadIdx.x != 0)
    return;
  for (i = 0; i < 1024; i++) {
    d1[i+i+i] = 1.0;
    d2[i+i+i] = 0.0;
    d3[i+i+i] = 2.0;
  }
}

__global__ void doit(int start, int end) {
   mult(data1, data2, data3);
}



