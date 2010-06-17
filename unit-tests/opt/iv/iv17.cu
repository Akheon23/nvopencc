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
  int j, k, l;
  if (threadIdx.x != 0)
    return;
  j = 0;
  k = 0;
  l = 1;
  for (i = 0; i < 1024; i++) {
    d1[j+k+l] = 1.0;
    d2[j+k+l] = 0.0;
    d3[j+k+l] = 2.0;
    j++;
    k++;
    l++;
  }
}

__global__ void doit(int start, int end) {
   mult(data1, data2, data3);
}



