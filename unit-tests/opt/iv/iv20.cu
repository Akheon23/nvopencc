/*
 * a simple test
 */

__shared__ float data1[32];
__shared__ float data2[32];
__shared__ float data3[32];

__device__ void mult(__shared__ float d1[32],
                     __shared__ float d2[32],
                     __shared__ float d3[32],
                     int idx) 
{
  int i;
  int j, k, l;
  j = 0;
  k = 0;
  l = 0;
  for (i = 0; i < 1024; i++) {
    j++;
    k++;
    l++;
    d1[i+j+k+l] = 1.0;
  }
}

__global__ void doit(int start, int end) {
  int i = 99;
    mult(data1, data2, data3, i);
}



