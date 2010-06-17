/*
 * a simple test
 */

__shared__ float data1[32][32];
__shared__ float data2[32][32];
__shared__ float data3[32][32];

__device__ void mult(__shared__ float d1[32][32],
                     __shared__ float d2[32][32],
                     __shared__ float d3[32][32],
                     int idx) 
{
  int i;
  for (i = 0; i < 31; i++) {
    d1[idx][i] = d2[idx+1][i-1] + d2[idx][i-1] + d2[idx-1][i-1] +
                 d2[idx+1][i]   + d2[idx][i]   + d2[idx-1][i]   + 
                 d2[idx+1][i+1] + d2[idx][i+1] + d2[idx-1][i+1];
  }
}

__global__ void doit(int start, int end) {
  int i;
  for (i = start; i < end; i++) {
    mult(data1, data2, data3, i);
  }
}



