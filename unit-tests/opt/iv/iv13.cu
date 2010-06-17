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
  int j, k, l;
  j = -1;
  k = 0;
  l = 1;
  for (i = 0; i < 31; i+=2) {

    d1[idx][i] = d2[idx+1][j] + d2[idx][j] + d2[idx-1][j] +
                 d2[idx+1][k] + d2[idx][k] + d2[idx-1][k] + 
                 d2[idx+1][l] + d2[idx][l] + d2[idx-1][l];

    d1[idx][i+1] = d2[idx+1][j+1] + d2[idx][j+1] + d2[idx-1][j+1] +
                 d2[idx+1][k+1] + d2[idx][k+1] + d2[idx-1][k+1] + 
                 d2[idx+1][l+1] + d2[idx][l+1] + d2[idx-1][l+1];
    j+=2;
    k+=2;
    l+=2;
  }
}

__global__ void doit(int start, int end) {
  int i;
  for (i = start; i < end; i++) {
    mult(data1, data2, data3, i);
  }
}



