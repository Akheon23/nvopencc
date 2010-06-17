/*
 * a simple test
 */

__shared__ float data1[32];
__shared__ float data2[32];
__shared__ float data3[32];

__global__ void doit(int start, int end) {
  int i;
  for (i = 0; i < end; i++) {
     data1[i-start] = data2[i-start] + data3[i-start];
  }
}

__global__ void doit1(int start, int end) {
  int i;
  float * p1 = &data2[2];
  for (i = 0; i < end; i++) {
     data1[i-start] = p1[i-start-2] + data3[i-start];
  }
}



