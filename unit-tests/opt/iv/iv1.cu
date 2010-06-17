/*
 * a simple test
 */

__shared__ float data1[32][32];
__shared__ float data2[32][32];
__shared__ float data3[32][32];

__device__ void mult(float d1[32][32],
		     float d2[32][32],
		     float d3[32][32],
		     int idx0,
		     int idx1,
		     int idx2) {
  int i;
  int iv0 = 0;
  int iv1 = 0;
  int iv2 = 0;
  for (i = 0; i < 32; i++) {
    d1[iv0][idx0] = d2[iv1][idx1] + d3[iv2][idx2];
    iv0 += 1;
    iv1 += 1;
    iv2 += 1;
  }
}

__global__ void doit(int start, int end) {
  int i;
  int id0 = start;
  int id1 = start;
  int id2 = start;
  for (i = start; i < end; i++) {
    mult(data1, data1, data1, id0, id1, id2);
    id0 += 1;
    id1 += 1;
    id2 += 1;
  }
}

__device__ void mult1(float d1[32][32],
		     float d2[32][32],
		     float d3[32][32],
		     int idx0,
		     int idx1,
		     int idx2) {
  int i;
  int iv0 = 0;
  int iv1 = 0;
  int iv2 = 0;
  for (i = 0; i < 32; i++) {
    d1[iv0][idx0] = d2[iv1][idx1] + d3[iv2][idx2];
    iv0 += 1;
    iv1 += 1;
    iv2 += 1;
  }
}

__global__ void doit1(int start, int end) {
  int i;
  int id0 = start;
  int id1 = start;
  int id2 = start;
  for (i = start; i < end; i++) {
    mult(data1, data1, data1, id0, id1, id2);
    id0 += 1;
    id1 += 1;
    id2 += 1;
  }
}



