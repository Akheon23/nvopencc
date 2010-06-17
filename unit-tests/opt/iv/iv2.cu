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
	  int idx2,
	  int incr)
{
  int i;
  int iv0 = 0;
  int iv1 = 0;
  int iv2 = 0;
  for (i = 0; i < 32; i++)
  {
    d1[idx0][iv2] = d2[idx1][iv0] + d3[idx2][iv1];
    iv0 = iv2 + incr;
    iv1 = iv0 + incr;
    iv2 = iv1 + incr;
  }
}

__global__ void doit(int start, int end, int incr) {
  int i;
  int id0 = start;
  int id1 = start;
  int id2 = start;
  for (i = start; i < end; i++) {
    mult(data1, data1, data1, id0, id1, id2, incr);
    id0 += 1;
    id1 += 1;
    id2 += 1;
  }
}



