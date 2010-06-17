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
  int iv1 = 1;
  int iv2 = 2;
  for (i = 0; i < 32; i++)
  {
    d1[idx0][iv2+idx2] = d2[idx1][iv0+idx0] + d3[idx2][iv1+idx1];
    iv0 += incr;
    iv1 += incr;
    iv2 += incr;
  }
}

__global__ void doit(int start, int end, int incr) {
  int i;
  int id0 = start;
  int id1 = start+1;
  int id2 = start+2;
  for (i = start; i < end; i++) {
    mult(data1, data1, data1, id0, id1, id2, incr);
    id0 += incr;
    id1 += incr;
    id2 += incr;
  }
}



