/*
 * a simple test
 */

__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

static void mult(float d1[32][32],
		float d2[32][32],
		float d3[32][32],
		int idx) {
  int i;
  for (i = 0; i < 32; i++) {
    d1[idx][i] = d2[idx][i] + d3[idx][i];

    // The following should never return
    // Currently open64 skips strength reduction on
    // this loop.

    if (i > 32) return;
  }
}

__attribute__((__global__)) void doit(
 __attribute__((__shared__)) int start, 
 __attribute__((__shared__)) int end) 
{
  int i;
  for (i = start; i < end; i++) {
    mult(data1, data2, data3, i);
  }
}

