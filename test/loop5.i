/*
 * a simple test
 */

__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

__attribute__((__global__)) void doit(int start, int end, int idx) {
  int i, j;
  for (i = start; i < end; i++) {
    for (j = start; j < end; j++) {
      data1[i][j] += (data2[i-idx][j-idx] + data2[i-idx][j] + data2[i-idx][j+idx] +
		      data2[i][j-idx]   + data2[i][j]   + data2[i][j+idx] +
		      data2[i+idx][j-idx] + data2[i+idx][j] + data2[i+idx][j+idx])/9.0;
    }
  }
}

