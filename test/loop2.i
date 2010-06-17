/*
 * a simple test
 */

__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

__attribute__((__global__)) void doit(int start, int end) {
  int i, j;
  for (i = start; i < end; i++) {
    for (j = start; j < end; j++) {
      data1[i][j] += (data2[i-1][j-1] + data2[i-1][j] + data2[i-1][j+1] +
		      data2[i][j-1]   + data2[i][j]   + data2[i][j+1] +
		      data2[i+1][j-1] + data2[i+1][j] + data2[i+1][j+1])/9.0;
    }
  }
}

