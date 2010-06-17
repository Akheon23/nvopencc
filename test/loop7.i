/*
 * a simple test
 */


__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

__attribute__((__global__)) void doit(int start, int end, int idx) {
  int i, j;
  for (j = start; j < end; j++) {
    for (i = start; i < end; i+=8) {
      data1[idx+i][j-idx]   = data2[idx+i][j-idx]   * data3[idx+i][j-idx];
      data1[idx+i+1][j-idx] = data2[idx+i+1][j-idx] * data3[idx+i+1][j-idx];
      data1[idx+i+2][j-idx] = data2[idx+i+2][j-idx] * data3[idx+i+2][j-idx];
      data1[idx+i+3][j-idx] = data2[idx+i+3][j-idx] * data3[idx+i+3][j-idx];
      data1[idx+i+4][j-idx] = data2[idx+i+4][j-idx] * data3[idx+i+4][j-idx];
      data1[idx+i+5][j-idx] = data2[idx+i+5][j-idx] * data3[idx+i+5][j-idx];
      data1[idx+i+6][j-idx] = data2[idx+i+6][j-idx] * data3[idx+i+6][j-idx];
      data1[idx+i+7][j-idx] = data2[idx+i+7][j-idx] * data3[idx+i+7][j-idx];
    }
  }
}

