/*
 * a simple test
 */

__attribute__((__shared__)) float data1[3*32][32];

__attribute__((__global__)) void doit(int start, int end) {
  int i, j;
  for (i = start; i < end; i+=8) {
    for (j = start; j < end; j++) {
      data1[i][j] = data1[i+32][j] * data1[i+2*32][j];
      data1[i+1][j] = data1[i+1+32][j] * data1[i+2*32][j];
      data1[i+2][j] = data1[i+2+32][j] * data1[i+2+2*32][j];
      data1[i+3][j] = data1[i+3+32][j] * data1[i+3+2*32][j];
      data1[i+4][j] = data1[i+4+32][j] * data1[i+4+2*32][j];
      data1[i+5][j] = data1[i+5+32][j] * data1[i+5+2*32][j];
      data1[i+6][j] = data1[i+6+32][j] * data1[i+6+2*32][j];
      data1[i+7][j] = data1[i+7+32][j] * data1[i+7+2*32][j];
    }
  }
}
