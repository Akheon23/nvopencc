/*
 * a simple test
 */


__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

__attribute__((__global__)) void doit(int start, int end, int idx) {
  int i, j;
  for (j = start; j < end; j+=4) {
    for (i = start; i < end; i+=4) {

      data1[idx+i][j-idx]   = data2[idx+i][j-idx]   * data3[idx+i][j-idx]; 
      data1[idx+i][j+1-idx]   = data2[idx+i][j+1-idx]   * data3[idx+i][j+1-idx]; 
      data1[idx+i][j+2-idx]   = data2[idx+i][j+2-idx]   * data3[idx+i][j+2-idx]; 
      data1[idx+i][j+3-idx]   = data2[idx+i][j+3-idx]   * data3[idx+i][j+3-idx];

      data1[idx+i+1][j-idx]   = data2[idx+i+1][j-idx]   * data3[idx+i+1][j-idx]; 
      data1[idx+i+1][j+1-idx]   = data2[idx+i+1][j+1-idx]   * data3[idx+i+1][j+1-idx]; 
      data1[idx+i+1][j+2-idx]   = data2[idx+i+1][j+2-idx]   * data3[idx+i+1][j+2-idx]; 
      data1[idx+i+1][j+3-idx]   = data2[idx+i+1][j+3-idx]   * data3[idx+i+1][j+3-idx];

      data1[idx+i+2][j-idx]   = data2[idx+i+2][j-idx]   * data3[idx+i+2][j-idx]; 
      data1[idx+i+2][j+1-idx]   = data2[idx+i+2][j+1-idx]   * data3[idx+i+2][j+1-idx]; 
      data1[idx+i+2][j+2-idx]   = data2[idx+i+2][j+2-idx]   * data3[idx+i+2][j+2-idx]; 
      data1[idx+i+2][j+3-idx]   = data2[idx+i+2][j+3-idx]   * data3[idx+i+2][j+3-idx];

      data1[idx+i+3][j-idx]   = data2[idx+i+3][j-idx]   * data3[idx+i+3][j-idx]; 
      data1[idx+i+3][j+1-idx]   = data2[idx+i+3][j+1-idx]   * data3[idx+i+3][j+1-idx]; 
      data1[idx+i+3][j+2-idx]   = data2[idx+i+3][j+2-idx]   * data3[idx+i+3][j+2-idx]; 
      data1[idx+i+3][j+3-idx]   = data2[idx+i+3][j+3-idx]   * data3[idx+i+3][j+3-idx];

    }
  }
}

