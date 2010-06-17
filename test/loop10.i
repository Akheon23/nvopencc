/*
 * a simple test
 */

__attribute__((__shared__)) float data1[32][32];
__attribute__((__shared__)) float data2[32][32];
__attribute__((__shared__)) float data3[32][32];

__attribute__((__global__)) void doit(int start, int end, int idx) {
  int i, j;
  for (i = start; i < end; i+=4) {
    for (j = start; j < end; j+=4) {
      data1[i][j] += (data2[i-idx][j-idx] + data2[i-idx][j] + data2[i-idx][j+idx] +
		      data2[i][j-idx]   + data2[i][j]   + data2[i][j+idx] +
		      data2[i+idx][j-idx] + data2[i+idx][j] + data2[i+idx][j+idx])/9.0;


      data1[i][j+1] += (data2[i-idx][j+1-idx] + data2[i-idx][j+1] + data2[i-idx][j+1+idx] +
		      data2[i][j+1-idx]   + data2[i][j+1]   + data2[i][j+1+idx] +
		      data2[i+idx][j+1-idx] + data2[i+idx][j+1] + data2[i+idx][j+1+idx])/9.0;


      data1[i][j+2] += (data2[i-idx][j+2-idx] + data2[i-idx][j+2] + data2[i-idx][j+2+idx] +
		      data2[i][j+2-idx]   + data2[i][j+2]   + data2[i][j+2+idx] +
		      data2[i+idx][j+2-idx] + data2[i+idx][j+2] + data2[i+idx][j+2+idx])/9.0;


      data1[i][j+3] += (data2[i-idx][j+3-idx] + data2[i-idx][j+3] + data2[i-idx][j+3+idx] +
		      data2[i][j+3-idx]   + data2[i][j+3]   + data2[i][j+3+idx] +
		      data2[i+idx][j+3-idx] + data2[i+idx][j+3] + data2[i+idx][j+3+idx])/9.0;


      data1[i+1][j] += (data2[i+1-idx][j-idx] + data2[i+1-idx][j] + data2[i+1-idx][j+idx] +
		      data2[i+1][j-idx]   + data2[i+1][j]   + data2[i+1][j+idx] +
		      data2[i+1+idx][j-idx] + data2[i+1+idx][j] + data2[i+1+idx][j+idx])/9.0;


      data1[i+1][j+1] += (data2[i+1-idx][j+1-idx] + data2[i+1-idx][j+1] + data2[i+1-idx][j+1+idx] +
		      data2[i+1][j+1-idx]   + data2[i+1][j+1]   + data2[i+1][j+1+idx] +
		      data2[i+1+idx][j+1-idx] + data2[i+1+idx][j+1] + data2[i+1+idx][j+1+idx])/9.0;


      data1[i+1][j+2] += (data2[i+1-idx][j+2-idx] + data2[i+1-idx][j+2] + data2[i+1-idx][j+2+idx] +
		      data2[i+1][j+2-idx]   + data2[i+1][j+2]   + data2[i+1][j+2+idx] +
		      data2[i+1+idx][j+2-idx] + data2[i+1+idx][j+2] + data2[i+1+idx][j+2+idx])/9.0;


      data1[i+1][j+3] += (data2[i+1-idx][j+3-idx] + data2[i+1-idx][j+3] + data2[i+1-idx][j+3+idx] +
		      data2[i+1][j+3-idx]   + data2[i+1][j+3]   + data2[i+1][j+3+idx] +
		      data2[i+1+idx][j+3-idx] + data2[i+1+idx][j+3] + data2[i+1+idx][j+3+idx])/9.0;


      data1[i+2][j] += (data2[i+2-idx][j-idx] + data2[i+2-idx][j] + data2[i+2-idx][j+idx] +
		      data2[i+2][j-idx]   + data2[i+2][j]   + data2[i+2][j+idx] +
		      data2[i+2+idx][j-idx] + data2[i+2+idx][j] + data2[i+2+idx][j+idx])/9.0;


      data1[i+2][j+1] += (data2[i+2-idx][j+1-idx] + data2[i+2-idx][j+1] + data2[i+2-idx][j+1+idx] +
		      data2[i+2][j+1-idx]   + data2[i+2][j+1]   + data2[i+2][j+1+idx] +
		      data2[i+2+idx][j+1-idx] + data2[i+2+idx][j+1] + data2[i+2+idx][j+1+idx])/9.0;


      data1[i+2][j+2] += (data2[i+2-idx][j+2-idx] + data2[i+2-idx][j+2] + data2[i+2-idx][j+2+idx] +
		      data2[i+2][j+2-idx]   + data2[i+2][j+2]   + data2[i+2][j+2+idx] +
		      data2[i+2+idx][j+2-idx] + data2[i+2+idx][j+2] + data2[i+2+idx][j+2+idx])/9.0;


      data1[i+2][j+3] += (data2[i+2-idx][j+3-idx] + data2[i+2-idx][j+3] + data2[i+2-idx][j+3+idx] +
		      data2[i+2][j+3-idx]   + data2[i+2][j+3]   + data2[i+2][j+3+idx] +
		      data2[i+2+idx][j+3-idx] + data2[i+2+idx][j+3] + data2[i+2+idx][j+3+idx])/9.0;


      data1[i+3][j] += (data2[i+3-idx][j-idx] + data2[i+3-idx][j] + data2[i+3-idx][j+idx] +
		      data2[i+3][j-idx]   + data2[i+3][j]   + data2[i+3][j+idx] +
		      data2[i+3+idx][j-idx] + data2[i+3+idx][j] + data2[i+3+idx][j+idx])/9.0;


      data1[i+3][j+1] += (data2[i+3-idx][j+1-idx] + data2[i+3-idx][j+1] + data2[i+3-idx][j+1+idx] +
		      data2[i+3][j+1-idx]   + data2[i+3][j+1]   + data2[i+3][j+1+idx] +
		      data2[i+3+idx][j+1-idx] + data2[i+3+idx][j+1] + data2[i+3+idx][j+1+idx])/9.0;


      data1[i+3][j+2] += (data2[i+3-idx][j+2-idx] + data2[i+3-idx][j+2] + data2[i+3-idx][j+2+idx] +
		      data2[i+3][j+2-idx]   + data2[i+3][j+2]   + data2[i+3][j+2+idx] +
		      data2[i+3+idx][j+2-idx] + data2[i+3+idx][j+2] + data2[i+3+idx][j+2+idx])/9.0;


      data1[i+3][j+3] += (data2[i+3-idx][j+3-idx] + data2[i+3-idx][j+3] + data2[i+3-idx][j+3+idx] +
		      data2[i+3][j+3-idx]   + data2[i+3][j+3]   + data2[i+3][j+3+idx] +
		      data2[i+3+idx][j+3-idx] + data2[i+3+idx][j+3] + data2[i+3+idx][j+3+idx])/9.0;

    }
  }
}

