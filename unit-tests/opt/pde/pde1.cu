__device__ float a, b, c;

__global__ void doit1(int start, int end) {

  float k;
  if (start == 4) {
    k = a * b + 2;
  } else if (start == 5) {
    k = a* b + 3;
  } else {
    k = a * b + 4;
  }
  c = k;
}

__global__ void doit2(int start, int end) {

  float k;
  for (int i = start; i < end; i++) {
    if (i == 4) {
      k = a * b + 5;
      a = 4;
      break;
    } else if (i == 5) {
      k = a* b + 5;
      b = 5;
      break;
    } else {
      k = a * b + 5;
      c = 99;
    }
  }
  c = k;
}

__global__ void doit3(int start, int end) {

  float k;

  for (int i = start; i < end; i++) {
    if (start == 4) {
      k = a * b + 5;
      c = 4;
    } else if (start == 5) {
      k = a* b + 5;
      c = 5;
    }
    c = k + i;
  }
}

__global__ void doit4(int start, int end) {

  float k;
  if (start > 999) goto L;
  if (start == 4) {
    k = a * b + 5;
    a = 4;
  } else if (start == 5) {
    k = a* b + 5;
    b = 5;
  } else {
L:
    k = a * b + 5;
    c = 99;
  }
  c = k;
}

__global__ void doit5(int start, int end) {

  float k;
  if (start > 999) goto L;
  if (start == 4) {
    k = a * b + 5;
    a = 4;
  } else if (start == 5) {
    k = a* b + 5;
    b = 5;
  } else {
    k = a * b + 5;
    c = 99;
  }
  c = k;
L:
;
}
