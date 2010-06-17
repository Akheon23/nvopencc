__global__ void conv3(int *inp, int *out)
{
  int i = 0;
  if (inp[0] == out[0]) {
    while ( i < 10) {
      int tmp = inp[i];
      if (tmp == i)
	break;
      __syncthreads();
      out[i] = tmp;
      i++;
    }
    __syncthreads();
    out[i] = 31;
  }
}
