__global__ void conv8(int *inp, int *out)
{
  int i;
  int sum = 0;

  do {
    sum += inp[i];
    i++;
  } while(i < inp[threadIdx.x]);

  __syncthreads();
  out[0] = sum;
}
