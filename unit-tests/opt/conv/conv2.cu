__global__ void conv2(int *inp, int *out)
{
  if(inp[0] == 1) {
    if(inp[2] == 3 || inp[3] == 4) {
      if(inp[4] == 4 || inp[5] == 5 || inp[6] == 6) {
        __syncthreads();
        out[2] = 4;
      }
      else {
        out[5] = 4;
      }
    }
    else {
      out[3] = 3;
    }
  }
  else if(inp[1] == 1) {
    __syncthreads();
    if((inp[9] == 3) || (inp[10] == 4 && inp[11] == 3)) {
      __syncthreads();
      out[4] = 42;
      if(inp[5] == 4 || inp[6] == 44) {
        out[4] = 455;
      }
      else {
        __syncthreads();
        out[5] = 56;
      }
    }
  }
}
