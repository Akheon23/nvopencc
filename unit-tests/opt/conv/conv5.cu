__global__ void foo(int *inp, int *out)
{
  if(inp[0] == 1) {
    if(inp[2] == 2) {
      out[3] = 3;
    }
    else {
      __syncthreads();
    }
  }
  else {
    if(inp[3] == 4) {
      out[0] = 4;
      __syncthreads();
    }
    else {
      if(inp[4] == 5) {
        out[5] = 5;
      }
      else if(inp[5] == 5) {
        out[4] = inp[5] + 4;
	int i = 0;
	while (inp[i]) {
	  if(inp[14] == 66) {
	    out[44] = 5;
	    __syncthreads();
	  }
	  else {
	    out[32] = 5;
	  }
	  i++;
	}
        __syncthreads();
	out[99] = 99;
      }
      else {
        __syncthreads();
        out[4] = 5;
      }
    }
  }
  out[22] = 22;
}
