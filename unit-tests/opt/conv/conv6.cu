// setup variables for calculation
__shared__ unsigned int iBAM;


#define ASK 1
#define MID 2
#define BID 3
#define TOLX 4

__device__ struct  {
int vol[200];
int errmap[200];
} optout;


__global__ void myfunc(void)
{
    int tid = threadIdx.x;
    
	// going through each type, ASK, MID, and BID
	for (unsigned int ii = 0; ii < 3; ii++) 
	{
		__syncthreads(); 
		switch(ii)
		{
		case 0:
			if (tid == 0) 
			{
				iBAM = ASK;
			}
			break;

		case 1:
			if (tid == 0)
			{
				iBAM = MID;
			}
			break;

		case 2:
			if (optout.vol[MID] > TOLX) 	// should always be true here
			{
				if (tid == 0) 
				{
					iBAM = BID;
				}
			}
			else
				continue;
			break;
		} // end switch

		__syncthreads();

		if (tid == 0)
		{
			optout.vol[iBAM] = iBAM + 1.0f;
			optout.errmap[iBAM] = iBAM + 11;
		}
		__syncthreads();
	} // end for
	
}