 
typedef struct __attribute__((aligned(4)))
{
  unsigned char x, y, z, w;
} uchar4;

typedef union TData {
uchar4 ub4;
int ival;} TData;

__attribute__((__global__)) void testKernel(
__attribute__((__shared__)) TData *const d_odata, 
__attribute__((__shared__)) TData *const d_idata, 
__attribute__((__shared__)) int tid, 
__attribute__((__shared__)) int thread_n, 
__attribute__((__shared__)) const int data_size){
{
auto int pos;
for (pos = tid; (pos < data_size); pos += thread_n)
{
	auto TData data;
        data = d_idata[pos];
        data.ub4.x++;
        data.ub4.y++;
        data.ub4.z++;
        data.ub4.w++;
        d_odata[pos] = data;
} 
}}
