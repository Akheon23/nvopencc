struct cublasSgemmParams {
const float *B;
float *C;
unsigned k;
unsigned ldb;};

static __attribute__((__shared__)) float BB[1056];

__attribute__((__global__)) void sgemm_main_gld_hw_na_nb_fulltile(
__attribute__((__shared__)) const struct cublasSgemmParams parms,
__attribute__((__shared__)) unsigned tid)
{
auto unsigned l;
auto unsigned idxBB;
idxBB = tid;
for (l = 0U; (l < parms.k); l += 32U)
{
 auto unsigned addrB;
 ++idxBB;
 addrB = (parms.ldb + l);
 BB[idxBB] = parms.B[addrB];
}
*parms.C = BB[tid];
}
