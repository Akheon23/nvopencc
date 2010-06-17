static __attribute__((__shared__)) int childBuf[12];

__attribute__((__global__)) void cudaRunQueriesCUDA(
__attribute__((__shared__)) int *const index,
__attribute__((__shared__)) int *const searchResultsD,
__attribute__((__shared__)) int pos)
{
auto int docVal = index[childBuf[pos]];
while ((docVal != (-1)))
{
 ++childBuf[pos];
 docVal = index[childBuf[pos]];
}
searchResultsD[0] = docVal;
}
