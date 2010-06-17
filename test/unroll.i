__attribute__((__global__)) int sum;

__attribute__((__global__)) void test0 (void)
{
int i;
for (i = 0; i < 8; i++) {
   sum += i*i + i;
}
}

__attribute__((__global__)) void test1 (void)
{
int i;
#pragma unroll
for (i = 0; i < 8; i++) {
   sum += i*i + i;
}
}

__attribute__((__global__)) void test2 (void)
{
int i;
#pragma unroll 1
for (i = 0; i < 8; i++) {
   sum += i*i + i;
}
}

__attribute__((__global__)) void test3 (void)
{
int i;
#pragma unroll 2
for (i = 0; i < 8; i++) {
   sum += i*i + i;
}
}

__attribute__((__global__)) void test4 (void)
{
int i;
#pragma unroll 2
for (i = 8; i > 1; i -= 3) {
   sum += i*i + i;
}
}

