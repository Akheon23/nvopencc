static void _Z19compactStreamsFinal(
unsigned short *s_left_count, 
unsigned short *s_right_count, 
unsigned *left_count, 
unsigned *right_count)
{
(s_left_count[1]) = ((unsigned short)((*left_count)));
(s_right_count[1]) = ((unsigned short)((*right_count)));
}

__attribute__((__global__)) void _Z17bisectKernelLarge(
 __attribute__((__shared__)) const unsigned tid)
{
static __attribute__((__shared__)) unsigned short s_left_count[513];
static __attribute__((__shared__)) unsigned short s_right_count[513];
auto unsigned left_count;
auto unsigned right_count;
left_count = ((unsigned)(((unsigned short *)s_left_count)[tid]));
right_count = ((unsigned)(((unsigned short *)s_right_count)[tid]));
if (0U == tid)
{
(((unsigned short *)s_left_count)[0]) = ((unsigned short)0U);
(((unsigned short *)s_right_count)[0]) = ((unsigned short)0U);
}
_Z19compactStreamsFinal(((unsigned short *)s_left_count), ((unsigned short *)s_right_count), (&left_count), (&right_count));
__syncthreads();
} 
