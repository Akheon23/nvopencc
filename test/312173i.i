
__attribute__((__global__)) void ilogb_main(
__attribute__((__shared__)) int a,
__attribute__((__shared__)) int *res)
{
  unsigned int i;
  unsigned int j;
  int expo;

  for (j = 0; j < 100; ++j) {
    if (a == 0) {
      expo = -1;
    } else {
      expo = -126;
      i = a & 0xff;
      i = i << 8;
      expo = expo | i;
    }
    res[j] = expo;
  }
}

