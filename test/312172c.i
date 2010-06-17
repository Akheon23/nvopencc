__attribute__((__global__)) void tgamma_main(
__attribute__((__shared__)) float x,
__attribute__((__shared__)) float *res)
{
int i;
float rr;
for (i = 0; i < 10; ++i) {
  float s;
  if (x >= 0.0f) {
    s = 1.0f / x;
    if (x > 34.03f) {
      s = s / (x - 1.0f);
    }
    rr = s;
  } else {
    s = 2.0f / x;
    if (x > -34.03f) {
      s = s / (x + 1.0f);
    }
    rr = s;
  }
}
res[i] = rr;
}

