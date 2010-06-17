__attribute__((__global__)) unsigned long long out;

__attribute__((__global__)) void test (
__attribute__((__shared__)) unsigned tx)
{
    unsigned i = 1;
    unsigned n = (tx & 63);
    unsigned long long prod = 1ULL;

    while (i <= n)
    {
        prod = ((unsigned long long) i) * prod;
        i += 1;
    }
    out = prod;
}
