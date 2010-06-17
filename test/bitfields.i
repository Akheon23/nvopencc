typedef struct {
	signed a : 4;
	unsigned b : 6;
	int c : 6;
} sbt;

__attribute__((__shared__)) struct {
   signed x:2;
   signed y:2;
   signed z:2;
} MyStruct;

__attribute__((__shared__)) sbt x;

__attribute__((__global__)) void test (void)
{
	x.a = x.b + x.c;
	MyStruct.x = MyStruct.z;
}
