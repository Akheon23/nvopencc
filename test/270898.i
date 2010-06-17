typedef struct float3
{
  float x, y, z;
} float3;

typedef struct __attribute__((aligned(16))) float4
{
  float x, y, z, w;
} float4;

__attribute__((__shared__)) float sf; 
__attribute__((__global__)) float3 gf3; 
__attribute__((__global__)) float4 gf; 
__attribute__((__global__)) float4 s[100]; 
__attribute__((__global__)) float4 x[100]; 

__attribute__((__global__)) void test1 (void)
{
float t;
t = gf.x + gf.y + gf.z;
sf = t;
}

__attribute__((__global__)) void test2 (void)
{
float t;
t = gf.x + gf.y + gf.z;
sf = t;
}

__attribute__((__global__)) void test3 (
__attribute__((__shared__)) int i,
__attribute__((__shared__)) int expr)
{
s[i] = x[i]; 
s[i].x += expr; 
s[i].y += expr; 
s[i].z += expr; 
x[i+1] = s[i]; 
}

__attribute__((__global__)) void test4 (
__attribute__((__shared__)) int i,
__attribute__((__shared__)) int expr)
{
float4 t;
t = x[i]; 
t.x += expr; 
t.y += expr; 
t.z += expr; 
x[i+1] = t; 
}

