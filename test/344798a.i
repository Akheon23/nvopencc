struct barracuda_matrix {
int number_of_dimensions;
int dimensions[128];
int bytes_per_element;
unsigned char *buffer;};

struct slot_info_t {
struct barracuda_matrix input_matrix;
unsigned char *slot;
int slot_id;
int element_size;
int elements_per_slot;
int number_of_sets;
int slotsize;};

struct _transfer16_t {
unsigned char buffer[16];};
typedef struct _transfer16_t transfer16_t;

static __attribute__((__device__)) unsigned char *source0;
static __attribute__((__device__)) unsigned char *allocation0;
static __attribute__((__shared__)) unsigned char raw_block_memory [];

static void _Z11test_kernelPvS_S_(
void *_a,
void *_output)
{
float *a;
float *output;
a = (float*)_a;
output = (float*)_output;
*output = *a;
*(output+1) = *(a+1);
*(output+2) = *(a+2);
}

static void _Z22single_thread_transferPvS_i(
void *_from,
void *_to,
int __val_paramsize)
{
auto unsigned char *from;
auto unsigned char *to;
from = ((unsigned char *)_from);
to = ((unsigned char *)_to);
while (__val_paramsize > 16)
{
auto transfer16_t *src;
auto transfer16_t *dst;
src = ((transfer16_t *)_from);
dst = ((transfer16_t *)_to);
(*dst) = ((*src));
from += 16;
to += 16;
__val_paramsize -= 16;
}
}

static void _Z35all_pairs_for_current_slots_Crosser16barracuda_matrix11slot_info_tiS0_i(
struct barracuda_matrix output_matrix,
struct slot_info_t slot_info_a)
{
auto unsigned char temp[128] __attribute__((aligned(4)));
auto void *output_location1;
auto void *inputa;
output_location1 = (void *)(output_matrix.buffer);
inputa = (void *)((slot_info_a.slot));
_Z11test_kernelPvS_S_(inputa, ((void *)((unsigned char *)temp)));
_Z22single_thread_transferPvS_i(((void *)((unsigned char *)temp)), output_location1, ((output_matrix.bytes_per_element)));
}

__attribute__((__global__)) void run_cartesian_product_Crosser_global(void)
{
auto struct barracuda_matrix output;
auto struct slot_info_t slot_info_a;
output.bytes_per_element = 12;
output.buffer = allocation0;
(slot_info_a.slot) = source0;
(slot_info_a.element_size) = 12;
(slot_info_a.slotsize) = 12;
(slot_info_a.slot) = raw_block_memory;
_Z35all_pairs_for_current_slots_Crosser16barracuda_matrix11slot_info_tiS0_i(output, slot_info_a);
}
