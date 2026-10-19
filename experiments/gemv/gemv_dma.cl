// Specialized W8A8 GEMV for X2. Pack weights once as [N/32,K/4,4,8,4].
// Each 256-bit QDPA computes eight outputs with four K elements each.
typedef signed char char32 __attribute__((ext_vector_type(32)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef bool bool32 __attribute__((ext_vector_type(32)));
typedef union { int8 words; char32 bytes; } Broadcast;
static inline int requant(int value, int scale) {
    // Exact (value*scale + 2^18) >> 19 without overflowing signed int32.
    int q = ((value >> 10)*scale + (((value & 1023)*scale + 262144) >> 10)) >> 9;
    return q < -128 ? -128 : (q > 127 ? 127 : q);
}
__kernel void gemv_packed(__global const int* x,
                          __global const char32* wg, __global const char32* wu,
                          __global char* yg, __global char* yu, int k4, int n32) {
    __local char32 cache[4*896];
    __local volatile unsigned int descriptors[32] __attribute__((aligned(32)));
    int tid = __builtin_aipu_read_local_id_x();
    __local char32* w=cache+tid*896;
    __local volatile unsigned int* d=descriptors+tid*8;
    unsigned size=k4*128;
    d[0]=(size<<16)|size;d[1]=(1<<24)|size;d[2]=0;d[3]=0;d[4]=d[0];d[5]=d[1];
    for (int block = tid; block < 2*n32; block += 4) {
        int up = block >= n32;
        int nb = up ? block-n32 : block;
        __global const char32* source=(up ? wu : wg) + nb*k4*4;
        __builtin_aipu_dma_inter(0,0,0,8,1,1,0,(int)d,(int)w,(int)source);
        __builtin_aipu_wfe_inter(-2,1);
        __global char* y = (up ? yu : yg) + nb*32;
        int scale = up ? 4252 : 4772;
        int8 a0=(int8)0, a1=(int8)0, a2=(int8)0, a3=(int8)0;
        for (int k=0; k<k4; ++k) {
            Broadcast v; v.words=(int8)(x[k]);
            a0=__builtin_aipu_qdpa_tb_tb_pb(a0, w[4*k+0], v.bytes, (bool32)true);
            a1=__builtin_aipu_qdpa_tb_tb_pb(a1, w[4*k+1], v.bytes, (bool32)true);
            a2=__builtin_aipu_qdpa_tb_tb_pb(a2, w[4*k+2], v.bytes, (bool32)true);
            a3=__builtin_aipu_qdpa_tb_tb_pb(a3, w[4*k+3], v.bytes, (bool32)true);
        }
        char32 out=(char32)0;
        for (int j=0;j<8;++j) {
            out[j] = requant(a0[j],scale); out[8+j] = requant(a1[j],scale);
            out[16+j] = requant(a2[j],scale); out[24+j] = requant(a3[j],scale);
        }
        *(__global char32*)y=out;
    }
}
