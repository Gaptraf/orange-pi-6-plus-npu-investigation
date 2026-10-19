// W8A8 GEMV: two SRAM buffers per TEC, overlap weight DMA with QDPA.
typedef signed char char32 __attribute__((ext_vector_type(32)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef bool bool32 __attribute__((ext_vector_type(32)));
typedef union {int8 words;char32 bytes;} Broadcast;
static inline int requant(int value,int scale){
    int q=((value>>10)*scale+(((value&1023)*scale+262144)>>10))>>9;
    return q < -128 ? -128 : (q>127 ? 127:q);
}
__kernel void gemv_packed(__global const int* x,__global const char32* wg,
                         __global const char32* wu,__global char* yg,__global char* yu,
                         int k4,int n32){
    __local char32 cache[4*2*896]; // 229376 bytes, within 256 KiB shared SRAM.
    __local volatile unsigned int descriptors[4*2*8] __attribute__((aligned(32)));
    int tid=__builtin_aipu_read_local_id_x();
    __local char32* base=cache+tid*2*896;
    __local volatile unsigned int* d=descriptors+tid*16;
    unsigned size=k4*128;
    for(int s=0;s<2;++s){
        d[s*8+0]=(size<<16)|size;d[s*8+1]=(1<<24)|size;d[s*8+2]=0;
        d[s*8+3]=0;d[s*8+4]=d[s*8+0];d[s*8+5]=d[s*8+1];
    }
    if(tid>=2*n32)return;
    int first_up=tid>=n32;
    __global const char32* first=(first_up?wu:wg)+(first_up?tid-n32:tid)*k4*4;
    __builtin_aipu_dma_inter(0,0,0,8,1,1,0,(int)d,(int)base,(int)first);
    int slot=0;
    for(int block=tid;block<2*n32;block+=4){
        int next=block+4,nextslot=slot^1;
        if(next<2*n32){
            int up=next>=n32;
            __global const char32* src=(up?wu:wg)+(up?next-n32:next)*k4*4;
            __builtin_aipu_dma_inter(0,nextslot*257,0,8,1,1,0,(int)(d+nextslot*8),
                                      (int)(base+nextslot*896),(int)src);
        }
        __builtin_aipu_wfe_inter(~(1<<slot),1);
        __local const char32* w=base+slot*896;
        int8 a0=(int8)0,a1=(int8)0,a2=(int8)0,a3=(int8)0;
        for(int k=0;k<k4;++k){
            Broadcast v;v.words=(int8)(x[k]);
            a0=__builtin_aipu_qdpa_tb_tb_pb(a0,w[4*k+0],v.bytes,(bool32)true);
            a1=__builtin_aipu_qdpa_tb_tb_pb(a1,w[4*k+1],v.bytes,(bool32)true);
            a2=__builtin_aipu_qdpa_tb_tb_pb(a2,w[4*k+2],v.bytes,(bool32)true);
            a3=__builtin_aipu_qdpa_tb_tb_pb(a3,w[4*k+3],v.bytes,(bool32)true);
        }
        int up=block>=n32,nb=up?block-n32:block,scale=up?4252:4772;
        char32 out=(char32)0;
        for(int j=0;j<8;++j){
            out[j]=requant(a0[j],scale);out[8+j]=requant(a1[j],scale);
            out[16+j]=requant(a2[j],scale);out[24+j]=requant(a3[j],scale);
        }
        *((__global char32*)(up?yu:yg)+nb)=out;
        slot=nextslot;
    }
}
