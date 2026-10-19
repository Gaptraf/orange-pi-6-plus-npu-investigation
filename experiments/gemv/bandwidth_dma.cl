// Pure constant-buffer DMA reader, no matrix/vector arithmetic.
// Four TECs, two 28-KiB shared SRAM buffers per TEC, distinct DDR chunks.
typedef signed char char32 __attribute__((ext_vector_type(32)));
__kernel void bandwidth_dma(__global const char32* input,
                            __global const char32* stream,
                            __global char32* samples, int chunks) {
    __local char32 cache[4*2*896];
    __local volatile unsigned int descriptors[4*2*8] __attribute__((aligned(32)));
    int tid=__builtin_aipu_read_local_id_x();
    __local char32* base=cache+tid*2*896;
    __local volatile unsigned int* d=descriptors+tid*16;
    unsigned size=28672;
    for(int s=0;s<2;++s){
        d[s*8+0]=(size<<16)|size; d[s*8+1]=(1<<24)|size;
        d[s*8+2]=0; d[s*8+3]=0;
        d[s*8+4]=d[s*8+0]; d[s*8+5]=d[s*8+1];
    }
    char32 salt=input[0];
    // ASID1 is the graph's constant memory, ASID0 is ordinary tensor memory.
    __builtin_aipu_dma_inter(0,0,0,8,1,1,0,(int)d,(int)base,(int)(stream+tid*896));
    int slot=0;
    for(int block=tid;block<chunks;block+=4){
        int next=block+4,nextslot=slot^1;
        if(next<chunks)
            __builtin_aipu_dma_inter(0,nextslot*257,0,8,1,1,0,
                (int)(d+nextslot*8),(int)(base+nextslot*896),(int)(stream+next*896));
        __builtin_aipu_wfe_inter(~(1<<slot),1);
        // Check both ends of every chunk, also checking changed input propagation.
        samples[block*2]=base[slot*896]^salt;
        samples[block*2+1]=base[slot*896+895]^salt;
        slot=nextslot;
    }
}
