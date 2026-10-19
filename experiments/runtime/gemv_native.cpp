// Bounded direct runner for the captured, unchanged INT8 projection binaries.
// AIPU public declarations: Arm-China/Compass_NPU_Driver standard_api.h.
// NOE declarations and shared tensor layout: installed cix_noe_standard_api.h.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cix_noe_standard_api.h>
using Bytes=std::vector<int8_t>;
using Clock=std::chrono::steady_clock;
static double us(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::micro>(b-a).count();}
struct Desc {uint32_t id,size; float scale;int32_t zero_point;int data_type;};
struct API {
    void* handle=nullptr;void* ctx=nullptr;uint64_t graph=0,job=0;bool graph_live=false,job_live=false;
    // All optional AIPU configuration fields are zero. Reserve aligned storage
    // covering both the installed older layout and the current public layout.
    alignas(16) uint64_t aipu_default_config[16]{};
    job_config_npu_t noe_npu_config{};
    job_config_t noe_config{&noe_npu_config};
    graph_config_npu_t noe_graph_npu_config{};
    graph_config_t noe_graph_config{&noe_graph_npu_config};
    bool use_noe=false;
    int(*init)(void**); int(*deinit)(const void*);int(*error)(const void*,int,const char**);
    int(*load)(const void*,const char*,uint64_t*,void*);int(*unload)(const void*,uint64_t);
    int(*load_memory)(const void*,const char*,uint32_t,uint64_t*,void*);
    int(*create)(const void*,uint64_t,uint64_t*,void*);int(*clean)(const void*,uint64_t);
    int(*infer)(const void*,uint64_t,int32_t);int(*input)(const void*,uint64_t,uint32_t,const void*);
    int(*output)(const void*,uint64_t,int,uint32_t,void*);
    int(*count)(const void*,uint64_t,int,uint32_t*);int(*desc)(const void*,uint64_t,int,uint32_t,Desc*);
    template<class T> void bind(T& dst,const std::string& name) {
        dst=reinterpret_cast<T>(dlsym(handle,name.c_str()));if(!dst)throw std::runtime_error("missing "+name);
    }
    explicit API(const std::string& backend) {
        const bool noe=backend=="noe";
        use_noe=noe;
        const char* path=noe?"/usr/share/cix/lib/libnoe.so":"/opt/opi6-npu-llm/stage1/venv/lib/python3.11/site-packages/onnxruntime/capi/libaipu_driver.so";
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
        std::string pre=noe?"noe_":"aipu_";
        bind(init,pre+"init_context");bind(deinit,pre+"deinit_context");bind(error,pre+"get_error_message");
        bind(load,pre+"load_graph");bind(unload,pre+"unload_graph");bind(create,pre+"create_job");bind(clean,pre+"clean_job");
        bind(load_memory,pre+"load_graph_helper");
        bind(infer,pre+(noe?"job_infer_sync":"finish_job"));bind(input,pre+"load_tensor");bind(output,pre+"get_tensor");
        bind(count,pre+"get_tensor_count");bind(desc,pre+"get_tensor_descriptor");
        check(init(&ctx),"init");
    }
    void check(int r,const char* at) {if(r){const char* msg=nullptr;error(ctx,r,&msg);throw std::runtime_error(std::string(at)+" status="+std::to_string(r)+" "+(msg?msg:""));}}
    void create_job(){check(create(ctx,graph,&job,use_noe?(void*)&noe_config:(void*)aipu_default_config),"create_job");job_live=true;}
    void clean_job(){check(clean(ctx,job),"clean_job");job_live=false;}
    ~API(){if(ctx){if(job_live)clean(ctx,job);if(graph_live)unload(ctx,graph);deinit(ctx);}if(handle)dlclose(handle);}
};
static Bytes read(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("open "+path);
    auto size=f.tellg();if(size<=0)throw std::runtime_error("empty "+path);Bytes data(size);f.seekg(0);f.read((char*)data.data(),size);if(!f)throw std::runtime_error("read "+path);return data;
}
static void array(std::ostream& out,const std::vector<double>& x){out<<"[";for(size_t i=0;i<x.size();++i){if(i)out<<",";out<<x[i];}out<<"]";}
static double pct(std::vector<double> x,double q){std::sort(x.begin(),x.end());double p=q*(x.size()-1);size_t i=p;return x[i]+(x[std::min(i+1,x.size()-1)]-x[i])*(p-i);}
int main(int argc,char**argv) {
    try {
        if(argc!=4)throw std::runtime_error("usage: native_runner aipu|noe group_directory result.json");
        std::string backend=argv[1],dir=argv[2];if(backend!="aipu"&&backend!="noe")throw std::runtime_error("backend");
        cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("affinity");
        API api(backend);
        std::cerr<<"context initialized; loading graph"<<std::endl;
        if(api.use_noe){
            // NOE file loader expects a serialized CIX container. Our captured
            // file is raw AIPU ELF; the documented memory loader accepts this.
            auto binary=read(dir+"/graph0.bin");
            api.check(api.load_memory(api.ctx,(const char*)binary.data(),binary.size(),&api.graph,&api.noe_graph_config),"load_graph_helper");
        }else{
            api.check(api.load(api.ctx,(dir+"/graph0.bin").c_str(),&api.graph,nullptr),"load_graph");
        }
        api.graph_live=true;
        std::cerr<<"graph loaded; inspecting tensors"<<std::endl;
        uint32_t ni=0,no=0;api.check(api.count(api.ctx,api.graph,0,&ni),"count in");api.check(api.count(api.ctx,api.graph,1,&no),"count out");
        if(ni!=1||no<1||no>6)throw std::runtime_error("unexpected IO count");
        Desc input_desc{};api.check(api.desc(api.ctx,api.graph,0,0,&input_desc),"input descriptor");
        if(input_desc.data_type!=3)throw std::runtime_error("expected S8 input");
        std::vector<Desc> descs(no);std::vector<Bytes> ys(no);std::vector<Bytes> xs;
        std::vector<std::vector<Bytes>> expected(8);
        for(unsigned j=0;j<no;++j){api.check(api.desc(api.ctx,api.graph,1,j,&descs[j]),"output descriptor");if(descs[j].data_type!=3)throw std::runtime_error("expected S8 output");ys[j].resize(descs[j].size);}
        for(int c=0;c<8;++c){xs.push_back(read(dir+"/input"+std::to_string(c)+".bin"));if(xs.back().size()!=input_desc.size)throw std::runtime_error("input size mismatch");
            for(unsigned j=0;j<no;++j)expected[c].push_back(read(dir+"/expected"+std::to_string(c)+"_"+std::to_string(j)+".bin"));}
        auto get=[&](){for(unsigned j=0;j<no;++j)api.check(api.output(api.ctx,api.job,1,j,ys[j].data()),"get_tensor");};
        std::vector<unsigned> mapping(no);std::iota(mapping.begin(),mapping.end(),0);
        std::cerr<<"creating job"<<std::endl;
        api.create_job();api.check(api.input(api.ctx,api.job,0,xs[0].data()),"load_tensor");api.check(api.infer(api.ctx,api.job,1000),"infer");get();
        bool matched=false;
        do{matched=true;for(unsigned j=0;j<no;++j)if(ys[j]!=expected[0][mapping[j]])matched=false;if(matched)break;}while(std::next_permutation(mapping.begin(),mapping.end()));
        if(!matched){for(unsigned j=0;j<no;++j){std::ofstream f(dir+"/actual0_"+std::to_string(j)+".bin",std::ios::binary);f.write((char*)ys[j].data(),ys[j].size());std::cerr<<"OUTPUT "<<j<<" ";for(size_t k=0;k<std::min<size_t>(64,ys[j].size());++k)std::cerr<<int(ys[j][k])<<",";std::cerr<<std::endl;}throw std::runtime_error("output differs from expected integer reference");}
        int checks=0;
        auto validate=[&](int c){for(unsigned j=0;j<no;++j)if(ys[j]!=expected[c][mapping[j]])throw std::runtime_error("output mismatch case="+std::to_string(c));++checks;};
        validate(0);
        std::ofstream out(argv[3]);if(!out)throw std::runtime_error("result open");out.precision(10);
        out<<"{\"backend\":\""<<backend<<"\",\"affinity\":[0],\"input_bytes\":"<<input_desc.size<<",\"output_bytes\":[";
        for(unsigned j=0;j<no;++j){if(j)out<<",";out<<descs[j].size;}out<<"],\"native_to_ort_output\":[";
        for(unsigned j=0;j<no;++j){if(j)out<<",";out<<mapping[j];}out<<"],\"modes\":[";
        const std::vector<std::string> modes={"persistent_io","persistent_infer_only","fresh_job_io"};
        for(unsigned mode=0;mode<modes.size();++mode){
            if(mode)out<<",";
            if(mode==2)api.clean_job();
            if(mode==1)api.check(api.input(api.ctx,api.job,0,xs[0].data()),"preload");
            std::vector<double> totals,create_times,load_times,infer_times,get_times,clean_times;
            for(int i=-20;i<240;++i){
                int c=mode==1?0:(i+24)%8;
                auto t0=Clock::now();if(mode==2)api.create_job();auto t1=Clock::now();
                if(mode!=1){api.check(api.input(api.ctx,api.job,0,xs[c].data()),"load_tensor");}
                auto t2=Clock::now();
                api.check(api.infer(api.ctx,api.job,1000),"infer");auto t3=Clock::now();
                if(mode!=1){get();}
                auto t4=Clock::now();
                if(mode==2){api.clean_job();}
                auto t5=Clock::now();
                if(i>=0){totals.push_back(us(t0,t5));create_times.push_back(us(t0,t1));load_times.push_back(us(t1,t2));infer_times.push_back(us(t2,t3));get_times.push_back(us(t3,t4));clean_times.push_back(us(t4,t5));}
                // Correctness checks are outside the measured interval.
                if(mode!=1)validate(c);
            }
            if(mode==1){get();validate(0);}
            out<<"{\"name\":\""<<modes[mode]<<"\",\"runs\":240,\"p50_us\":"<<pct(totals,.5)<<",\"p95_us\":"<<pct(totals,.95)<<",\"raw_us\":";array(out,totals);
            out<<",\"create_us\":";array(out,create_times);out<<",\"load_us\":";array(out,load_times);out<<",\"infer_us\":";array(out,infer_times);out<<",\"get_us\":";array(out,get_times);out<<",\"clean_us\":";array(out,clean_times);out<<"}";out.flush();
            std::cout<<backend<<" "<<dir<<" "<<modes[mode]<<" p50_us="<<pct(totals,.5)<<" p95_us="<<pct(totals,.95)<<std::endl;
        }
        out<<"],\"validation_checks\":"<<checks<<",\"bit_exact_with_reference\":true}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
