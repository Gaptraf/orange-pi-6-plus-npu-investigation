// Documented NOE API; bounded profiling, separate from speed measurements.
#define main benchmark_main_unused
#include "native_runner.cpp"
#undef main
#include <filesystem>
int main(int argc,char**argv){try{
    if(argc!=5)throw std::runtime_error("usage: profile_capture graph.bin reference_dir output_dir runs");
    std::string ref=argv[2],out=argv[3];int runs=std::stoi(argv[4]);
    if(runs<1||runs>10)throw std::runtime_error("runs must be 1..10");
    std::filesystem::create_directories(out);
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("affinity");
    API api("noe");
    int(*ioctl_fn)(void*,int,int,void*);api.bind(ioctl_fn,"noe_ioctl");
    api.check(ioctl_fn(api.ctx,npu,NOE_IOCTL_ENABLE_TICKCOUNTER,nullptr),"enable tickcounter");
    struct Guard{API& a;decltype(ioctl_fn) fn;~Guard(){int s=fn(a.ctx,npu,NOE_IOCTL_DISABLE_TICKCOUNTER,nullptr);std::cerr<<"disable_tick_status="<<s<<std::endl;}}guard{api,ioctl_fn};
    auto bin=read(argv[1]);
    api.check(api.load_memory(api.ctx,(const char*)bin.data(),bin.size(),&api.graph,&api.noe_graph_config),"load");api.graph_live=true;
    uint32_t n=0,no=0;api.check(api.count(api.ctx,api.graph,NOE_TENSOR_TYPE_PROFILER,&n),"profile count");
    if(n!=1)throw std::runtime_error("expected one profiler tensor");
    Desc d{};api.check(api.desc(api.ctx,api.graph,NOE_TENSOR_TYPE_PROFILER,0,&d),"profile descriptor");
    if(d.size<16||d.size>16000000)throw std::runtime_error("unexpected profiler size");
    api.check(api.count(api.ctx,api.graph,1,&no),"output count");
    if(no<1||no>8)throw std::runtime_error("unexpected output count");
    std::vector<Bytes> ys(no),expected(no);
    for(unsigned j=0;j<no;++j){Desc y{};api.check(api.desc(api.ctx,api.graph,1,j,&y),"output desc");ys[j].resize(y.size);}
    api.create_job();std::ofstream log(out+"/capture.json");log<<"{\"profile_bytes\":"<<d.size<<",\"runs\":[";
    for(int r=0;r<runs;++r){int c=r%8;auto x=read(ref+"/input"+std::to_string(c)+".bin");
        for(unsigned j=0;j<no;++j)expected[j]=read(ref+"/expected"+std::to_string(c)+"_"+std::to_string(j)+".bin");
        api.check(api.input(api.ctx,api.job,0,x.data()),"input");auto t=Clock::now();
        api.check(api.infer(api.ctx,api.job,1000),"infer");auto elapsed=us(t,Clock::now());
        for(unsigned j=0;j<no;++j)api.check(api.output(api.ctx,api.job,1,j,ys[j].data()),"output");
        std::vector<unsigned> perm(no);std::iota(perm.begin(),perm.end(),0);bool match=false;
        do{match=true;for(unsigned j=0;j<no;++j)if(ys[j]!=expected[perm[j]])match=false;if(match)break;}while(std::next_permutation(perm.begin(),perm.end()));
        if(!match)throw std::runtime_error("profile graph output differs from nonprofile reference");
        Bytes data(d.size);api.check(api.output(api.ctx,api.job,NOE_TENSOR_TYPE_PROFILER,0,data.data()),"get profile");
        std::string name="run"+std::to_string(r)+".bin";std::ofstream f(out+"/"+name,std::ios::binary);f.write((char*)data.data(),data.size());if(!f)throw std::runtime_error("write profile");
        if(r)log<<",";log<<"{\"case\":"<<c<<",\"file\":\""<<name<<"\",\"infer_us_with_profile\":"<<elapsed<<",\"bit_exact\":true}";log.flush();
    }
    log<<"]}\n";api.clean_job();std::cout<<"Captured "<<runs<<" verified runs, "<<d.size<<" bytes each"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
