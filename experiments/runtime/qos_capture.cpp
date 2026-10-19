// One verified inference per process; the only experimental variable is QoS.
#define main benchmark_main_unused
#include "native_runner.cpp"
#undef main
#include <filesystem>
#include <cstddef>
#include "armchina_aipu.h"
int main(int argc,char**argv){try{
    if(argc!=6)throw std::runtime_error("usage: qos_capture graph.bin reference_dir output_dir qos case");
    std::string ref=argv[2],out=argv[3];int qos=std::stoi(argv[4]),c=std::stoi(argv[5]);
    if((qos!=0&&qos!=1)||c<0||c>=8)throw std::runtime_error("invalid qos/case");
    if(std::filesystem::exists(out))throw std::runtime_error("output already exists; refuse repeat");
    std::filesystem::create_directories(out);
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus))throw std::runtime_error("affinity");
    auto reset=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"qos_observer_reset"));
    auto observed=reinterpret_cast<unsigned(*)(aipu_job_desc*,int*)>(dlsym(RTLD_DEFAULT,"qos_observer_get"));
    if(!reset||!observed)throw std::runtime_error("ioctl observer missing");
    API api("noe");
    api.noe_npu_config.qos_level=qos==0?NOE_JOB_QOS_SLOW:NOE_JOB_QOS_HIGH;
    if(api.noe_npu_config.misc!=(unsigned(qos)<<8))throw std::runtime_error("unexpected NOE configuration layout");
    int(*ioctl_fn)(void*,int,int,void*);api.bind(ioctl_fn,"noe_ioctl");
    api.check(ioctl_fn(api.ctx,npu,NOE_IOCTL_ENABLE_TICKCOUNTER,nullptr),"enable tickcounter");
    struct Guard{API& a;decltype(ioctl_fn) fn;~Guard(){int s=fn(a.ctx,npu,NOE_IOCTL_DISABLE_TICKCOUNTER,nullptr);std::cerr<<"disable_tick_status="<<s<<std::endl;}}guard{api,ioctl_fn};
    auto bin=read(argv[1]);api.check(api.load_memory(api.ctx,(const char*)bin.data(),bin.size(),&api.graph,&api.noe_graph_config),"load");api.graph_live=true;
    uint32_t np=0,no=0;api.check(api.count(api.ctx,api.graph,NOE_TENSOR_TYPE_PROFILER,&np),"profile count");
    if(np!=1)throw std::runtime_error("expected one profiler tensor");
    Desc d{};api.check(api.desc(api.ctx,api.graph,NOE_TENSOR_TYPE_PROFILER,0,&d),"profile descriptor");
    if(d.size<16||d.size>16000000)throw std::runtime_error("unexpected profiler size");
    api.check(api.count(api.ctx,api.graph,1,&no),"output count");if(no!=3)throw std::runtime_error("expected three outputs");
    std::vector<Bytes> ys(no),expected(no);for(unsigned j=0;j<no;++j){
        Desc y{};api.check(api.desc(api.ctx,api.graph,1,j,&y),"output desc");ys[j].resize(y.size);
        expected[j]=read(ref+"/expected"+std::to_string(c)+"_"+std::to_string(j)+".bin");}
    api.create_job();auto x=read(ref+"/input"+std::to_string(c)+".bin");
    api.check(api.input(api.ctx,api.job,0,x.data()),"input");reset();auto start=Clock::now();
    api.check(api.infer(api.ctx,api.job,1000),"infer");auto elapsed=us(start,Clock::now());
    aipu_job_desc job{};int rc=-999;unsigned count=observed(&job,&rc);
    // Preserve the observed descriptor even if propagation or validation fails.
    std::ofstream evidence(out+"/submission.json");
    evidence<<"{\"qos_requested\":"<<qos<<",\"noe_misc\":"<<api.noe_npu_config.misc
       <<",\"schedule_calls\":"<<count<<",\"schedule_returncode\":"<<rc<<",\"exec_flag\":"<<job.exec_flag
       <<",\"job_id\":"<<job.job_id<<",\"head_tcb_pa\":"<<job.head_tcb_pa<<",\"aipu_version\":"<<job.aipu_version
       <<",\"ioctl_request\":"<<static_cast<unsigned long>(AIPU_IOCTL_SCHEDULE_JOB)
       <<",\"descriptor_bytes\":"<<sizeof(job)<<",\"exec_flag_offset\":"<<offsetof(aipu_job_desc,exec_flag)<<"}\n";evidence.close();
    const unsigned mask=AIPU_JOB_EXEC_FLAG_QOS_SLOW|AIPU_JOB_EXEC_FLAG_QOS_FAST;
    const unsigned wanted=qos?AIPU_JOB_EXEC_FLAG_QOS_FAST:AIPU_JOB_EXEC_FLAG_QOS_SLOW;
    if(count!=1||rc!=0||(job.exec_flag&mask)!=wanted)throw std::runtime_error("QoS not verified at kernel ioctl boundary");
    for(unsigned j=0;j<no;++j)api.check(api.output(api.ctx,api.job,1,j,ys[j].data()),"output");
    std::vector<unsigned> perm(no);std::iota(perm.begin(),perm.end(),0);bool match=false;
    do{match=true;for(unsigned j=0;j<no;++j)if(ys[j]!=expected[perm[j]])match=false;if(match)break;}while(std::next_permutation(perm.begin(),perm.end()));
    if(!match)throw std::runtime_error("output differs from integer reference");
    Bytes data(d.size);api.check(api.output(api.ctx,api.job,NOE_TENSOR_TYPE_PROFILER,0,data.data()),"get profile");
    std::ofstream f(out+"/profile.bin",std::ios::binary);f.write((char*)data.data(),data.size());if(!f)throw std::runtime_error("profile write");f.close();
    std::ofstream log(out+"/capture.json");log.precision(12);
    log<<"{\"qos\":"<<qos<<",\"case\":"<<c<<",\"profile_bytes\":"<<d.size<<",\"infer_us_with_profile\":"<<elapsed
       <<",\"bit_exact\":true,\"qos_verified\":true,\"output_mapping\":[";
    for(unsigned j=0;j<no;++j){if(j)log<<",";log<<perm[j];}log<<"]}\n";log.close();
    api.clean_job();std::cout<<"qos="<<qos<<" case="<<c<<" exec_flag="<<job.exec_flag<<" bit_exact=true\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
