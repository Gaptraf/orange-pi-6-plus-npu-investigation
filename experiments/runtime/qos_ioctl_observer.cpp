// Observe the submitted ioctl descriptor without modifying it or the ioctl result.
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include "armchina_aipu.h"
static std::atomic<unsigned> schedules{0};
static aipu_job_desc captured{};
static int captured_result=-999;
extern "C" void qos_observer_reset(){schedules.store(0);captured={};captured_result=-999;}
extern "C" unsigned qos_observer_get(aipu_job_desc* desc,int* result){*desc=captured;*result=captured_result;return schedules.load();}
extern "C" int ioctl(int fd,unsigned long request,...) noexcept {
    using Fn=int(*)(int,unsigned long,...);
    static Fn next=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"ioctl"));
    if(!next)std::abort();
    // The AIPU _IO commands carry no argument; do not read absent varargs.
    if(_IOC_TYPE(request)==AIPU_IOCTL_MAGIC && _IOC_SIZE(request)==0)
        return next(fd,request);
    va_list ap;va_start(ap,request);void* arg=va_arg(ap,void*);va_end(ap);
    aipu_job_desc before{};
    const bool is_schedule=request==AIPU_IOCTL_SCHEDULE_JOB && arg;
    if(is_schedule)std::memcpy(&before,arg,sizeof(before));
    int result=next(fd,request,arg);int saved_errno=errno;
    if(is_schedule){captured=before;captured_result=result;schedules.fetch_add(1);}
    errno=saved_errno;return result;
}
