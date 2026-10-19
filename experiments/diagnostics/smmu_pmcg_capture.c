// One bounded MMU-700 r1p2 measurement. Does not change SMMU translation,
// global PMCG controls/filter, counter 0, IRQs, clocks or power management.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#include <linux/io.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mutex.h>
#include <linux/ktime.h>
#define PFX "opi6_pmcg_capture: "
#define START _IO('P', 1)
#define STOP _IO('P', 2)
#define FIRST 16
#define N 8
#define MASK 0x00ff0000ULL
struct unit {
    const char *name;
    phys_addr_t base, ctr;
    void __iomem *p, *v;
    u32 events[N], old_event[N], old_value[N], old_smr[N];
    u32 old0, cr, e0, s0, irq;
    u64 enabled, inten, overflow, begin_ns, end_ns;
    bool prepared;
};
static struct unit units[] = {
    {.name="tcu", .base=0xb1b2000, .ctr=0xb1d2000, .events={0,1,2,3,4,5,0x84,0x85}},
    {.name="tbu0", .base=0xb3d2000, .ctr=0xb3e2000, .events={0,1,2,0x80,0x81,0x82,0x83,0x84}},
    {.name="tbu1", .base=0xb3f2000, .ctr=0xb402000, .events={0,1,2,0x80,0x81,0x82,0x83,0x84}},
};
static DEFINE_MUTEX(lock);
static bool used, running, opened;

static bool baseline(struct unit *u)
{
    unsigned int i;
    u->cr=readl(u->p+0xe04); u->e0=readl(u->p+0x400); u->s0=readl(u->p+0xa00);
    u->irq=readl(u->p+0xe50); u->enabled=readq(u->p+0xc00);
    u->inten=readq(u->p+0xc40); u->overflow=readq(u->p+0xc80);
    pr_info(PFX "baseline unit=%s cr=%#x e0=%#x sid=%#x enabled=%#llx interrupts=%#llx overflow=%#llx\n",
            u->name,u->cr,u->e0,u->s0,u->enabled,u->inten,u->overflow);
    if(u->cr!=1 || u->e0!=1 || u->s0!=0x1e || u->enabled!=1 || u->inten || u->irq || (u->overflow&MASK))
        return false;
    for(i=0;i<N;++i) {
        u->old_event[i]=readl(u->p+0x400+4*(FIRST+i));
        u->old_smr[i]=readl(u->p+0xa00+4*(FIRST+i));
        u->old_value[i]=readl(u->v+4*(FIRST+i));
        if(u->old_event[i] || u->old_smr[i]) return false;
    }
    return true;
}
static bool stable(struct unit *u, u64 expected_enable)
{
    return readl(u->p+0xe04)==u->cr && readl(u->p+0x400)==u->e0 &&
        readl(u->p+0xa00)==u->s0 && readl(u->p+0xe50)==u->irq &&
        readq(u->p+0xc00)==expected_enable && readq(u->p+0xc40)==u->inten;
}
static bool restore(struct unit *u)
{
    unsigned int i;
    bool ok=true;
    if(!u->prepared) return true;
    writeq(MASK,u->p+0xc20);
    for(i=0;i<N;++i) {
        writel(u->old_event[i],u->p+0x400+4*(FIRST+i));
        writel(u->old_value[i],u->v+4*(FIRST+i));
    }
    // Clear only our overflow bits (verified clear before START).
    writeq(MASK,u->p+0xc80);
    ok=stable(u,u->enabled);
    for(i=0;i<N;++i)
        ok &= readl(u->p+0x400+4*(FIRST+i))==u->old_event[i] &&
              readl(u->v+4*(FIRST+i))==u->old_value[i] &&
              readl(u->p+0xa00+4*(FIRST+i))==u->old_smr[i];
    pr_info(PFX "restore unit=%s verified=%u enabled=%#llx overflow=%#llx\n",
            u->name,ok,readq(u->p+0xc00),readq(u->p+0xc80));
    u->prepared=false;
    return ok;
}
static int stop_capture(void)
{
    unsigned int k,i;
    bool ok=true;
    if(!running) return -EINVAL;
    for(k=0;k<ARRAY_SIZE(units);++k) {
        struct unit *u=&units[k];
        bool valid=stable(u,u->enabled|MASK);
        writeq(MASK,u->p+0xc20);
        u->end_ns=ktime_get_ns();
        valid &= !(readq(u->p+0xc80)&MASK);
        pr_info(PFX "result unit=%s valid_controls=%u window_ns=%llu counter0_delta=%u overflow=%#llx\n",
                u->name,valid,u->end_ns-u->begin_ns,readl(u->v)-u->old0,readq(u->p+0xc80));
        for(i=0;i<N;++i) {
            u32 e=readl(u->p+0x400+4*(FIRST+i));
            u32 s=readl(u->p+0xa00+4*(FIRST+i));
            bool same=e==u->events[i] && s==u->old_smr[i];
            pr_info(PFX "counter unit=%s index=%u event=%#x value=%u config_valid=%u\n",
                    u->name,FIRST+i,u->events[i],readl(u->v+4*(FIRST+i)),same);
            valid &= same;
        }
        ok &= valid;
    }
    for(k=0;k<ARRAY_SIZE(units);++k) ok &= restore(&units[k]);
    running=false;
    pr_info(PFX "stop valid=%u\n",ok);
    return ok?0:-EIO;
}
static int start_capture(void)
{
    unsigned int k,i;
    int ret=-EIO;
    if(used) return -EALREADY;
    used=true;
    for(k=0;k<ARRAY_SIZE(units);++k) if(!baseline(&units[k])) return -EBUSY;
    for(k=0;k<ARRAY_SIZE(units);++k) {
        struct unit *u=&units[k];
        u->prepared=true;
        for(i=0;i<N;++i) {
            writel(u->events[i],u->p+0x400+4*(FIRST+i));
            writel(0,u->v+4*(FIRST+i));
            if(readl(u->p+0x400+4*(FIRST+i))!=u->events[i] || readl(u->v+4*(FIRST+i)))
                goto failed;
        }
        if(!stable(u,u->enabled)) goto failed;
    }
    for(k=0;k<ARRAY_SIZE(units);++k) {
        struct unit *u=&units[k];
        u->old0=readl(u->v); u->begin_ns=ktime_get_ns();
        writeq(MASK,u->p+0xc00);
        if(!stable(u,u->enabled|MASK)) goto failed;
    }
    running=true;
    pr_info(PFX "start configured=1\n");
    return 0;
failed:
    for(k=0;k<ARRAY_SIZE(units);++k) restore(&units[k]);
    pr_err(PFX "start failed; no inference authorized by this collector\n");
    return ret;
}
static long capture_ioctl(struct file *f,unsigned int cmd,unsigned long arg)
{
    int ret;
    mutex_lock(&lock);
    if(cmd==START) ret=start_capture();
    else if(cmd==STOP) ret=stop_capture();
    else ret=-ENOTTY;
    mutex_unlock(&lock);
    return ret;
}
static int capture_open(struct inode *n,struct file *f)
{
    int ret=0;
    mutex_lock(&lock);
    if(opened || used) ret=-EBUSY; else opened=true;
    mutex_unlock(&lock); return ret;
}
static int capture_release(struct inode *n,struct file *f)
{
    mutex_lock(&lock); if(running) stop_capture(); opened=false; mutex_unlock(&lock); return 0;
}
static const struct file_operations ops={.owner=THIS_MODULE,.open=capture_open,.release=capture_release,.unlocked_ioctl=capture_ioctl};
static struct miscdevice misc={.minor=MISC_DYNAMIC_MINOR,.name="opi6_smmu_pmcg",.fops=&ops,.mode=0600};
static void unmap_all(void)
{
    unsigned int k;
    for(k=0;k<ARRAY_SIZE(units);++k) {
        if(units[k].p) iounmap(units[k].p);
        if(units[k].v) iounmap(units[k].v);
    }
}
static int __init capture_init(void)
{
    unsigned int k,i;
    int ret=-ENODEV;
    struct device *dev=bus_find_device_by_name(&platform_bus_type,NULL,"CIXH4000:00");
    struct iommu_fwspec *spec;
    if(!dev) return -ENODEV;
    spec=dev_iommu_fwspec_get(dev);
    if(!spec || spec->num_ids!=1 || spec->ids[0]!=0x1e) {put_device(dev);return -EINVAL;}
    put_device(dev);
    for(k=0;k<ARRAY_SIZE(units);++k) {
        struct unit *u=&units[k];
        u->p=ioremap(u->base,0x1000); u->v=ioremap(u->ctr,0x1000);
        if(!u->p || !u->v) goto fail;
        if(readl(u->p+0xe00)!=0xd01f1f || readl(u->p+0xfe0)!=(k?0x88:0x87) ||
           readl(u->p+0xfe8)!=0x1b || readl(u->p+0xfec)!=0x20) goto fail;
        pr_info(PFX "mapped unit=%s base=%#llx counters=%#llx\n",u->name,(u64)u->base,(u64)u->ctr);
        if(k) {
            void __iomem *disc=ioremap(u->base-0x2000+0x8000,0x1000);
            if(!disc) goto fail;
            for(i=0;i<15;++i)
                pr_info(PFX "discovery unit=%s sysdisc=%u value=%#x\n",u->name,i,readl(disc+0xe30+4*i));
            iounmap(disc);
        }
    }
    ret=misc_register(&misc); if(ret) goto fail;
    pr_info(PFX "ready no_counters_changed=1\n"); return 0;
fail:
    unmap_all(); return ret;
}
static void __exit capture_exit(void)
{
    misc_deregister(&misc); if(running) stop_capture(); unmap_all(); pr_info(PFX "unloaded\n");
}
module_init(capture_init);
module_exit(capture_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("One-shot MMU-700 NPU PMCG capture using free counters, preserving global state");
