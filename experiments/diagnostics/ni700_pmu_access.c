/* CIX MMHUB NI-700: identify PMU and conditionally test a disabled counter.
 * No security overrides, clock operations, traffic settings, or counter start.
 * PMU address is a validated discovery leaf, not a scanned address.
 */
#include <linux/module.h>
#include <linux/io.h>
#include <linux/ioport.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#define PFX "opi6_ni_pmu_access: "
#define PMU_BASE 0x0d063000UL
#define ASNI0_BASE 0x0d05a000UL
#define ASNI1_BASE 0x0d05b000UL
static unsigned int writes;

static void putreg(u32 value, void __iomem *p)
{
    writel(value,p);
    writes++;
}

static int __init ni700_pmu_access_init(void)
{
    void __iomem *pmu=NULL, *asni[2]={NULL,NULL};
    struct device *dev;
    struct iommu_fwspec *spec;
    u32 type,cfgr,enabled,irq,overflow,cr;
    u32 old_value,old_type,old_sel[2],got_value,got_type,got_sel[2];
    unsigned int i;
    int ret=0,slot=-1;
    bool claimed=false, restored=true;
    pr_info(PFX "begin npu_jobs=0 security_writes=0\n");
    dev=bus_find_device_by_name(&platform_bus_type,NULL,"CIXH4000:00");
    if (!dev) return -ENODEV;
    spec=dev_iommu_fwspec_get(dev);
    if (!spec || spec->num_ids!=1 || spec->ids[0]!=0x1e) {
        put_device(dev); return -EINVAL;
    }
    put_device(dev);
    pmu=ioremap(PMU_BASE,0x1000);
    if (!pmu) return -ENOMEM;
    type=readl(pmu);
    pr_info(PFX "pmu address=%#lx node_type=%#x\n",PMU_BASE,type);
    if (type!=0x200006) { ret=-ENODEV; goto done; }
    cfgr=readl(pmu+0xe00);
    pr_info(PFX "pmcfgr address=%#lx value=%#x expected=%#x\n",PMU_BASE+0xe00,cfgr,0x00417f08);
    /* A read-only fixed nonzero register is the upstream Linux access test.
     * A zero value is NOT an idle or disabled PMU. Never attempt writes then.
     */
    if (!cfgr) {
        pr_info(PFX "result=pmu_access_unavailable reason=zero_fixed_pmcfgr writes=0\n");
        goto done;
    }
    if (cfgr!=0x00417f08) {
        pr_info(PFX "result=unknown_pmu_configuration writes=0\n");
        goto done;
    }
    if (!request_mem_region(PMU_BASE,0x1000,"opi6-ni700-pmu-access")) {
        pr_info(PFX "result=pmu_region_owned writes=0\n");
        goto done;
    }
    claimed=true;
    enabled=readl(pmu+0xc00); irq=readl(pmu+0xc40);
    overflow=readl(pmu+0xc80); cr=readl(pmu+0xe04);
    pr_info(PFX "state enabled=%#x interrupts=%#x overflow=%#x control=%#x\n",enabled,irq,overflow,cr);
    for (i=0;i<8;++i) {
        pr_info(PFX "counter index=%u value=%#x event_type=%#x\n",i,readl(pmu+8+i*8),readl(pmu+0x400+i*4));
        if (!((enabled|irq|overflow)&BIT(i))) slot=i;
    }
    if (slot<0) { pr_info(PFX "result=no_free_counter writes=0\n"); goto done; }
    /* No active PMU users are disturbed; keep this test conservative. */
    if (enabled || irq || overflow) {
        pr_info(PFX "result=pmu_in_use writes=0\n"); goto done;
    }
    for (i=0;i<2;++i) {
        u32 nt,ni;
        asni[i]=ioremap(i?ASNI1_BASE:ASNI0_BASE,0x1000);
        if (!asni[i]) { ret=-ENOMEM; goto done; }
        nt=readl(asni[i]); ni=readl(asni[i]+4);
        pr_info(PFX "asni port=%u type=%#x info=%#x\n",i,nt,ni);
        if (nt!=((22+i)<<16 | 4) || ni!=0x7d3) { ret=-ENODEV; goto done; }
        old_sel[i]=readl(asni[i]+0x10);
    }
    /* slot is 7 since all counters are disabled. Select only its upper byte.
     * Do not enable counting or reset global state. A denied write is WI.
     */
    if (slot!=7 || readl(pmu+0xc00)!=enabled || readl(pmu+0xc40)!=irq) {
        pr_info(PFX "result=state_changed writes=0\n"); goto done;
    }
    old_value=readl(pmu+8+7*8); old_type=readl(pmu+0x400+7*4);
    pr_info(PFX "saved slot=7 value=%#x type=%#x sel0=%#x sel1=%#x\n",old_value,old_type,old_sel[0],old_sel[1]);
    putreg(0x12345678,pmu+8+7*8);
    got_value=readl(pmu+8+7*8);
    putreg((4<<9)|22,pmu+0x400+7*4);
    got_type=readl(pmu+0x400+7*4);
    for (i=0;i<2;++i) {
        putreg((old_sel[i]&~0x3f000000U)|(0x21U<<24),asni[i]+0x10);
        got_sel[i]=readl(asni[i]+0x10);
    }
    pr_info(PFX "readback value=%#x type=%#x sel0=%#x sel1=%#x\n",got_value,got_type,got_sel[0],got_sel[1]);
    for (i=0;i<2;++i) putreg(old_sel[i],asni[i]+0x10);
    putreg(old_type,pmu+0x400+7*4);
    putreg(old_value,pmu+8+7*8);
    restored=readl(pmu+8+7*8)==old_value && readl(pmu+0x400+7*4)==old_type;
    for (i=0;i<2;++i) restored=restored && readl(asni[i]+0x10)==old_sel[i];
    restored=restored && readl(pmu+0xc00)==enabled && readl(pmu+0xc40)==irq && readl(pmu+0xc80)==overflow && readl(pmu+0xe04)==cr;
    pr_info(PFX "result=configuration_test pmu_writable=%u asni0_writable=%u asni1_writable=%u restored=%u\n",
        got_value==0x12345678 && got_type==((4<<9)|22),
        got_sel[0]==((old_sel[0]&~0x3f000000U)|(0x21U<<24)),
        got_sel[1]==((old_sel[1]&~0x3f000000U)|(0x21U<<24)),restored);
    if (!restored) ret=-EIO;
done:
    for (i=0;i<2;++i) if (asni[i]) iounmap(asni[i]);
    if (claimed) release_mem_region(PMU_BASE,0x1000);
    iounmap(pmu);
    pr_info(PFX "end status=%d writes=%u restored=%u\n",ret,writes,restored);
    return ret;
}
static void __exit ni700_pmu_access_exit(void) { pr_info(PFX "unloaded\n"); }
module_init(ni700_pmu_access_init);
module_exit(ni700_pmu_access_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("CIX MMHUB NI-700 PMU access validation without changing security");
