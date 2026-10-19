/* Read existing NI-700 configuration only. Never modify security or QoS.
 * Arm TRM: denied Non-secure reads return zero without an error response.
 * Zero values must not be interpreted as disabled regulators without access proof.
 */
#include <linux/module.h>
#include <linux/io.h>
#define PFX "opi6_ni700_cfg: "
static int __init npu_ni700_config_init(void)
{
    const phys_addr_t bases[] = {0x0d05a000, 0x0d05b000};
    const u32 offsets[] = {0x14,0x44,0x48,0x84,0x88,0x8c,0x90,
        0x94,0x98,0x9c,0xa0,0xa4,0xa8,0xac,0xb0,0xb4,0xb8,
        0xbc,0xc0,0xc4,0xc8,0xcc,0xd0};
    unsigned int i,j;
    pr_info(PFX "begin read_only=1 inference_jobs=0\n");
    for (i=0;i<ARRAY_SIZE(bases);++i) {
        void __iomem *p=ioremap(bases[i],0x1000);
        u32 type,info;
        if (!p) return -ENOMEM;
        type=readl(p); info=readl(p+4);
        pr_info(PFX "port=%u address=%#llx type=%#x info=%#x\n",i,(u64)bases[i],type,info);
        if (type != ((22+i)<<16 | 4) || info != 0x7d3) {
            iounmap(p); return -ENODEV;
        }
        for (j=0;j<ARRAY_SIZE(offsets);++j)
            pr_info(PFX "port=%u offset=%#x value=%#x\n",i,offsets[j],readl(p+offsets[j]));
        iounmap(p);
    }
    pr_info(PFX "end status=0\n");
    return 0;
}
static void __exit npu_ni700_config_exit(void) { pr_info(PFX "unloaded\n"); }
module_init(npu_ni700_config_init);
module_exit(npu_ni700_config_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Read-only NPU NI-700 configuration snapshot; no access override");
