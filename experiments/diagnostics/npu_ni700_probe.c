/* Read-only NI-700 hardware identification; never writes any MMIO register.
 * CIX TF-A sky1_qos.c identifies NPU ASNI pages 0xd05a000/0xd05b000
 * and MMHUB root 0xd040000. Follow only architectural child pointers.
 */
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#include <linux/io.h>
#define PFX "opi6_ni700: "
#define BASE 0x0d040000UL
#define SIZE 0x50000UL /* Stop before independently identified SYSHUB root. */
static bool seen[SIZE / 0x1000];
static unsigned int total;

static int walk(u32 off, unsigned int depth, u32 parent)
{
    void __iomem *p;
    u32 type, info, n, i, pointers[64];
    if (depth > 4 || off >= SIZE || (off & 0xfff) || seen[off >> 12])
        return -EINVAL;
    seen[off >> 12] = true;
    if (++total > ARRAY_SIZE(seen)) return -E2BIG;
    p = ioremap(BASE + off, 0x1000);
    if (!p) return -ENOMEM;
    type = readl(p); info = readl(p + 4);
    pr_info(PFX "node address=%#lx parent=%#x depth=%u type_raw=%#x info=%#x\n",
            BASE + off, parent, depth, type, info);
    if ((depth < 4 && (type & 0xffff) != depth) ||
        (depth == 4 && ((type & 0xffff) < 4 || (type & 0xffff) > 9))) {
        iounmap(p); return -ENODEV;
    }
    if (depth == 0) {
        u32 id0=readl(p+0xfe0), id1=readl(p+0xfe4), id2=readl(p+0xfe8), id3=readl(p+0xfec);
        pr_info(PFX "identity pid0=%#x pid1=%#x pid2=%#x pid3=%#x\n", id0,id1,id2,id3);
        if (((id0 & 255) | ((id1 & 15) << 8)) != 0x43b) {
            iounmap(p); return -ENODEV;
        }
    }
    if (depth == 4) { iounmap(p); return 0; }
    n = info;
    if (!n || n > ARRAY_SIZE(pointers)) { iounmap(p); return -E2BIG; }
    for (i=0; i<n; ++i) {
        pointers[i] = readl(p + 8 + i*4);
        pr_info(PFX "child parent=%#lx index=%u offset=%#x\n", BASE+off, i, pointers[i]);
        if (pointers[i] >= SIZE || (pointers[i] & 0xfff)) {
            iounmap(p); return -ERANGE;
        }
    }
    iounmap(p);
    for (i=0; i<n; ++i) {
        int ret=walk(pointers[i], depth+1, off);
        if (ret) return ret;
    }
    return 0;
}

static int __init npu_ni700_probe_init(void)
{
    const phys_addr_t npu[] = {0x0d05a000, 0x0d05b000};
    struct device *dev;
    struct iommu_fwspec *spec;
    unsigned int i;
    int ret;
    pr_info(PFX "begin read_only=1 inference_jobs=0\n");
    dev = bus_find_device_by_name(&platform_bus_type, NULL, "CIXH4000:00");
    if (!dev) return -ENODEV;
    spec = dev_iommu_fwspec_get(dev);
    if (!spec || spec->num_ids != 1 || spec->ids[0] != 0x1e) {
        put_device(dev); return -EINVAL;
    }
    put_device(dev);
    for (i=0; i<ARRAY_SIZE(npu); ++i) {
        void __iomem *p=ioremap(npu[i], 0x1000);
        u32 type, info;
        if (!p) return -ENOMEM;
        type=readl(p); info=readl(p+4);
        pr_info(PFX "npu port=%u address=%#llx type_raw=%#x info=%#x\n",
                i, (u64)npu[i], type, info);
        iounmap(p);
        if ((type & 0xffff) != 4 || !info || info == 0xffffffff) return -ENODEV;
    }
    ret=walk(0, 0, 0xffffffff);
    pr_info(PFX "end status=%d nodes=%u\n", ret, total);
    return ret;
}
static void __exit npu_ni700_probe_exit(void) { pr_info(PFX "unloaded\n"); }
module_init(npu_ni700_probe_init);
module_exit(npu_ni700_probe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Read-only CIX MMHUB NI-700 interface widths and discovery tree");
