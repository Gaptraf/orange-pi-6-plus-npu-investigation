// Read-only architectural SMMUv3 table walk, checked against the kernel API.
// No private kernel layouts, clocks, DMA mappings, resets or counter writes.
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/bitfield.h>
#include <linux/sched.h>

#define PFX "opi6_smmu_path: "
#define TABLE_ADDR GENMASK_ULL(47, 12)
#define SID 0x1e
static u64 table_copy[4][512];
static u64 leaves[4], mapped[4], attrs[8], contiguous_hint, tables, checked, mismatches;
static u64 run_start, run_end, run_count;
static int run_level = -1;
static struct iommu_domain *domain;

static int ram_copy(u64 phys, void *out, size_t len)
{
    if (!len || phys + len < phys || !pfn_valid(PHYS_PFN(phys)) ||
        !pfn_valid(PHYS_PFN(phys + len - 1))) return -EINVAL;
    if (!virt_addr_valid(phys_to_virt(phys)) || !virt_addr_valid(phys_to_virt(phys + len - 1)))
        return -EINVAL;
    return copy_from_kernel_nofault(out, phys_to_virt(phys), len);
}
static void flush_run(void)
{
    if (run_level < 0) return;
    pr_info(PFX "mapping iova=%#llx end=%#llx level=%d leaves=%llu\n",
            run_start, run_end, run_level, run_count);
    run_level = -1;
}
static int walk(u64 table, unsigned int level, u64 base, unsigned int entries)
{
    unsigned int i, shift = 12 + 9 * (3 - level);
    int ret;
    if (level > 3 || entries > 512 || ++tables > 16384) return -E2BIG;
    ret = ram_copy(table, table_copy[level], entries * 8);
    if (ret) return ret;
    for (i = 0; i < entries; ++i) {
        u64 pte = le64_to_cpu(table_copy[level][i]);
        u64 iova = base + ((u64)i << shift), size = 1ULL << shift;
        if (!(pte & 1)) continue;
        if (level < 3 && (pte & 3) == 3) {
            ret = walk(pte & TABLE_ADDR, level + 1, iova, 512);
            if (ret) return ret;
        } else {
            u64 phys = (pte & TABLE_ADDR) & ~(size - 1);
            if (!level || (level == 3 ? (pte & 3) != 3 : (pte & 3) != 1)) return -EINVAL;
            if (++leaves[level] > 4000000) return -E2BIG;
            mapped[level] += size;
            attrs[(pte >> 2) & 7] += size;
            if (pte & BIT_ULL(52)) ++contiguous_hint;
            // Sample each early leaf and every 256th thereafter, checking both ends.
            if (leaves[level] <= 64 || !(leaves[level] & 255)) {
                phys_addr_t first = iommu_iova_to_phys(domain, iova);
                phys_addr_t last = iommu_iova_to_phys(domain, iova + size - 1);
                ++checked;
                if (first != phys || last != phys + size - 1) {
                    ++mismatches;
                    pr_info(PFX "mismatch iova=%#llx pte=%#llx expected=%#llx actual=%#llx/%#llx\n",
                            iova, pte, phys, (u64)first, (u64)last);
                    return -EIO;
                }
            }
            if (run_level != level || run_end != iova) {
                flush_run(); run_start = iova; run_level = level; run_count = 0;
            }
            run_end = iova + size; ++run_count;
        }
    }
    cond_resched();
    return 0;
}
static int __init smmu_path_probe_init(void)
{
    struct device *dev = NULL, *smmu_dev = NULL;
    struct resource *resource;
    void __iomem *regs = NULL, *pmcg = NULL;
    struct iommu_fwspec *spec;
    u64 root, desc, ste[8], cd[8], ste_phys, cd_phys, ttbr;
    u32 cfg, fmt, split, ias, start_level, entries, cr0;
    int i, ret = -ENODEV;
    pr_info(PFX "begin read_only=1\n");
    dev = bus_find_device_by_name(&platform_bus_type, NULL, "CIXH4000:00");
    smmu_dev = bus_find_device_by_name(&platform_bus_type, NULL, "arm-smmu-v3.1.auto");
    if (!dev || !smmu_dev) goto done;
    spec = dev_iommu_fwspec_get(dev);
    domain = iommu_get_domain_for_dev(dev);
    if (!domain || domain->type != IOMMU_DOMAIN_DMA || !spec || spec->num_ids != 1 || spec->ids[0] != SID)
        goto done;
    resource = platform_get_resource(to_platform_device(smmu_dev), IORESOURCE_MEM, 0);
    if (!resource || resource->start != 0x0b1b0000 || resource_size(resource) < 0x3000) goto done;
    regs = ioremap(resource->start, 0x1000);
    if (!regs) goto done;
    cr0 = readl(regs + 0x20);
    root = readq(regs + 0x80);
    cfg = readl(regs + 0x88);
    pr_info(PFX "smmu_base=%#llx idr0=%#x idr1=%#x idr5=%#x cr0=%#x strtab_base=%#llx strtab_cfg=%#x\n",
            (u64)resource->start, readl(regs), readl(regs + 4), readl(regs + 0x14), cr0, root, cfg);
    if (!(cr0 & 1)) goto done;
    fmt = (cfg >> 16) & 3; split = (cfg >> 6) & 31;
    root &= GENMASK_ULL(51, 6);
    if (fmt == 0) ste_phys = root + SID * 64;
    else if (fmt == 1 && split >= 6 && split <= 10) {
        ret = ram_copy(root + (SID >> split) * 8, &desc, 8);
        if (ret) goto done;
        desc = le64_to_cpu(desc);
        if (!(desc & 31)) { ret = -EINVAL; goto done; }
        ste_phys = (desc & GENMASK_ULL(51, 6)) + (SID & ((1U << split) - 1)) * 64;
    } else { ret = -EINVAL; goto done; }
    ret = ram_copy(ste_phys, ste, sizeof(ste)); if (ret) goto done;
    for (i = 0; i < 8; ++i) { ste[i] = le64_to_cpu(ste[i]); pr_info(PFX "ste index=%d value=%#llx\n", i, ste[i]); }
    if (!(ste[0] & 1) || ((ste[0] >> 1) & 7) != 5) { ret = -EINVAL; goto done; }
    cd_phys = ste[0] & GENMASK_ULL(51, 6);
    fmt = (ste[0] >> 4) & 3;
    if (fmt == 2) {
        ret = ram_copy(cd_phys, &desc, 8); if (ret) goto done;
        desc = le64_to_cpu(desc);
        if (!(desc & 1)) { ret = -EINVAL; goto done; }
        cd_phys = desc & GENMASK_ULL(51, 12);
    } else if (fmt != 0) { ret = -EINVAL; goto done; }
    ret = ram_copy(cd_phys, cd, sizeof(cd)); if (ret) goto done;
    for (i = 0; i < 8; ++i) { cd[i] = le64_to_cpu(cd[i]); pr_info(PFX "cd index=%d value=%#llx\n", i, cd[i]); }
    if (!(cd[0] & BIT_ULL(31)) || !(cd[0] & BIT_ULL(41)) || (cd[0] & (BIT_ULL(14) | BIT_ULL(15))) ||
        ((cd[0] >> 6) & 3)) { ret = -EINVAL; goto done; }
    ias = 64 - (cd[0] & 63);
    if (ias < 22 || ias > 48) { ret = -EINVAL; goto done; }
    start_level = 4 - DIV_ROUND_UP(ias - 12, 9);
    entries = 1U << (ias - (12 + 9 * (3 - start_level)));
    ttbr = cd[1] & GENMASK_ULL(51, 4);
    pr_info(PFX "translation sid=%#x granule=4096 ias=%u start_level=%u root_entries=%u ttbr=%#llx\n",
            SID, ias, start_level, entries, ttbr);
    ret = walk(ttbr, start_level, 0, entries);
    flush_run();
    for (i = 0; i < 4; ++i) pr_info(PFX "summary level=%d leaves=%llu bytes=%llu\n", i, leaves[i], mapped[i]);
    for (i = 0; i < 8; ++i) if (attrs[i]) pr_info(PFX "attribute index=%d bytes=%llu\n", i, attrs[i]);
    pr_info(PFX "walk_result status=%d tables=%llu checked_leaf_pairs=%llu mismatches=%llu contiguous_hints=%llu\n",
            ret, tables, checked, mismatches, contiguous_hint);
    if (ret) goto done;
    // TCU PMCG lives within the live SMMU ACPI resource; read IDs only.
    pmcg = ioremap(resource->start + 0x2000, 0x1000);
    if (!pmcg) { ret = -ENOMEM; goto done; }
    pr_info(PFX "tcu_pmcg base=%#llx cfgr=%#x cr=%#x iidr=%#x ceid0=%#llx ceid1=%#llx\n",
            (u64)resource->start + 0x2000, readl(pmcg + 0xe00), readl(pmcg + 0xe04),
            readl(pmcg + 0xe08), readq(pmcg + 0xe20), readq(pmcg + 0xe28));
done:
    if (pmcg) iounmap(pmcg);
    if (regs) iounmap(regs);
    if (smmu_dev) put_device(smmu_dev);
    if (dev) put_device(dev);
    pr_info(PFX "end status=%d\n", ret);
    return ret;
}
static void __exit smmu_path_probe_exit(void) { pr_info(PFX "unloaded\n"); }
module_init(smmu_path_probe_init);
module_exit(smmu_path_probe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Read-only NPU SMMU page-size and TCU PMCG identification probe");
