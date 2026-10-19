// Read cached SCMI performance operating points and GET memory-path limits.
#include <linux/module.h>
#include <linux/device.h>
#include <linux/scmi_protocol.h>
#include <linux/err.h>
#include <linux/string.h>
#include <linux/ctype.h>
extern struct bus_type scmi_bus_type;

static int match_perf_device(struct device *dev, const void *unused)
{
    struct scmi_device *sdev = to_scmi_dev(dev);
    return sdev->protocol_id == SCMI_PROTOCOL_PERF && sdev->handle;
}
static int __init scmi_perf_get_probe_init(void)
{
    struct device *dev;
    struct scmi_device *sdev;
    const struct scmi_perf_proto_ops *ops;
    struct scmi_protocol_handle *ph;
    int count, id;
    pr_info("opi6_perf_get: begin\n");
    dev = bus_find_device(&scmi_bus_type, NULL, NULL, match_perf_device);
    if (!dev) return -ENODEV;
    sdev = to_scmi_dev(dev);
    ops = sdev->handle->devm_protocol_get(sdev, SCMI_PROTOCOL_PERF, &ph);
    if (IS_ERR(ops)) {
        int ret = PTR_ERR(ops);
        put_device(dev);
        return ret;
    }
    count = ops->num_domains_get(ph);
    pr_info("opi6_perf_get: device=%s count=%d\n", dev_name(dev), count);
    if (count > 0 && count <= 32) {
        for (id = 0; id < count; ++id) {
            const struct scmi_perf_domain_info *info = ops->info_get(ph, id);
            char lower[SCMI_MAX_STR_SIZE + 1] = {0};
            u32 maximum = 0, minimum = 0, level = 0;
            unsigned long freq = 0, power = 0;
            int index, ret;
            if (IS_ERR_OR_NULL(info)) continue;
            pr_info("opi6_perf_get: id=%d name=%s set_perf=%u\n", id, info->name, info->set_perf);
            for (index = 0; index < SCMI_MAX_STR_SIZE && info->name[index]; ++index)
                lower[index] = tolower(info->name[index]);
            if (!strstr(lower, "npu") && !strstr(lower, "mmhub") &&
                !strstr(lower, "ci700") && !strstr(lower, "ddr") && !strstr(lower, "dsu")) continue;
            // est_power_get only searches the already cached OPP list; no hardware SET.
            for (index = 0; index < 32; ++index) {
                unsigned long requested = freq;
                ret = ops->est_power_get(ph, id, &freq, &power);
                if (ret || freq < requested) break;
                pr_info("opi6_perf_get: id=%d opp=%d freq=%lu power=%lu\n", id, index, freq, power);
                if (freq == ULONG_MAX) break;
                ++freq;
            }
            ret = ops->limits_get(ph, id, &maximum, &minimum);
            pr_info("opi6_perf_get: id=%d limits_status=%d min_level=%u max_level=%u\n", id, ret, minimum, maximum);
            ret = ops->level_get(ph, id, &level, false);
            pr_info("opi6_perf_get: id=%d level_status=%d level=%u\n", id, ret, level);
            freq = 0;
            ret = ops->freq_get(ph, id, &freq, false);
            pr_info("opi6_perf_get: id=%d freq_status=%d freq=%lu\n", id, ret, freq);
        }
    }
    sdev->handle->devm_protocol_put(sdev, SCMI_PROTOCOL_PERF);
    put_device(dev);
    pr_info("opi6_perf_get: end\n");
    return 0;
}
static void __exit scmi_perf_get_probe_exit(void) { pr_info("opi6_perf_get: unloaded\n"); }
module_init(scmi_perf_get_probe_init);
module_exit(scmi_perf_get_probe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Read-only SCMI performance metadata and GET limits");
