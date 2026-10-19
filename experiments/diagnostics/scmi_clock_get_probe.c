// Read existing SCMI clock metadata and GET current rates. No CCF or SET calls.
#include <linux/module.h>
#include <linux/device.h>
#include <linux/scmi_protocol.h>
#include <linux/err.h>
#include <linux/string.h>

extern struct bus_type scmi_bus_type;

static int match_clock_device(struct device *dev, const void *unused)
{
    struct scmi_device *sdev = to_scmi_dev(dev);
    return sdev->protocol_id == SCMI_PROTOCOL_CLOCK && sdev->handle;
}

static int __init scmi_clock_get_probe_init(void)
{
    struct device *dev;
    struct scmi_device *sdev;
    const struct scmi_clk_proto_ops *ops;
    struct scmi_protocol_handle *ph;
    int count, id, selected = 0;
    pr_info("opi6_scmi_get: begin\n");
    dev = bus_find_device(&scmi_bus_type, NULL, NULL, match_clock_device);
    if (!dev) return -ENODEV;
    sdev = to_scmi_dev(dev);
    pr_info("opi6_scmi_get: device=%s name=%s protocol=%u\n", dev_name(dev), sdev->name, sdev->protocol_id);
    ops = sdev->handle->devm_protocol_get(sdev, SCMI_PROTOCOL_CLOCK, &ph);
    if (IS_ERR(ops)) {
        int ret = PTR_ERR(ops);
        put_device(dev);
        return ret;
    }
    count = ops->count_get(ph);
    pr_info("opi6_scmi_get: count=%d\n", count);
    if (count > 0 && count <= 1024) {
        for (id = 0; id < count; ++id) {
            const struct scmi_clock_info *info = ops->info_get(ph, id);
            u64 rate = 0;
            int ret, index;
            if (!info) continue;
            if (!strnstr(info->name, "npu", SCMI_MAX_STR_SIZE) &&
                !strnstr(info->name, "mmhub", SCMI_MAX_STR_SIZE) &&
                !strnstr(info->name, "ci700", SCMI_MAX_STR_SIZE)) continue;
            if (++selected > 16) break;
            pr_info("opi6_scmi_get: query id=%d name=%s\n", id, info->name);
            ret = ops->rate_get(ph, id, &rate);
            pr_info("opi6_scmi_get: id=%d name=%s status=%d rate=%llu discrete=%u\n",
                    id, info->name, ret, (unsigned long long)rate, info->rate_discrete);
            if (info->rate_discrete) {
                pr_info("opi6_scmi_get: id=%d num_rates=%d\n", id, info->list.num_rates);
                for (index = 0; index < info->list.num_rates && index < SCMI_MAX_NUM_RATES; ++index)
                    pr_info("opi6_scmi_get: id=%d index=%d allowed_rate=%llu\n", id, index,
                            (unsigned long long)info->list.rates[index]);
            } else {
                pr_info("opi6_scmi_get: id=%d min=%llu max=%llu step=%llu\n", id,
                        (unsigned long long)info->range.min_rate,
                        (unsigned long long)info->range.max_rate,
                        (unsigned long long)info->range.step_size);
            }
            if (ret) break;
        }
    }
    sdev->handle->devm_protocol_put(sdev, SCMI_PROTOCOL_CLOCK);
    put_device(dev);
    pr_info("opi6_scmi_get: end selected=%d\n", selected);
    return 0;
}
static void __exit scmi_clock_get_probe_exit(void) { pr_info("opi6_scmi_get: unloaded\n"); }
module_init(scmi_clock_get_probe_init);
module_exit(scmi_clock_get_probe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Read-only GET of existing SCMI clock metadata and rates");
