"""Recalculate published comparisons. Standard library only; no device access."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
def load(name):return json.loads((ROOT/'evidence'/name).read_text(encoding='utf8'))
def calculate():
    review=load('review-summary.json')['rows']
    def row(provider,tokens):return next(r for r in review if r['provider']==provider and r['tokens']==tokens and r['tiling']=='fps')
    projections={str(m):dict(cpu_ort_us=row('cpu',m)['sum_ort_medians_us'],
                            npu_ort_us=row('npu',m)['sum_ort_medians_us'],
                            npu_noe_us=row('npu',m)['sum_native_medians_us'],
                            noe_speedup_over_cpu_ort=row('cpu',m)['sum_ort_medians_us']/row('npu',m)['sum_native_medians_us']) for m in [1,64]}
    cpu=load('review-matched-cpu.json')
    tg=next(r for b in cpu['benchmarks'] for r in b['results'] if b['threads']==4 and r['n_prompt']==0 and r['n_gen']==64)
    lines=[json.loads(s) for s in (ROOT/'evidence/logs/gemv-comparison.log').read_text().splitlines() if s.strip()]
    runs=[r for r in lines if 'round' in r]
    gemv={name:statistics.median(r['modes']['persistent_io']['p50_us'] for r in runs if r['name']==name) for name in ['baseline','dma','prefetch','prefetch3']}
    stock=load('profile-summary.json')['gate-up-m1']['runs']
    custom=load('gemv-profile-summary.json')['full-prefetch3']['runs']
    stock_rate=statistics.median(r['read_bytes']/r['counter_span_us']/1000 for r in stock)
    custom_rate=statistics.median(r['read_bytes']/r['span_us']/1000 for r in custom)
    qos=load('qos-summary.json')['groups']
    low=qos['0']['npu_us']['median'];high=qos['1']['npu_us']['median']
    ports=load('npu-bus-topology/analysis.json')
    assert len(ports['npu_ports'])==2
    smmu=load('smmu-pmcg/analysis.json')
    out=dict(
        scope='No complete NPU LLM benchmark; projections, CPU model and profiles are separate workloads.',
        projection_sums=projections,
        full_cpu_Q8_tg64=dict(tokens_per_second=tg['avg_ts'],sample_stdev=tg['stddev_ts'],
                             milliseconds_per_token=1000/tg['avg_ts'],samples=len(tg['samples_ts'])),
        gemv_persistent_IO_median_us=gemv,
        custom_vs_stock_latency_ratio=gemv['prefetch3']/gemv['baseline'],
        constant_bytes_ratio=8716288/5911040,
        profile_read_GBps=dict(stock_gate_up=stock_rate,custom_three_core=custom_rate),
        qos=dict(low_priority_us=low,high_priority_us=high,high_priority_time_reduction_percent=100*(low-high)/low),
        nominal_interfaces=dict(width_bits_each=256,count=2,assumed_clock_hz=750000000,
                                per_port_GBps=256/8*750000000/1e9,sum_GBps=2*256/8*750000000/1e9,
                                sustainable_aggregate_proven=False),
        smmu=dict(block_2MiB_count=smmu['block_2MiB_count'],page_4KiB_count=smmu['page_4KiB_count'],
                  tbu0_misses=smmu['units']['tbu0']['events']['0x2'],tbu1_misses=smmu['units']['tbu1']['events']['0x2']),
    )
    assert abs(projections['1']['noe_speedup_over_cpu_ort']-1.7593)<0.001
    assert abs(projections['64']['noe_speedup_over_cpu_ort']-2.9399)<0.001
    assert abs(gemv['prefetch3']-518.5985)<1e-6
    assert abs(stock_rate-22.86784)<0.001 and 21.2<custom_rate<21.4
    assert not ports['aggregate_physical_24_GBps_limit_proven']
    return out
if __name__=='__main__':print(json.dumps(calculate(),indent=2))
