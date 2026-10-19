# Detailed experimental results

Experiment dates: 15-19 October 2026. These are the final interpretations, including subsequent corrections. GB/s is decimal. A sum of graph medians, an instrumented device interval, and an end-to-end model measurement are kept separate.

## E01 - Basic operation and stability

Source: [stage1 evidence](evidence/stage1-evidence.json), [diagnosis log](evidence/logs/stage1-diagnosis.log), [post-fix suite](evidence/logs/stage1-suite.log).

| Post-fix test | Shape / condition | Executions | Maximum error |
|---|---|---:|---:|
| INT8 Add | `[4,32]` | 1,000 | 0 |
| INT8 MatMul, constant weights | `[4,32] x [32,16]` | 1,000 | 0 |
| INT8 MatMul, changing inputs | Same | 1,000 | 0 |
| INT8 MatMul, non-power-of-two scales | Scales 0.031 / 0.023 / 1.7 | 1,000 | 1 INT8 step |
| FP16 Add | `[4,32]` | 1,000 | 0.000244140625 |
| INT8 MatMul, one row | `[1,512] x [512,512]` | 100 | 0 |
| INT8 MatMul, 64 rows | `[64,512] x [512,512]` | 100 | 0 |
| Session lifecycle / multiple graphs | Recreated and resident sessions | 100 | At most 1 INT8 step |
| **Total** | | **5,300** | Within specified tolerances |

Before the fix, the changing-input MatMul could hang on its second execution. A separately built module using the community command-pool state fix passed the suite. This is evidence for the tested fix, not a complete causal trace of the internal race.

FP16/FP32 MatMul was rejected by the tested provider. FP32 Add failed its selected `atol=1e-4, rtol=1e-3` tolerance in five checks; one saved result exactly matched FP16-rounded computation. [Precision record](evidence/stage1-precision.json).

## E02 - Transformer operators and a hybrid first block

Source: [stage2 evidence](evidence/stage2-evidence.json). Synthetic inputs and independent CPU controls were used; these tests do not assess natural-language quality.

| Case | Original result | Checked alternative |
|---|---|---|
| FP16 Mul | 10/10 pass, max abs 0.000649 | No rewrite required |
| FP16 RMSNorm | 10/10 pass with combined abs/relative tolerance | Max abs 0.004150 |
| Causal mask + Softmax | 10/10 fail, max abs 0.9730 | Expanded mask as input: 10/10 pass, max abs 0.000241 |
| FP16 RoPE | 10/10 fail, max abs 2.1108 | Expanded input constants and Mul sign change: 10/10 pass, max abs 0.000947 |
| Sigmoid -> Mul SiLU | Compiler crash | Separate sessions: 10/10 chains, max abs 0.005773 |
| FP16 KV Concat | 10/10 pass | Exact equality |
| INT8 batched QK MatMul | 10/10 pass | Exact equality |

The hybrid block used seven NPU QLinearMatMul projections, with other math and conversion explicitly on CPU. Eight sequences of eight tokens produced 56 NPU calls. Calibration used 12 separate sequences, sampled from token IDs 0-511 with genuine embeddings, not a representative language dataset.

| Block-output comparison | Relative L2 | Cosine similarity | Max abs |
|---|---:|---:|---:|
| Hybrid NPU versus FP32 | 19.26% | 0.981523 | 0.350874 |
| Same simple INT8 scheme on CPU versus FP32 | 19.21% | 0.981598 | 0.341511 |
| Hybrid NPU versus same INT8 CPU scheme | 7.20% | 0.997412 | 0.116929 |

These errors prevented calling the block a quality-preserving port. Relative L2 is not a token error rate. The quantization scheme itself contributed heavily.

## E03 - ORT overhead and native execution

The historical route progressed from seven ORT calls, to four fused projection calls, to persistent direct native execution. Early full-layer extrapolations are not presented as measured NPU token rates. The authoritative corrected comparison is E04.

The native runner showed that repeated creation of a fresh job is expensive, but API interception showed ORT already reused its job in the checked route. Therefore the hypothesis that ORT created a job on every call was rejected. [Corrected runner](experiments/runtime/native_runner.cpp).

## E04 - Corrected matched-workload comparison

Source: [per-configuration results](evidence/review-graph-suite.json), [summary](evidence/review-summary.json), [matched-source CPU model](evidence/review-matched-cpu.json).

Eight input cases, 20 warmups, and 240 measured calls per configuration. Four graphs represent QKV, O, gate/up, and down. Profiling and dumps off; setup and weight loading excluded; persistent jobs used.

| Case | CPU ORT projection sum | NPU ORT projection sum | Direct NOE projection sum |
|---|---:|---:|---:|
| One token, fps | 1,301.342 us (1 thread) | 1,524.1805 us | 739.686 us |
| 64 tokens, fps | 7,996.4435 us (8 threads) | 3,974.12 us | 2,720.0105 us |
| One token, none | Not measured | 1,524.8855 us | 713.447 us |

The direct path matched the corresponding ORT output bitwise in all 6,264 recorded validation checks in the main series; it executed 9,372 inferences. NPU-versus-independent-integer-reference error was at most one INT8 step. These are projection checks, not generated-text checks.

The same-source full CPU Q8_0 model, llama.cpp commit `81bc6b8`, CPU-only, repack and flash attention enabled:

| Full-model test | 4 threads | 8 threads |
|---|---:|---:|
| tg64 | 69.678906 +/- 0.413961 tokens/s | 68.407 +/- 0.223 tokens/s |
| pp64 | 512.021266 +/- 1.980163 tokens/s | 826.819 +/- 3.993 tokens/s |

Five repetitions; +/- is sample standard deviation. The source revision matches the NPU weights; Q8_0 and W8A8 remain different quantization schemes. No full-model NPU comparison at equal text quality exists.

## E05 - Hardware profiling and the corrected split

Source: [five profiles per case](evidence/profile-summary.json), [paired timing](evidence/profile-paired-host.json), [IR correction](evidence/profile-concat-fix.json).

| Projection case | Tokens | Cores | Median marked device interval |
|---|---:|---|---:|
| QKV | 1 | 0,1,2 | 42.79 us |
| O | 1 | 0 | 33.64 us |
| Gate/up | 1 | 0,1 | 259.405 us |
| Down | 1 | 0 | 138.62 us |
| Split gate/up, six outputs | 1 | 0,1,2 | 266.39 us |
| Gate/up | 64 | 0,1,2 | 996.49 us |

The original Concat IR used output scale 1.0 instead of 12.94507027 / 25.59990692. Preserving the correct scale and identity requantization restored checked output equality. Speed is reported only after numerical correction.

With the same host compiler, non-profiled gate/up calls were 343.69 us stock versus 356.41 us split with corrected Concat: no acceleration. This is separate from the profiled intervals above.

## E06 - Specialized packed GEMV

Source: [three interleaved rounds](evidence/logs/gemv-comparison.log), [accuracy](evidence/gemv-full-prefetch3-accuracy.json), [extreme inputs](evidence/gemv-stress.json), [five profiles per custom case](evidence/gemv-profile-summary.json).

| Implementation | Median persistent call with I/O | Median infer-only call |
|---|---:|---:|
| Stock NPU FC | 343.969 us | 333.174 us |
| Custom sequential DMA | 837.603 us | 836.273 us |
| Custom double buffer | 664.419 us | 653.514 us |
| Custom double buffer, three cores | 518.5985 us | 500.154 us |

The custom kernels read 8,716,288 constant bytes; stock read 5,911,040 in the compared profiles. Packing preserved INT8 values but did not compress them. The three-core custom median marked interval was 410.1617 us, with 8,738,368 total read bytes, about 21.30 GB/s. Final output concatenation was excluded from its call time.

## E07 - Concurrent engines

Source: [12 executions and overlap markers](evidence/mixed-bandwidth-summary.json).

| Case | Whole marked graph interval | Whole-graph counter read rate |
|---|---:|---:|
| New standalone DMA reader | 355.706 us | 17.965 GB/s |
| Same DMA + two stock AIFF branches | 595.461 us | 20.775 GB/s |

Overlap was confirmed, and DMA slowed when AIFF was active. The whole mixed graph includes a DMA-only tail. The conditional 22.90-24.17 GB/s overlap-window bound is not an independently measured peak, and its upper bound is not a demonstrated exceedance of 24 GB/s.

## E08 - Job priority

Source: [20 alternating runs](evidence/qos-summary.json).

Low/high priority changed submitted flags from `18=MULTI_GROUP+SLOW` to `20=MULTI_GROUP+FAST`. Median device time was 595.3242 / 595.3550 us. The 0.0052% difference is far smaller than variation. This excludes a useful improvement from that setting for this single-job workload, not every fabric QoS policy.

## E09 - Clocks, mappings, and SMMU

Source: [clock/performance analysis](evidence/mmhub-smmu/analysis.json), [SMMU counter result](evidence/smmu-pmcg/analysis.json).

SCMI GET returned 750 MHz for NPU memory/MMHUB and 1500 MHz for CI700. These matched the highest exposed Performance OPP levels in the snapshot. Higher numbers in Clock metadata were not accepted as supported safe block operating points; no SET command was issued.

The observed 4 GiB NPU mapping consisted of 2,048 blocks of 2 MiB, with no 4 KiB pages. Seventy-two translations were cross-checked without mismatch. During one existing graph execution, TBU0 / TBU1 recorded 19,358 / 6,921 forwarded transactions and zero MicroTLB misses or table-walk requests. Both TBUs worked; neither these counts nor zero misses establish the bandwidth of their hit path.

## E10 - NI-700 topology and access boundary

Source: [discovery analysis](evidence/npu-bus-topology/analysis.json), [PMU access analysis](evidence/ni700-pmu-access/analysis.json).

| Discovered interface | Address | Node ID | NODE_INFO | Data width |
|---|---|---:|---|---:|
| NPU ASNI0 | 0x0d05a000 | 22 | 0x7d3 | 256 bits |
| NPU ASNI1 | 0x0d05b000 | 23 | 0x7d3 | 256 bits |

Four MMHUB AMNI outputs also report 256 bits. The discovery hierarchy is not a datapath routing map. At 750 MHz, one 256-bit data channel has a nominal 24 GB/s rate; summing two to 48 GB/s requires independence and simultaneous saturation, which were not established.

PMU node identification at 0x0d063000 returned 0x00200006. PMCFGR at 0x0d063e00 returned zero instead of the documented 0x00417f08. The access probe performed no MMIO writes and no NPU jobs. The exact security state and downstream bottleneck remain unresolved.

## Stopping point

No complete NPU LLM was implemented or timed. No 40-45 GB/s path was demonstrated. No absolute 24 GB/s physical ceiling was proved. No BIOS/TF-A flashing or security-policy change was performed. The work stopped because the remaining uncertainty did not justify escalating the owner's practical acceleration project into firmware modification.
