# Orange Pi 6 Plus: 30 TOPS, but where is the usable LLM acceleration?

**By Gaptraf. An owner's investigation, 15-19 October 2026.**

I wanted a local language model that ran faster on the Orange Pi 6 Plus NPU than on its CPU. I did not want to develop a board-support package, debug quantization metadata, or reverse-engineer an interconnect just to find out whether the accelerator was worth using.

That is where this investigation went. We got real NPU computation working, fixed or worked around several problems, built a direct runtime, profiled the hardware, and wrote a specialized matrix-vector kernel. **We did not obtain a complete NPU LLM with a demonstrated speed advantage over the CPU.**

**The practical warning for a buyer is stronger than "some engineering is required." With the tested software route, simply porting a conventional LLM to this NPU token by token is expected to be slower than well-configured CPU llama.cpp.** For Qwen2.5-0.5B, the measured projection calls alone, repeated across its 24 layers, already exceed the complete CPU model's measured time per token, before the remaining NPU/CPU work is added. This is a comparison-based estimate for our implementation, not a measured full-model NPU result or a verdict on every LLM.

For my intended use, the advertised AI capability failed to translate into an acceptable developer experience. I consider the large-model positioning misleading for a buyer expecting an accessible, working NPU LLM path. The evidence supports that criticism. It does not support pretending we measured a full NPU model, proved a physical bandwidth ceiling, or established deliberate deception.

## Earlier work and what this report contributes

This is a follow-up measurement report. Earlier community work already investigated this NPU's LLM limitations; the general problem was known. Our contribution is the recorded correctness tests, matched runtime comparisons and additional memory-path evidence. The driver fix is credited to its author. See [prior work and the overlap assessment](RELATED_WORK.md) before interpreting these results as a new discovery.

## 1. The sales pitch and the missing practical answer

The official Orange Pi product page calls it:

> The ultimate development board for on-device AI in the new era of large models

The page, as accessed on 19 October 2026, lists 45 TOPS combined across CPU, GPU, and NPU, and 28.8 TOPS for the NPU. CIX describes the P1 NPU as 30 TOPS INT8. The headline arithmetic capability is not a measured LLM decoding rate. [Orange Pi product page](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html), [CIX P1 application example](https://www.cixtech.com/list_6/151.html).

A buyer still needs a practical answer: which model runs on the NPU, with which software, at what quality, and how much faster is it than CPU execution? Our investigation did not establish a supported, reproducible answer for the tested configuration.

CIX's own ERNIE application example is instructive. It identifies CPU/KleidiAI and GPU/Vulkan for the chatbot, and the NPU for PaddleOCR. That is a legitimate heterogeneous application, but its language-model performance is not evidence of NPU language-model acceleration. [CIX example](https://www.cixtech.com/list_6/151.html).

## 2. What we actually tested

The board was an Orange Pi 6 Plus with a CIX P1 CD8180 and 16 GB RAM, running kernel 6.6.89-cix and BIOS 1.4. We tested the installed Zhouyi ONNX Runtime 1.20.0 route and direct NOE execution, then used CixBuilder 6.1.3407.2 / aipuocc 1.9.31 for host compilation and profiling.

The model architecture was Qwen2.5-0.5B-Instruct, pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. It has 24 decoder layers, hidden width 896, and intermediate width 4864. NPU work covered synthetic operator cases, a hybrid first block, and its real quantized projections. The full CPU baseline used Q8_0 weights converted from that same source revision. [Environment](evidence/environment.json), [matched CPU record](evidence/review-matched-cpu.json).

For successful NPU operator tests, automatic CPU fallback was disabled and provider profiles were checked. The hybrid block's remaining CPU operations were explicit. Correctness checks were performed against independent references; they were not assumed from a successful API return.

This was one board and a specific software stack. Later tests retained a pre-existing kernel clock/runtime-PM deadlock, without a fresh-boot replication. That limits the generality of the timing results. No complete NPU text-generation or matched text-quality result was obtained. [Methods and limitations](METHODS.md).

## 3. The first obstacle: a second execution could hang

With two changing inputs to an INT8 MatMul, the first invocation returned the correct result and the second hung waiting for NPU completion. It reproduced on ordinary random inputs and with a pause between invocations.

A publicly documented command-pool state fix was applied to a separately built AIPU module. The patched test module was loaded temporarily; the installed disk module and startup configuration were preserved. The initial post-fix suite then completed **5,300 executions without hangs or tolerance failures**. The fix came from the community, not from a newly released vendor image in this experiment. [Community fix](https://github.com/n4hy/NPU_OrangePi6Plus), [initial evidence](evidence/stage1-evidence.json).

That establishes both sides of the story: the tested setup had a reproducible stability problem, and the NPU became usable for these operations after a targeted fix. It does not establish indefinite stability or that every later driver release has the same defect.

## 4. A graph can run and still return the wrong answer

The operator tests found problems beyond missing support:

| Tested graph or operation | Observed failure | Successful workaround or boundary |
|---|---|---|
| FP16 and FP32 MatMul | Rejected by the tested ORT/CIX route | INT8 QLinearMatMul worked; hardware impossibility was not established. |
| FP32 Add | All five checked runs failed the selected FP32 tolerance | One saved sample matched an FP16-rounded computation exactly. |
| Causal mask + Softmax | 10/10 runs produced incorrect results; max absolute error about 0.973 | Expanding the mask and passing it as an input passed 10/10 checks. |
| RoPE | 10/10 original tests failed; one saved output was all zero | Expanded input cos/sin tensors and multiplication for sign changes passed. |
| SiLU expressed as Sigmoid followed by Mul | Compiler process crashed with a glibc allocation diagnostic | Two separate NPU graphs passed; the split adds a host intermediate transfer. |
| Split projection followed by Concat | Up to 90 INT8 output steps of error | Correcting the quantization metadata in IR restored checked output equality. |

The original Softmax, RoPE, and SiLU ONNX graphs passed their CPU controls. The input-expansion workarounds changed more than one condition, so these observations do not identify the exact internal compiler defect. They do establish that a buyer cannot treat graph acceptance as a correctness guarantee. [Operator records](evidence/stage2-evidence.json), [Concat fix record](evidence/profile-concat-fix.json).

The hybrid first block also exposed our own quantization limitations. Its output had about **19.26% relative L2 error** against FP32; a CPU implementation using the same simple quantization had about 19.21%. These are tensor-error norms, not percentages of incorrect tokens. A large part of this loss was in our quantization scheme. It would be wrong to blame all of it on NPU arithmetic or to present this block as a quality-preserving LLM port.

## 5. The NPU is not universally slower than the CPU

An early experiment used the wrong loading function for raw AIPU binaries in `libnoe`. That was our error. After switching to the memory-based helper with the correct configuration, the direct NOE path worked and materially changed the comparison.

Using identical quantized ONNX projections, identical inputs, and the same shapes:

| Projection workload | CPU ONNX Runtime | NPU via ORT | NPU via direct NOE |
|---|---:|---:|---:|
| One token | 1.301 ms | 1.524 ms | **0.740 ms** |
| 64 tokens | 7.996 ms | 3.974 ms | **2.720 ms** |

These are **sums of separately measured medians for four projection graphs**, not a complete Transformer block. Session creation, compilation, and weight loading were excluded. The direct runtime reused its job. [Comparison data](evidence/review-summary.json), [per-configuration records](evidence/review-graph-suite.json).

The direct NPU route was about **1.76 times faster** than CPU ORT at one token and **2.94 times faster** at 64 tokens. That result must remain visible in a critical review. It also makes the software problem concrete: changing the calling path changed the conclusion for the same computation.

The practical CPU reference was a different workload: **the complete model in CPU-only llama.cpp**. With matched-source Q8_0 weights, tg64 achieved **69.679 +/- 0.414 tokens/s** with four threads, over five repetitions. Eight-thread pp64 achieved 826.819 +/- 3.993 tokens/s. The +/- values are sample standard deviations, not confidence intervals.

Mechanically repeating the measured 0.740 ms projection sum across 24 layers gives about 17.75 ms, before attention, normalization, nonlinearities, KV-cache work, or vocabulary output. The complete CPU model took about 14.35 ms per generated token. This was a discouraging estimate for scaling our particular NPU implementation. **It was not a measured NPU decoding rate or a proof that every possible NPU implementation must lose.** Q8_0 and NPU W8A8 also differ in quantization and quality.

For someone who buys the board, selects a conventional LLM, and expects the NPU to make it faster, this is the central problem. The tested route first required compatibility fixes and numerical workarounds; straightforward scaling of its measured projection performance still pointed to a slower result than optimized CPU execution. The fact that a direct NPU call beats CPU ORT on a subset of operations does not answer that buyer's application-level question.

The comparison must preserve the task and acceptable answer quality. Replacing the model with a smaller one or using more aggressive lossy quantization is not, by itself, evidence of NPU acceleration: the CPU can run that lighter workload too. Likewise, changing the model's name or architecture does not establish a way around the observed read-throughput plateau. We found no such demonstrated route.

## 6. Profiling: more cores did not solve single-token execution

The vendor profiler decoded the traces and confirmed real multi-core activity. For the one-token gate/up workload, the stock graph used cores 0 and 1; a corrected split used all three.

| Hardware-profile case | Cores observed | Marked interval | NPU-reported read bytes |
|---|---|---:|---:|
| Stock gate/up, one token | 0, 1 | 259.405 us | 5,932,032 |
| Split gate/up, one token | 0, 1, 2 | 266.39 us | about 5.935 MB |
| Stock gate/up, 64 tokens | 0, 1, 2 | 996.49 us | about 17.930 MB |

The stock one-token report showed about **0.228% average MAC utilization** relative to its reported three-core peak, and about **6.25% MTP utilization when working**. These are different profiler metrics, not interchangeable measures of total NPU activity. At 64 tokens, the latter rose to about 57.17%. [Profile summary and individual runs](evidence/profile-summary.json).

A one-token projection reuses a weight far less than a many-token matrix multiplication. At roughly two operations per externally read INT8 weight byte, sustaining a hypothetical 30 TOPS stream would require 15 TB/s without reuse or compression. This is an arithmetic illustration of why TOPS alone cannot predict decoding speed, not a claimed memory requirement for every implementation.

The non-profiled, corrected three-core split did not accelerate gate/up: about 356.41 us versus 343.69 us with the same host compiler. More arithmetic cores did not remove the observed bottleneck.

## 7. We wrote a specialized GEMV. It still lost

The custom implementation used X2 QDPA vector instructions, an INT8 packing layout, constant-resident weights, DMA from the correct address space into SRAM, double buffering, and a three-core output split. CPU calls supplied the input vector; they did not upload the weights again on every invocation.

| Gate/up implementation | Median call including input/output |
|---|---:|
| Stock NPU FullyConnected | **343.969 us** |
| Custom GEMV, sequential DMA | 837.603 us |
| Custom GEMV, double buffering | 664.419 us |
| Custom GEMV, double buffering, three cores | **518.599 us** |

The custom progression improved substantially, but its best version remained about **1.51 times slower** than stock NPU execution. The final concatenation of its six output buffers was excluded, so this was already an optimistic comparison. Three interleaved rounds used the same eight input cases and checked outputs. [Raw timing records](evidence/logs/gemv-comparison.log), [extreme-input checks](evidence/gemv-stress.json), [kernel source](experiments/gemv/gemv_prefetch.cl).

The custom profile read all 8,716,288 bytes of the two INT8 weight matrices once. The stock graph reported only 5,911,040 constant bytes, consistent with its hardware-compressed weights. The custom layout transferred about 47.46% more constant data; its measured call was about 50.77% slower. This strongly implicates weight delivery, but differences between the compute kernels prevent attributing every microsecond to memory.

The initial custom kernel also used the wrong address space for weights. Explicit DMA from ASID 1 fixed it. That was another integration error on our side, disclosed rather than counted as a vendor failure.

## 8. The bandwidth mystery: observations are not a physical limit

Relevant NPU profiles repeatedly showed about **21-23 GB/s** of reported read traffic. Reanalyzing a narrower AIFF-active window gives about 23.86 GB/s under a 32-byte-per-BIF-count assumption. That unit was not independently verified against a P1 register specification, and these are not independent DDR-controller measurements. [Bandwidth analysis](evidence/bandwidth-review/analysis.json).

We tested different explanations instead of immediately declaring the hardware incapable:

| Hypothesis or intervention | Result |
|---|---|
| Run TPC DMA and AIFF concurrently | The streams overlapped and competed; their bandwidth did not add up to 40-45 GB/s. |
| Raise the NOE job priority | Submitted SLOW/FAST flags changed as intended; median device time was 595.324 versus 595.355 us, with no useful improvement. |
| Eliminate small pages and TLB misses | The observed mappings already used 2 MiB blocks. Both NPU TBUs had zero MicroTLB misses in the captured interval. |
| Use another arithmetic schedule and three cores | Our correct custom GEMV still operated near 21.3 GB/s and lost to stock. |
| Look for a clear software bandwidth cap | No confirmed limiter explaining the plateau was found. |

Sources: [mixed-engine runs](evidence/mixed-bandwidth-summary.json), [priority test](evidence/qos-summary.json), [SMMU counter analysis](evidence/smmu-pmcg/analysis.json).

Then discovery registers identified **two 256-bit NPU interfaces**, not one. At the observed 750 MHz interface-clock snapshot, one has a nominal 24 GB/s data-channel rate; two independently saturated interfaces would sum to 48 GB/s. That sum is not a demonstrated end-to-end NPU-to-DDR bandwidth. Their routing, shared downstream resources, and request distribution are not established. [Hardware discovery](evidence/npu-bus-topology/analysis.json).

The available Linux MMIO path could read PMU identification, but its fixed configuration register returned zero instead of the documented nonzero value. The probe stopped without configuring counters. Security filtering is consistent with this result, but the exact security setting was not read. We could not obtain the port-level occupancy and backpressure evidence needed to resolve the bottleneck. [PMU access result](evidence/ni700-pmu-access/analysis.json).

**24 GB/s is not a proved physical ceiling. 40-45 GB/s is not a demonstrated software opportunity either.** No BIOS or TF-A flashing was performed. The investigation stopped before firmware modification because that escalation was not justified by the practical objective.

## 9. What the vendors should explain

For a board sold around large-model AI, the missing deliverables are practical:

- A correct, maintained NPU LLM example with pinned model and SDK versions.
- End-to-end prefill and decode results, compared against a well-configured CPU baseline at comparable quality.
- A versioned operator/type/shape support matrix and known compiler-correctness issues.
- A documented memory path: both NPU interfaces, shared bottlenecks, clocks, outstanding-request limits, and applicable QoS controls.
- Supported Linux diagnostics that let developers distinguish insufficient requests from downstream backpressure.
- A clear account of which parts of an advertised AI application actually use the NPU.

Orange Pi, CIX, and the Zhouyi software ecosystem have different responsibilities. This project cannot assign every observed failure to one company. It can show that the combination delivered to this owner required extensive engineering and still did not meet the intended goal. [Specific technical questions](QUESTIONS_FOR_VENDORS.md).

## 10. My buying conclusion

For my goal, **I would not buy the Orange Pi 6 Plus based on its NPU TOPS figure or broad large-model claims**. The demonstrated compute capability did not compensate for unstable behavior, graph-correctness surprises, integration effort, and an unresolved performance bottleneck.

An ordinary buyer should not have to assume that an unspecified future runtime, a different model, or lower answer quality will rescue the advertised acceleration. With the stack we tested, the evidence points to a practical risk of spending substantial effort on an NPU port that runs slower than the CPU implementation already available. A supported end-to-end demonstration is the vendor's missing answer to that concern.

A theoretical peak is not a usable LLM product. Buyers need working software, trustworthy numerical results, transparent limitations, and reproducible application benchmarks. Those are the standards against which I am judging this experience.

This conclusion can change with evidence. A correct, reproducible full NPU LLM beating the matched CPU baseline would be a meaningful result. Until then, I regard this board's NPU LLM proposition as unproven and unsuitable for the straightforward acceleration I wanted.

## Evidence, reproducibility, and disclosures

The repository includes selected per-run records, decoded profiles, benchmark source, independent arithmetic checks, original-source hashes, public-file hashes, and a full [correction log](CORRECTIONS.md). [Offline verification](REPRODUCING.md) requires no board. Exact rerunning of the hardware experiments requires vendor software and model artifacts that are not bundled; hardware collection is not advertised as a turnkey process.

This is an owner-led, Codex-assisted investigation. It covers one board and the recorded releases. It contains no complete NPU model benchmark, power-efficiency comparison, or verdict on every use of the CPU/GPU/NPU. See [methods](METHODS.md), [all result stages](RESULTS.md), [source references](SOURCES.md), and [public evidence provenance](evidence/PROVENANCE.json).
