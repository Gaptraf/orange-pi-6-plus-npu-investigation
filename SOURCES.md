# Primary sources and evidence map

Public-source access date: **19 October 2026**. Vendor pages can change. `evidence/external-sources.json` records URLs and hashes of the locally inspected HTML; full vendor pages and PDFs are not redistributed. A hash identifies what was retrieved, but does not by itself let a reader reconstruct the page.

## Product claims and application example

- [Orange Pi 6 Plus official product page](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html): large-model positioning, 45 TOPS combined CPU/GPU/NPU, and 28.8 TOPS NPU in the specification table. The article's buyer-expectation judgment is the owner's interpretation of that positioning.
- [CIX P1 / ERNIE application announcement](https://www.cixtech.com/list_6/151.html), dated 19 August 2025: P1 NPU 30 TOPS INT8; the application table assigns CPU/KleidiAI and GPU/Vulkan to chatbot acceleration and the NPU to PaddleOCR. Its model benchmarks are not used as measurements of our Qwen experiment.
- [Orange Pi official downloads](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-6-Plus.html): vendor images, manuals, and firmware resources. The investigation performed no firmware flashing.

## NPU software and the tested stack

- [Community Orange Pi 6 Plus NPU driver fix](https://github.com/n4hy/NPU_OrangePi6Plus): source of the command-pool state fix tested in the initial stability work.
- [Arm China Compass NPU Driver](https://github.com/Arm-China/Compass_NPU_Driver/tree/2868d533694740de6891f9998812ceb62a899dee): public declarations and profiling sample. Public upstream and installed vendor libraries are not assumed identical.
- [Arm China ORT environment settings](https://github.com/Arm-China/Compass_Onnxruntime/blob/132d4d83c9c7b749803a81a3cd006b32a57d22dd/zhouyi/docs/Environment_Variable_Description.md): profiling and tiling settings; descriptions were checked against emitted artifacts in the experiment.
- [Compass Apache TVM, pinned source](https://github.com/Arm-China/Compass_Apache_TVM/tree/559cee30156fb6fe9f6d476d9b6996c671884c69): X2 kernel integration, address spaces, and DMA source.
- [Custom-operator integration guide](https://arm-china.github.io/Compass_Apache_TVM/how_to_guides/how_to_write_the_operator_plugin.html), [DMA guide](https://arm-china.github.io/Compass_Apache_TVM/how_to_guides/how_to_use_dma.html): APIs used to develop the custom kernel.
- [CIX Developer Center](https://developer.cixtech.com/): CIX P1 NOE SDK and AI ModelHub Development Guide V1.0, and CIX P1 NPU Development Guide V1.0. The first guide is dated 30 September 2025; it is not represented as documentation of every later release.
- [Radxa SDK setup instructions](https://docs.radxa.com/en/orion/o6n/app-development/artificial-intelligence/env-setup) and [CIX 25Q3 SDK archive](https://dl.radxa.com/orion/common/cix_noe_sdk_25_q3_release.tar.gz): the host compiler/profiler used in this work. Archive SHA-256: `ab35aa0141f9354debea286d2e31ad9a46ffe8d9870f5a0128c4c93b8ef013a1`. Instructions for a related P1 platform are not proof that its firmware is compatible with Orange Pi.

## CPU baseline and model

- [Qwen2.5-0.5B-Instruct pinned model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775): architecture and original weights.
- [llama.cpp benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md): interpretation of pp/tg tests. The measured local binary reports commit `81bc6b8`; the raw command and sample results are retained in `evidence/review-matched-cpu.json`.

## Interconnect, firmware source, and counters

- [CIX Linux, pinned revision](https://github.com/cixtech/cix_opensource__linux/tree/9c2c8c53ac83654ec452345ad0e539a265f02cf2): NPU SMMU/TBU descriptions and related platform configuration. A source DTS is not the running ACPI configuration.
- [CIX TF-A, pinned revision](https://github.com/cixtech/cix_opensource__arm-trusted-firmware/tree/114fb20577bcc4038025de4e12bca60e04dd5212): platform QoS mappings and interface names. Its exact match to the installed ATF was not established.
- [Arm NI-700 r2p1 TRM](https://documentation-service.arm.com/static/64b99ffadf6cd61d528c840c): ASNI/AMNI discovery fields, PMCFGR fixed value, access conditions, and event definitions. Product-version identification also covers r2p0 EAC; an exact minor hardware revision is not inferred.
- [Arm NI-700 r2p3 TRM](https://documentation-service.arm.com/static/64c2dc0438511951cb7a1956): cross-check of interface-field interpretation. This is not a claim our block is r2p3.
- [Arm MMU-700 r1p2 TRM](https://documentation-service.arm.com/static/64ac1a4338511951cb79dc1a): SMMU event definitions and access conditions.
- [Linux v6.12 NI PMU driver](https://github.com/torvalds/linux/blob/v6.12/drivers/perf/arm-ni.c#L493): treats zero PMCFGR as lack of PMU access. This behavior supports the access interpretation, not a bandwidth-ceiling conclusion.

## Claim-to-evidence map

| Article claim | Included evidence |
|---|---|
| Changing-input hang and post-fix stability | `evidence/stage1-evidence.json`, `evidence/logs/stage1-diagnosis.log`, `evidence/logs/stage1-suite.log` |
| Operator failures, controls, and workarounds | `evidence/stage2-evidence.json`, `experiments/operators/` |
| Correct direct API and matched projections | `evidence/review-summary.json`, `evidence/review-graph-suite.json`, `experiments/runtime/native_runner.cpp` |
| Matched-source CPU speed | `evidence/review-matched-cpu.json`, `evidence/review-weights.json` |
| Core use, device intervals, and utilization | `evidence/profile-summary.json`, `evidence/profile-decoded/` |
| Concat quantization correction | `evidence/profile-concat-fix.json`, `experiments/graph-rewrite/profile_fix_concat.py` |
| Custom GEMV speed and accuracy | `evidence/logs/gemv-comparison.log`, `evidence/gemv-profile-summary.json`, `evidence/gemv-stress.json` |
| Shared-resource contention | `evidence/mixed-bandwidth-summary.json`, `evidence/mixed-bandwidth-decoded/` |
| Priority did not help | `evidence/qos-summary.json`, `evidence/qos-decoded/` |
| Existing 2 MiB mappings and zero observed TLB misses | `evidence/smmu-pmcg/analysis.json` |
| Two 256-bit interfaces, unknown routing | `evidence/npu-bus-topology/analysis.json` |
| Current PMU path unavailable | `evidence/ni700-pmu-access/analysis.json` |

The SHA-256 values of the original Russian reports are retained in `evidence/PROVENANCE.json`. The English narrative incorporates their final corrections and does not repeat obsolete conclusions as current facts.
