# Questions for Orange Pi, CIX, and the NPU software maintainers

These are requests for concrete technical information. They are not statements that any one company caused every observed problem. No message has been sent to a vendor as part of preparing this repository.

1. **Supported NPU LLM path.** Which exact LLM, model revision, quantization, SDK, driver, and board image constitute your maintained NPU inference example for Orange Pi 6 Plus? Please provide code and complete commands.
2. **Fair application benchmark.** What are end-to-end prefill and decode rates, first-token latency, context length, batch size, and memory usage? How do they compare with CPU llama.cpp on the same board, from the same source weights and at comparable output quality?
3. **Arithmetic specifications.** Under what datatype, operation-count convention, clock, workload, and sparsity/reuse conditions are the 28.8/30 NPU TOPS and 45 combined TOPS specified? Which software schedules a single application across CPU/GPU/NPU, and with what measured benefit?
4. **Versioned correctness status.** Which released versions address the repeat-job hang and the tested mask/Softmax, RoPE, SiLU compilation, and Concat quantization issues? Can you reproduce the supplied graph patterns, or identify an unsupported construct explicitly?
5. **Operator/type/shape coverage.** Please provide the supported combinations, actual internal precision, quantization modes, broadcasting rules, dynamic-shape limits, and fallback behavior for the shipping stack.
6. **NPU memory architecture.** How are NPU ASNI IDs 22 and 23 connected to the four observed MMHUB AMNI outputs and DDR? Which portions are shared? What sustainable aggregate read bandwidth is expected for weights from DDR?
7. **Request distribution.** How do AIFF and TPC route traffic across those interfaces? Which SDK or compiler settings control distribution, burst generation, and outstanding-request depth? Can a single workload use both interfaces concurrently at full data rate?
8. **Fabric regulation.** Which reset/firmware values are used for read/write/combined regulators, TSPEC, outstanding limits, and QoS? Is any supported Linux control available for performance tuning?
9. **Supported diagnostics.** How can a developer obtain per-interface traffic, occupancy, starvation, pending-request, and backpressure statistics on the shipping image without changing firmware or security policy?
10. **Maintenance and distribution.** What is the supported release channel for compatible kernel headers, ARM64 build tools, NOE headers/libraries, compiler, profiler, and known-issue fixes? Which parts are open source, and which require vendor intervention?

For the register-access question, please distinguish NODE_INFO discovery from PMCFGR/configuration permissions and distinguish register access from debug-event enable signals. We read the former successfully; that did not establish access to the latter.

A reproducible answer or corrected benchmark can be submitted as an issue or pull request with raw evidence. We will update the conclusions when the evidence changes.
