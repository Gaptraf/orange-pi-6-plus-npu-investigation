# Corrections and unsuccessful hypotheses

A critical hardware review loses value if it hides errors in its own experiment. These corrections are part of the final conclusions.

| Earlier interpretation or mistake | Correction | Effect on the conclusion |
|---|---|---|
| Raw AIPU ELF passed to the installed NOE file loader | Use `noe_load_graph_helper` with the correct memory-buffer configuration. | The earlier crash was an experiment integration error. Direct NOE worked and beat CPU ORT on the matched projections. |
| An initially downloaded Q8 GGUF treated as identical-source weights | Some tensors differed, including an F32 norm. A new GGUF was converted from the same pinned Safetensors used by the NPU. | The corrected CPU tg64 baseline remained about 69.7 tokens/s. The old Q4 baseline is not central evidence in this publication. |
| Projection time multiplied by 24 presented too broadly | It is an estimate for the tested decomposition, omitting other work. | No full NPU generation rate or hardware performance limit is claimed. |
| Single-port `256 bits x 750 MHz` used as an explanation of the entire path | Hardware discovery identified two 256-bit NPU interfaces. | 24 GB/s is not proved to be the aggregate physical ceiling. 48 GB/s is not established end-to-end bandwidth either. |
| Ordinary pointer reads in the custom kernel assumed to reach constant weights | Explicit DMA from ASID 1 into SRAM was required. | Early wrong outputs were our integration error and were excluded from timing claims. |
| A split projection assumed correct because the CPU graph matched | The NPU Concat IR had an inconsistent output scale. The recorded IR workaround restored checked equality. | Only numerically correct split variants are used in the final speed comparison. |
| ORT suspected of creating a fresh job for every call | API interception showed job reuse in the tested route. | Fresh-job overhead exists, but does not explain the entire ORT penalty here. |
| Zero protected-register reads interpreted as disabled limits | Non-secure reads can be filtered; access was not established. | No regulator state or exact security bit is inferred from zero alone. |
| More NPU cores assumed to provide proportional bandwidth | Correct three-core stock split and custom GEMV did not provide that gain. | The tested interventions failed; a universal impossibility is not proved. |

Earlier numerical probabilities of future success were subjective assistant estimates, not measurements. They are not used as evidence in this publication.

The final article includes favorable NPU results, quantization failures attributable to our implementation, the retained kernel-state limitation, and the fact that no complete NPU LLM was measured. Findings should be cited together with those boundaries.
