# Methods and limits of the evidence

## Hardware and software

The recorded system is an Orange Pi 6 Plus, CIX P1 CD8180, 16 GB RAM, vendor kernel 6.6.89-cix, BIOS 1.4 dated 2026-01-30. The SMBIOS component inventory reports ATF `v2.7-6f1cdf4f3a61` and UEFI `BiosV1.4-W20260130-97268c4c`. The public CIX TF-A revision reviewed for source analysis was not proved identical to this installed ATF.

The numerical stack includes Python 3.11.17, NumPy 1.26.4, ONNX 1.17.0, and onnxruntime-zhouyi 1.20.0. Host compilation and vendor profile decoding used CixBuilder 6.1.3407.2, aipuocc 1.9.31, Linux x86-64 in WSL. Execution was on the board through its installed NOE stack. [Machine-readable environment](evidence/environment.json).

## Device execution and correctness

- NPU ORT tests disabled automatic CPU fallback and inspected the provider events. Merely listing the NPU provider was not accepted as proof.
- Synthetic operator outputs were compared against independent numerical references, with explicit tolerances and INT8 rounding/saturation.
- CPU controls used the same problematic ONNX graphs where recorded.
- The hybrid block explicitly kept some operations on CPU. It was never represented as an all-NPU block.
- Native and custom paths checked outputs outside the timed interval. Wrong graphs were excluded from valid speed conclusions.
- Bitwise agreement with another quantized implementation does not establish agreement with FP32 or generated-language quality.

## Timing

Matched projection comparisons used eight inputs, 20 warmups, and 240 measured calls per configuration. GEMV comparisons used three interleaved rounds; each runner measured distinct persistent-I/O, infer-only, and fresh-job modes. Medians of round medians are labeled as such.

Persistent-I/O timing includes input transfer, synchronous execution, and output retrieval, while compilation, initial graph loading, and weight preparation are excluded. Infer API timing still includes submission and completion handling; it is not pure matrix arithmetic.

Hardware profiling was performed separately. Cycle intervals were decoded at 1200 MHz; profiler-derived bandwidth uses the stated counter bytes and marked time window. Profiled and unprofiled timings are not mixed. Whole-graph bandwidth and a narrower engine-active window are distinct metrics.

CPU llama-bench used CPU backend, zero GPU layers, repack and flash attention enabled, five repetitions, pp64 and tg64. The original source weights were pinned and independently checked. CPU thread affinity differed between four- and eight-thread runs. Model load time is excluded. The samples are synthetic llama-bench sequences, not a server response-time or language-quality evaluation.

## Important limitations

1. **One board and specific software releases.** This is not a survey of every Orange Pi/CIX image or subsequent fix.
2. **Temporary patched driver.** The stock changing-input hang was addressed before the later NPU measurements. Results are not all stock-image out-of-box benchmarks.
3. **Existing kernel deadlock.** Reading `clk_summary` triggered a D-state wait in a clock/runtime-PM path associated with audio. Later successful NPU tests remained in that boot. Not every later waiting task was causally attributed to that read. There was no clean-boot replication, so a fully clean system baseline cannot be claimed. The saved stack is in [the final-state record](evidence/profile-final-state.json).
4. **No complete NPU LLM.** Multiplying one-layer projection times by 24 is a scenario estimate. It is not a measured token rate, a complete latency, or a hardware impossibility proof.
5. **Different quantization.** A common original model revision fixes a provenance problem, but CPU Q8_0 and NPU W8A8 still differ. The simple hybrid quantization did not preserve FP32 block accuracy sufficiently.
6. **Limited calibration data.** The first hybrid-block tests used token IDs 0-511 and did not measure perplexity, accuracy on natural-language tasks, or answer quality.
7. **Memory counters have boundaries.** NPU-reported traffic is not independently measured DDR-controller traffic. A 32-byte BIF conversion in the additional active-window estimate remains an assumption. The standalone reader's counter undercount is documented in the data.
8. **Discovery is not routing.** Two interface widths do not prove independent end-to-end bandwidth. Zero reads of protected configuration do not prove a regulator is disabled.
9. **No energy comparison.** Power, performance per watt, GPU acceleration, long contexts, and larger models were not benchmarked by this investigation.
10. **No causal overclaim.** Small changes under run-to-run variation are not advertised as optimizations. Failed interventions exclude only their tested conditions.

## Public export policy

The private workspace contains host access configuration, unrelated service state, vendor archives, and pretrained model data. The public bundle selects the relevant text records and source. Personal paths, private network addresses, and host UUIDs are redacted; irrelevant top-level fields are omitted as listed in `evidence/PROVENANCE.json`.

Numerical values in selected measurement records are retained. JSON formatting may change. Each exported file has the original-source SHA-256 and public-copy SHA-256, with the transformation described. The public manifest is an integrity and traceability aid; it is not independent attestation of the measurements or a promise that omitted private artifacts can be reconstructed from a hash.

Vendor PDFs, SDK libraries, pretrained model weights, compiled kernels, and remote-access helpers are not redistributed. The public code is sufficient for inspecting the implemented tests and reproducing the included arithmetic; exact hardware reruns have additional prerequisites described in [REPRODUCING.md](REPRODUCING.md).
