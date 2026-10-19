# Orange Pi 6 Plus: 30 TOPS, but where is the usable LLM acceleration?

**An owner's investigation of the gap between AI marketing and the NPU development experience.**

I wanted one practical result: run a local LLM faster on the NPU than on the board's CPU. After driver debugging, operator tests, a direct C++ runtime, hardware profiling, a custom GEMV kernel, and memory-system investigation, **this project did not produce a complete NPU LLM with a demonstrated speed advantage over CPU llama.cpp**.

**Buyer warning:** simply porting a conventional LLM through our tested route is expected to be slower than optimized CPU execution. The measured one-token projection sum, scaled to 24 Qwen layers, is about **17.75 ms before the remaining work**, versus about **14.35 ms per token for the complete CPU Q8_0 model**. This is an implementation-specific estimate, not a timed complete NPU LLM. A smaller model or lower answer quality is not an equivalent demonstration of acceleration.

The problems are substantial: a repeat-run driver hang, incorrect results for several tested graph patterns, a compiler crash, graph-specific workarounds, and limited access to the diagnostics needed to explain the remaining performance bottleneck. For someone buying this board specifically for accessible NPU LLM acceleration, I consider the experience unacceptable.

The evidence also contains a result that must not be hidden: **the NPU was faster than CPU ONNX Runtime on identical quantized projections when called through the correct direct API**. This is a documented criticism of the product's software and developer experience, not a claim that the NPU performs no useful computation.

**Read [the full article](ARTICLE.md), [the detailed results](RESULTS.md), [methods and limitations](METHODS.md), and [corrections](CORRECTIONS.md).**

## Results at a glance

| Question | What this project established |
|---|---|
| Does the NPU execute real operations? | Yes. After a temporary driver fix, the initial stability suite completed 5,300 executions within its numerical tolerances. |
| Is the ordinary FP16/FP32 MatMul path ready for use? | The tested ORT/CIX route rejected these MatMul cases. That is a software-route result, not proof of a missing hardware capability. |
| Were correctness failures observed? | Yes: tested masked Softmax and RoPE graphs returned incorrect results; a SiLU graph crashed compilation; a split/Concat graph used incorrect quantization scales. |
| Does direct NPU execution beat CPU ORT? | Yes: about **1.76x** at one token and **2.94x** at 64 tokens for the same projection workload. These are not full-model results. |
| How fast was the complete CPU model? | Matched-source Qwen2.5-0.5B-Instruct Q8_0: **69.679 +/- 0.414 tokens/s** for tg64 with four threads. |
| Did a custom GEMV beat the stock NPU implementation? | No: **518.599 us** versus **343.969 us** for gate/up. |
| What read throughput was observed? | About **21-23 GB/s** in the relevant NPU profiles. A separate overlap-window estimate approaches 24 GB/s under an explicitly stated counter-unit assumption. |
| Is 24 GB/s a proven physical ceiling? | **No.** Two 256-bit NPU interfaces were discovered; the full routing and bottleneck remain unresolved. |
| Was a complete NPU LLM benchmarked? | **No.** No NPU text-generation rate or matched text-quality result is claimed. |

GB/s means decimal gigabytes per second throughout this project.

## What was advertised?

As accessed on **19 October 2026**, Orange Pi's official product page promotes large-model applications, lists **45 TOPS combined CPU/GPU/NPU**, and lists **28.8 TOPS for the NPU**. CIX describes the P1 NPU as **30 TOPS INT8**. These are distinct claims, not a claim of 45 TOPS from the NPU alone. [Primary sources and scope](SOURCES.md).

The criticism is that peak arithmetic figures and broad large-model positioning do not tell buyers whether there is a working, correct, maintained NPU LLM software path. **This project does not establish that the peak TOPS specification is fabricated or that the manufacturer deliberately deceived anyone.**

## What is included?

- `ARTICLE.md`: the complete English article.
- `RESULTS.md`: measurements, correctness failures, and negative results by experiment.
- `evidence/`: selected machine-readable records, per-run decoded profiles, logs, environment details, and provenance.
- `experiments/`: operator generators, direct runners, custom kernels, graph rewrites, and archived diagnostic source.
- `analysis/`: offline calculations, integrity checks, and the static article-page builder.
- `docs/index.html`: a ready-to-host GitHub Pages article with no external scripts or fonts.
- `PUBLISHING.md`: instructions for publishing under the GitHub account `Gaptraf`.

Model weights, vendor SDK binaries, private SSH configuration, keys, personal host paths, and unrelated system inventories are excluded. Public evidence is a curated export; it is not represented as an untouched copy of the private workspace. [Evidence provenance](evidence/PROVENANCE.json).

## Verify the publication without a board

Python 3.10 or later, standard library only:

```sh
python analysis/verify.py
python analysis/recompute.py
```

The first command checks packaged-file hashes, provenance, links, and publication hygiene. The second recalculates the principal comparisons and prints JSON. Neither command connects to a board or executes NPU code. [Reproduction boundaries](REPRODUCING.md).

## Scope and recommendation

One Orange Pi 6 Plus / CIX P1 CD8180 with 16 GB RAM, kernel 6.6.89-cix, BIOS 1.4; experiments on 15-19 October 2026. The tested stack includes ORT Zhouyi 1.20.0, direct NOE, CixBuilder 6.1.3407.2, and aipuocc 1.9.31. A temporary patched AIPU driver was used for the measured NPU work. Later experiments retained an existing kernel clock/runtime-PM deadlock; there was no fresh-boot replication.

**I would not recommend buying this board on its NPU TOPS figure alone for local LLM acceleration.** A useful response from the vendors would be a reproducible, correct, end-to-end NPU LLM example, a fair CPU comparison, and documentation of the memory path. [Specific questions for Orange Pi, CIX, and the NPU software maintainers](QUESTIONS_FOR_VENDORS.md).

## Contributions and attribution

Corrections and faster implementations are welcome when accompanied by versions, numerical checks, complete commands, and raw results. Please distinguish a projection benchmark from a complete model, and prefill from decode. See [CONTRIBUTING.md](CONTRIBUTING.md).

The author used Codex to assist with code, experiments, analysis, and English editing. Assistant-originated errors are disclosed in [CORRECTIONS.md](CORRECTIONS.md). This is an owner-led investigation, not an independent certification or a survey of every vendor release.

Original project code and documentation are provided under the [MIT license](LICENSE), with archived diagnostic kernel modules retaining GPL declarations. External projects and vendor components retain their own licenses; see [attribution and exceptions](THIRD_PARTY_NOTICES.md).
