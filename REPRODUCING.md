# Reproducing and auditing the investigation

There are two different tasks: recalculate the published evidence offline, or rerun hardware collection. Only the first is a self-contained workflow in this repository. No board access is needed for it.

## 1. Audit the packaged evidence

Python 3.10 or later; no pip dependencies:

```sh
python analysis/verify.py
python analysis/recompute.py
```

`verify.py` checks the release manifest and the per-export provenance hashes, parses Python and JSON, validates local documentation links, and scans for excluded personal/access metadata. It does not load vendor libraries, contact SSH, read MMIO, or submit an NPU job.

`recompute.py` derives the matched projection speedups, CPU generation baseline, GEMV medians, profiler read rates, priority delta, and nominal interface arithmetic from the included records. It prints JSON to stdout. To retain it:

```sh
python analysis/recompute.py > recomputed.json
```

To regenerate the static article page after editing `ARTICLE.md`:

```sh
python analysis/build_site.py
```

That changes the generated page. Packaged hashes intentionally stop matching after edits; use your own commit history for a modified publication. `manifest.sha256.json` describes this prepared bundle, not future revisions.

## 2. Where the original hardware tools ran

| Location | Tools and purpose |
|---|---|
| Orange Pi 6 Plus, ARM64 Linux | Installed ORT/NOE and vendor libraries; isolated Python 3.11 environment; NumPy/ONNX; GCC/G++; numerical tests; direct C++ runners; device execution. |
| Linux x86-64 host / WSL | CIX 25Q3 SDK, CixBuilder 6.1.3407.2 and aipuocc 1.9.31; custom-kernel compilation and vendor profile decoding. |
| Any PC with Python 3.10+ | Offline arithmetic, evidence inspection, publication verification, and article generation. |

The original SSH jump arrangement is unrelated to the measurements and is deliberately not included. No remote-access setup is required to read this repository.

## 3. A standalone synthetic operator reproducer

`experiments/operators/stage1_runner.py` constructs its small ONNX graph and independent numerical reference. It does not need pretrained Qwen weights. It requires the vendor-compatible ARM64 ORT wheel, NumPy 1.26.4, ONNX 1.17.0, installed NPU libraries, and a working driver.

On a separately prepared board environment, an example changing-input graph command is:

```sh
export OPI6_EXPERIMENT_ROOT="$HOME/opi6-npu-operator-test"
timeout -k 3 30 python experiments/operators/stage1_runner.py \
  --case qmatmul_dynamic_int8 --runs 10 --tag public-check
```

This is a hardware test, not part of the offline verification. The historical unpatched setup could hang on repeat execution; a userspace timeout cannot guarantee recovery from an uninterruptible kernel wait. The command uses a chosen output directory and does not replace a driver or firmware. No hardware command was executed while preparing this publication.

`stage2_ops.py` generates the tested Transformer-shaped operator cases. Its output root can also be set with `OPI6_EXPERIMENT_ROOT`. The exact case branches and independent checks are visible in the source. `stage2_silu_split.py` records the two-session workaround and retains its original experiment path.

## 4. Native runtime and custom kernels

The C++ runners use the installed `cix_noe_standard_api.h` and dynamically loaded vendor libraries. The recorded build pattern was:

```sh
g++ -O2 -std=c++17 -I/usr/share/cix/include/npu \
  experiments/runtime/native_runner.cpp -ldl -o native_runner
```

They expect the captured graph/input/reference directory layout documented by the source. The complete pretrained graph binaries, packed weights, SDK headers, libraries, and compiler outputs are not bundled. Compiling a runner alone does not recreate those inputs.

The custom GEMV sources and private-op plugin show the packing layout, constant address-space handling, DMA buffering, and QDPA computation. Exact replay of the learned-weight comparison requires reacquiring the pinned model, reproducing its quantization and compiled graph artifacts, and using the recorded vendor toolchain. This repository does not promise a single-command reconstruction of that private build environment.

The Qwen revision is `7ae557604adf67be50417f59c2c2f167def9a775`. Source Safetensors SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. Matched-source Q8_0 GGUF SHA-256: `dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18`. These identify measured artifacts, not files included in this repository.

## 5. Hardware diagnostics are archived source, not an automatic recipe

`experiments/diagnostics/` contains the original source used to collect SCMI GET, SMMU, and NI-700 evidence. Some counter collectors temporarily configure and restore counters; discovery probes only read their allowed registers. They are version-specific kernel code, not ordinary portable applications.

No module-build, module-load, BIOS-flash, security-unlock, SMMU-mode-switch, or clock-SET automation is part of the public verification workflow. Reusing a physical address on another firmware/platform requires separate validation. The purpose of these files here is to let readers audit how the already collected evidence was obtained.

For the same tested system, do not repeat the `clk_summary` read that triggered a recorded clock/runtime-PM deadlock. That incident and its unresolved effect on the retained system state are disclosed in [METHODS.md](METHODS.md).

## 6. What a convincing independent replication would add

An independent clean-boot run, exact toolchain/model provenance, numerical checks, and a complete NPU model with comparable quality would materially strengthen or change these conclusions. Projection speed, CPU model speed, and conditional bandwidth estimates must remain separately labeled.
