# Attribution and redistribution boundaries

The MIT license in this repository applies to the original project code and documentation, except the archived diagnostic kernel modules, which retain their GPL declarations. It does not relicense external projects, vendor SDKs, firmware, model weights, manuals, or logs' underlying software. The diagnostic sources are provided under GPL version 2; see the [GPLv2 text](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).

- **Community AIPU fix:** [n4hy/NPU_OrangePi6Plus](https://github.com/n4hy/NPU_OrangePi6Plus). The stability work used this published fix. The repository does not bundle the full vendor driver tree or claim authorship of that fix.
- **Arm China / Zhouyi / Compass and CIX:** APIs and toolchain components used by the original test code. Vendor headers, libraries, SDK archives, compiled model binaries, and manuals are linked rather than bundled.
- **Arm NI-700 / MMU-700 and Linux:** public documentation and source informed the register/event interpretation. No full Arm manual or Linux source tree is redistributed here. Diagnostic C files are experiment-specific collectors, not a replacement for the platform's drivers.
- **Qwen and llama.cpp:** model and CPU baseline provenance are documented in `SOURCES.md`. Pretrained weights and generated GGUF files are excluded.
- **Codex:** assisted the owner with implementation, diagnosis, analysis, and English editing. Assistance does not constitute independent verification.

Selected logs and decoded measurement records are published as experimental evidence. Publication transforms host-identifying strings and formatting as recorded in `evidence/PROVENANCE.json`. Device counters, numerical test results, and relevant timings remain unchanged.
