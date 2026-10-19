# Prior work and the scope of this follow-up

Checked on 19 October 2026. This is a follow-up measurement report, not a claim to have discovered the general difficulty of LLM inference on this NPU. Related public work predates our 15–19 October experiments.

| Earlier source | Overlap | How this project should be read |
|---|---|---|
| [happyme531, X2 LLM development report](https://github.com/happyme531/happyme531/blob/6fce390b3b5be211118f19d73ab3acedd79e425b/CD8180-%22%E5%91%A8%E6%98%93%22X2-NPU-LLM-%E5%AE%9E%E9%AA%8C%E4%B8%8E%E5%BC%80%E5%8F%91%E6%80%BB%E7%BB%93.md), committed 27 July 2026 | Custom kernels, three-core experiments and a TPC DMA plateau near 24 GB/s. | Closest prior work. Our bandwidth observations are corroboration, not the first discovery of that plateau. |
| [n4hy, NPU_OrangePi6Plus](https://github.com/n4hy/NPU_OrangePi6Plus/blob/9619be3b15d8935095adbb51cd7335c7ebbb37d0/README.md), README history starts 4 February 2026 | Second-inference hang and command-pool fix. | We used the community fix. We claim subsequent measurements, not authorship of that fix. |
| [visorcraft, CIXBUILDER.md](https://github.com/visorcraft/orange-pi-6-plus-npu/blob/c6f679de45ef0036b8926e83ee82280280657852/CIXBUILDER.md), 11 February 2026 | Compiler and transformer-model deployment difficulties. | These general developer-experience problems were already public. |
| [ncz-os, INFERENCE_LIMITS.md](https://github.com/ncz-os/cix-installer/blob/f6e5b415da4a1dca2566ad451699b7bc2301d184/docs/INFERENCE_LIMITS.md), history starts 23 June 2026 | Practical accelerator selection and limitations of the available inference stack. | Related community observations; its broad architectural assertions are not adopted as proved facts here. |
| [CIX model-hub issue #1](https://github.com/cixtech/ai_model_hub/issues/1), opened 12 November 2025 | Users asked about missing NPU LLM deployment; a November reply described CPU/GPU LLM support. | An earlier public discussion, not a guarantee about all later releases. Commenter affiliation was not independently established. |
| [mrobinson-opi, 6plus_npu](https://github.com/mrobinson-opi/6plus_npu/blob/42a34aad24ac8a0690b65715c9cb45b326384a2b/README.md), README history starts 25 November 2025 | NPU setup and testing. | Basic bring-up is established prior work, not our contribution. |

Dates above come from public Git history for the named document or the issue timestamp. Immutable document links and SHA-256 hashes are listed in [the source index](evidence/related-work-sources.json). A document's first commit date does not prove every statement was present in that first revision.

## What our own records add

The following are results of this project's recorded runs. Equivalent numbers or the complete combination were not found in the reviewed documents; that is a scoped search result, not a worldwide priority claim.

- Incorrect causal-mask Softmax and RoPE outputs, a SiLU compiler crash, CPU controls and graph-specific workarounds, with saved evidence.
- Four matched Qwen2.5-0.5B projection graphs compared through CPU ORT, NPU ORT and direct NOE: the corrected direct path wins by about 1.76 times at M=1 and 2.94 times at M=64.
- A full CPU Q8_0 reference converted from the same model revision, measured at about 69.679 tokens/s. Scaling the separate projection timings is explicitly an estimate, not a complete NPU LLM measurement.
- Corrected split/Concat quantization metadata, hardware-profiled three-core execution, stock-kernel comparison and an INT8 GEMV attempt that did not beat the stock path.
- Additional stock AIFF, mixed DMA, QoS, queue, SMMU and NoC-path evidence. The physical bandwidth ceiling remains unresolved.

See [results](RESULTS.md), [methods](METHODS.md) and [provenance](evidence/PROVENANCE.json) for the measurements behind those statements.

## Duplication assessment

There is substantial overlap in the question and general conclusions. Publishing only another claim that TOPS does not imply fast LLM decoding would add little. Publishing the narrower correctness records, matched comparisons and reproducible follow-up is worthwhile, provided the earlier work is credited prominently.

The review covered public GitHub searches, the documents above and their linked discussions. It is not an exhaustive code-clone or plagiarism audit of every GitHub repository. We did not obtain a complete public source tree for every experiment described in the closest report, so we cannot certify source-level uniqueness. Shared vendor API calls, hardware terminology and common benchmark methods are expected. No claim of absolute novelty is made.
