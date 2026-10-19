# How to read the evidence directory

The final interpretation is in `ARTICLE.md`, `RESULTS.md`, and `CORRECTIONS.md` at the repository root. File names reflect the historical experiment stages. They do not automatically make every earlier hypothesis a current conclusion.

In particular, `native-summary.json` retains the original failed NOE invocation as historical evidence. The corrected NOE loader and final performance comparison are recorded by `review-summary.json` and `review-graph-suite.json`. The failed invocation is not evidence that NOE cannot work.

Each file selected from the private workspace has an entry in `PROVENANCE.json`, including its original SHA-256, exported SHA-256, omitted fields, and transformation description. Personal paths, private network details, and host UUIDs are redacted. JSON formatting is normalized; numerical measurement values are preserved.

`profile-decoded/`, `gemv-profile-decoded/`, `mixed-bandwidth-decoded/`, and `qos-decoded/` hold selected per-run output from the vendor profiler. They are decoded records, not the raw hardware trace binaries. Those binaries, model weights, and vendor tools are not redistributed here.

The SMMU and NI-700 analysis records include addresses, raw discovery values, and counter results needed to inspect the stated interpretation. They are not measurements of the DDR controller's complete traffic, and discovery does not reveal the data-routing graph.

`environment.json` is a publication inventory assembled from saved system records. `external-sources.json` identifies the public pages checked for the article. Neither file was created by a new benchmark.
