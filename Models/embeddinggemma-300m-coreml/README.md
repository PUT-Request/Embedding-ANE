---
language: multilingual
license: gemma
base_model: google/embeddinggemma-300m
tags:
  - coreml
  - apple-neural-engine
  - gemma3
  - sentence-embedding
  - on-device
  - matryoshka
library_name: coreml
pipeline_tag: sentence-similarity
---

## Use it from Swift

<!-- swift-usage-begin -->
### Add the package

`Package.swift`:

```swift
.package(url: "https://github.com/john-rocky/CoreML-LLM", branch: "main"),

// In your target:
.product(name: "CoreMLLLM", package: "CoreML-LLM"),
```

Platforms: iOS 18+ / macOS 15+.

### Download + encode

```swift
import CoreMLLLM

let modelsDir = try FileManager.default.url(
    for: .applicationSupportDirectory, in: .userDomainMask,
    appropriateFor: nil, create: true)

let eg = try await EmbeddingGemma.downloadAndLoad(modelsDir: modelsDir)

// 768-dim L2-normalised embedding
let v = try eg.encode(text: "How do I list files in Swift?")
// Matryoshka: cheap-to-truncate dims (768 / 512 / 256 / 128)
let v256 = try eg.encode(text: "How do I list files in Swift?",
                          dim: 256)

// Task-prefixed (RAG document vs. query)
let q = try eg.encode(text: "list files",
                       task: .retrievalQuery)
let d = try eg.encode(text: "Use FileManager.contentsOfDirectory(...)",
                       task: .retrievalDocument)
```

See [`Gemma3EmbeddingGemma.swift`](https://github.com/john-rocky/CoreML-LLM/blob/main/Sources/CoreMLLLM/Gemma3EmbeddingGemma.swift)
for task prefixes and dim list.
<!-- swift-usage-end -->



# EmbeddingGemma-300M for Apple CoreML (ANE-optimized)

CoreML conversion of `google/embeddinggemma-300m` produced with the
[CoreML-LLM](https://github.com/john-rocky/CoreML-LLM) pipeline. Targets
iOS 26 / macOS 26.

## What's in this repo

| File | Notes |
|---|---|
| `encoder.mlmodelc/` | Compiled stateless bidirectional encoder (INT8, ~295 MB) |
| `encoder.mlpackage/` | Complete generated Core ML source package (INT8, ~295 MB) |
| `model_config.json` | I/O contract, Matryoshka dims, task prefixes |
| `hf_model/` | Tokenizer files |

## ANE residency

**99.80% on Apple Neural Engine** (1950/1954 dispatched ops, verified via
`MLComputePlan` on macOS 26). Achieved by:
- residual-stream rescaling (semantic-preserving fp16 fit)
- fp16-safe L2 normalize (divide by max-abs first to keep `sum(x²)` bounded)
- iOS 26 deployment target

## Use it

Via the [CoreML-LLM Swift package](https://github.com/john-rocky/CoreML-LLM):

```swift
import CoreMLLLM
let bundleURL = try await Gemma3BundleDownloader.download(
    .embeddingGemma300m, into: appSupportDir)
let eg = try await EmbeddingGemma.load(bundleURL: bundleURL)
let vec = try eg.encode(text: "On-device embeddings",
                        task: .retrievalQuery,
                        dim: 768)  // or 512 / 256 / 128 (Matryoshka)
```

I/O contract:
- `input_ids (1, 256) int32`, `attention_mask (1, 256) fp16` (1.0 valid, 0.0 pad)
- `embedding (1, 768) fp16` — L2 unit norm; truncate the trailing dim and
  re-normalize for Matryoshka 512 / 256 / 128

The bundle in this repo is built for `max_seq_len=256`. To choose a different
fixed input length, re-run
`python conversion/build_embeddinggemma_bundle.py --max-seq-len <length> --quantize int8`.

## Sanity check

```
cosine("cat sat on mat", "feline rested on rug") = 0.7345  (high — similar)
cosine("cat sat on mat", "quantum mechanics")   = 0.4650  (low — different)
```

## License

Inherits Google's [Gemma terms of use](https://ai.google.dev/gemma/terms).

<!-- funnel:v1 -->

---

**More models in this format:** [Core ML Model Zoo](https://huggingface.co/collections/mlboydaisuke/core-ml-model-zoo-6a7078dc888e7b13efd35631) — 46 models, each with the recipe that produced it.

**Want a different model on-device?** [Open a request](https://github.com/john-rocky/on-device-requests) — free, open weights only; the export and its measured numbers get published publicly.

<!-- /funnel:v1 -->
