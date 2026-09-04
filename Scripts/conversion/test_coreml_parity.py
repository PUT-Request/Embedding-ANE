#!/usr/bin/env python3
"""CoreML mlpackage vs HF reference parity check.

Loads output/embeddinggemma-300m/bundle/encoder.mlpackage and runs the
same multilingual sentence set through it, comparing cosine similarity
against the HF SentenceTransformer reference.

Pass criteria:
    mean cosine(hf_emb, coreml_emb) at d=768  >=  0.99
    mean cosine at d=128 after Matryoshka truncate+renorm  >=  0.96
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from test_embeddinggemma_parity import SENTENCES, _cosine, _matryoshka_truncate, _load_hf_reference


def _encode_coreml(mlpackage: str, hf_dir: str, sentences: list[str],
                   max_seq_len: int) -> torch.Tensor:
    import coremltools as ct
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = ct.models.MLModel(mlpackage, compute_units=ct.ComputeUnit.ALL)

    out = []
    for s in sentences:
        enc = tok(s, padding="max_length", truncation=True,
                  max_length=max_seq_len, return_tensors="pt")
        input_ids = enc["input_ids"].numpy().astype(np.int32)
        attn_mask = enc["attention_mask"].numpy().astype(np.float16)
        res = model.predict(
            {"input_ids": input_ids, "attention_mask": attn_mask}
        )
        out.append(torch.from_numpy(res["embedding"].reshape(-1)).to(torch.float32))
    return torch.stack(out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="CoreML vs HF parity test")
    parser.add_argument("--mlpackage", type=str,
                        default=os.path.join(ROOT, "..", "output",
                                             "embeddinggemma-300m", "bundle",
                                             "encoder.mlpackage"))
    parser.add_argument("--hf-dir", type=str, default=os.path.join(ROOT, "..", "Models", "original"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isdir(args.mlpackage):
        raise SystemExit(f"mlpackage not found: {args.mlpackage}")

    hf_emb = _load_hf_reference(args.hf_dir, SENTENCES)
    our_emb = _encode_coreml(args.mlpackage, args.hf_dir, SENTENCES, args.max_seq_len)

    cos768 = [_cosine(h, o) for h, o in zip(hf_emb, our_emb)]
    print(f"\nCoreML cosine at d=768: mean={np.mean(cos768):.4f}  min={np.min(cos768):.4f}")

    cos128 = []
    for h, o in zip(hf_emb, our_emb):
        cos128.append(_cosine(_matryoshka_truncate(h, 128), _matryoshka_truncate(o, 128)))
    print(f"CoreML cosine at d=128: mean={np.mean(cos128):.4f}  min={np.min(cos128):.4f}")

    pass768 = float(np.mean(cos768)) >= 0.99
    pass128 = float(np.mean(cos128)) >= 0.96
    print(f"\nCOREML d=768: {'PASS' if pass768 else 'FAIL'}  (need mean >=0.99)")
    print(f"COREML d=128: {'PASS' if pass128 else 'FAIL'}  (need mean >=0.96)")
    sys.exit(0 if (pass768 and pass128) else 1)


if __name__ == "__main__":
    main()
