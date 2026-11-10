# Changelog

## Unreleased

- **Less memory to load a model:** the safetensors file is read with its size known, so loading bge-small takes 271 MB instead of 400 MB.

## 0.1.0 (2026-11-09)

The first release. `Encoder(dir).encode(texts)` returns L2-normalized
sentence embeddings for a sentence-transformers BERT directory, built on
max-core 26.7.0 with mojo 1.2.0.

- WordPiece tokenizer matching HF `tokenizers` (BertNormalizer,
  BertPreTokenizer, `##` continuations, truncation to 512 tokens).
- safetensors loader for F32, BF16 and F16 weights.
- CPU forward pass: packed batches through max-core's matmul, parallel
  SIMD attention, CLS or mean pooling.
- Tested against sentence-transformers on bge-small-en-v1.5 and
  all-MiniLM-L6-v2 (cosine ≥ 0.99999999999 on 200 inputs).
