"""Write the reference outputs the Mojo tests compare against.

    pixi run -e fixtures python tools/make_fixtures.py

For each model, downloads its files into build/models/<name>/ (gitignored)
and writes tests/fixtures/<name>/:
  texts.txt        one input per line (no newlines inside an input)
  ids.txt          HF tokenizer ids per input, space-separated, with
                   [CLS]/[SEP], truncated to MAX_TOKENS
  embeddings.f32   sentence-transformers output, float32 little-endian,
                   one row of `dim` values per input (normalized)
  meta.txt         model id, revision, dim, pooling, count
Only this script needs Python; the tests read these files.
"""

import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 512
FILES = ["config.json", "model.safetensors", "tokenizer.json", "vocab.txt",
         "tokenizer_config.json", "modules.json", "1_Pooling/config.json",
         "sentence_bert_config.json", "config_sentence_transformers.json"]

MODELS = [
    # (local name, hub id, pooling)
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", "cls"),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "mean"),
]


def texts() -> list[str]:
    base = [
        "",
        "a",
        "Hello, world!",
        "Mojo bindings for SQLite",
        "A fast JSON parser written in Mojo 🔥",
        "GPU kernels for matrix multiplication on Apple silicon and NVIDIA",
        "Learning Mojo: exercises from Advent of Code 2024",
        "pixi.toml, magic.lock and mojoproject.toml — what changed?",
        "fix: bump mojo to 25.4 and update the lock file",
        "feat(tensor): add broadcasting to elementwise ops",
        "chore(deps): update max to 26.7.0",
        "Merge pull request #42 from someone/feature-branch",
        "Revert \"Use SIMD width 8 on ARM\"",
        "WIP",
        "README.md",
        "def main():\tprint(\"tabs\tinside\")",
        "struct Point[T: DType](Copyable): var x: Scalar[T]",
        "https://github.com/modular/modular/tree/main/max/kernels",
        "e-mail: someone@example.com, phone +1 (555) 010-9999",
        "Numbers 3.14159, 2.71828, 1e-12 and 1,000,000.",
        "UPPER CASE TEXT SHOULD BE LOWERCASED BY AN UNCASED MODEL",
        "naïve café résumé coöperate Ångström",
        "Grüße aus München — Straße, Fußball",
        "日本語のテキストと漢字",
        "中文测试：机器学习模型",
        "한국어 문장입니다",
        "Русский текст для проверки",
        "Ελληνικά γράμματα αβγδ",
        "العربية نص تجريبي",
        "עברית טקסט",
        "हिन्दी पाठ परीक्षण",
        "ไทย ทดสอบ",
        "emoji only 😀😃😄😁🚀🔥✨",
        "mixed: 🔥Mojo🔥 is🚀fast",
        "zero​width​space and non breaking",
        "combining accents: é à ö",
        "control chars \u0007 bell and \u001b escape",
        "punctuation!!! ??? ... ,,, ;;; ::: --- ___ ***",
        "(parentheses) [brackets] {braces} <angles>",
        "unbelievably antidisestablishmentarianism supercalifragilistic",
        "x" * 300,
        "tokens " * 400,
        " ".join(f"word{i}" for i in range(600)),
    ]
    topics = ["inference", "bindings", "tooling", "web server", "game",
              "data frames", "tokenizer", "HTTP client", "linear algebra",
              "image processing", "audio", "CLI", "testing framework",
              "async runtime", "parser", "compression", "database driver"]
    adjectives = ["fast", "simple", "experimental", "pure Mojo", "minimal",
                  "GPU-accelerated", "zero-copy", "type-safe"]
    out = list(base)
    i = 0
    while len(out) < 200:
        t = topics[i % len(topics)]
        a = adjectives[(i * 7) % len(adjectives)]
        out.append(f"A {a} {t} library for Mojo, version 0.{i % 10}.{i % 7}")
        i += 1
    assert all("\n" not in s and "\r" not in s for s in out)
    return out


def main() -> None:
    inputs = texts()
    for name, hub_id, pooling in MODELS:
        model_dir = ROOT / "build" / "models" / name
        path = snapshot_download(hub_id, allow_patterns=FILES,
                                 local_dir=model_dir)
        st = SentenceTransformer(str(model_dir), device="cpu")
        st.max_seq_length = MAX_TOKENS
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(MAX_TOKENS)
        tok.no_padding()

        out = ROOT / "tests" / "fixtures" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "texts.txt").write_text("\n".join(inputs) + "\n",
                                       encoding="utf-8")
        ids = [tok.encode(s).ids for s in inputs]
        (out / "ids.txt").write_text(
            "\n".join(" ".join(map(str, r)) for r in ids) + "\n")

        emb = st.encode(inputs, batch_size=16, convert_to_numpy=True,
                        normalize_embeddings=True).astype("<f4")
        emb.tofile(out / "embeddings.f32")

        from huggingface_hub import HfApi
        sha = HfApi().model_info(hub_id).sha
        (out / "meta.txt").write_text(
            f"model {hub_id}\nrevision {sha}\ndim {emb.shape[1]}\n"
            f"pooling {pooling}\ncount {len(inputs)}\n"
            f"max_tokens {MAX_TOKENS}\n")
        print(f"{name}: {len(inputs)} inputs, dim {emb.shape[1]}, "
              f"max ids {max(map(len, ids))}, rev {sha}")


if __name__ == "__main__":
    sys.exit(main())
