# encoder.mojo

BERT-family text encoders in Mojo, on max-core (CPU and GPU).

A library a Mojo pipeline binary links to turn text into sentence
embeddings, with no Python at runtime and the weights kept outside the image.
The first consumer is mojo-census's `mojo_embed` pipeline.

```mojo
from encoder import Encoder

def main() raises:
    # A sentence-transformers model directory: config.json, model.safetensors,
    # vocab.txt and 1_Pooling/config.json.
    var enc = Encoder("models/bge-small-en-v1.5")
    var texts: List[String] = ["Mojo bindings for SQLite", "A JSON parser"]
    var emb = enc.encode(texts)  # [len(texts), enc.dim()] flattened, L2-normalized
    print(len(emb) // enc.dim(), "x", enc.dim())
```

**Supported models.** BERT encoders with absolute positions and GELU, the
uncased WordPiece tokenizer, and CLS or mean pooling. Weights may be F32,
BF16 or F16. These two are tested:

| Model | Layers | Pooling | Tokenizer ids vs HF `tokenizers` | Cosine vs sentence-transformers |
| --- | --- | --- | --- | --- |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 12 | CLS | 200/200 identical | min 0.99999999999 |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 6 | mean | 200/200 identical | min 0.99999999999 |

The 200 fixture inputs cover empty and 600-word texts (truncated at 512
tokens), accents, CJK, Korean, Arabic, Hindi, emoji, control and zero-width
characters, code and commit messages.
[`tools/make_fixtures.py`](tools/make_fixtures.py) writes them, and only it
needs Python (`pixi run -e fixtures`).

**Speed.** Throughput on 2,000 repo-description-sized texts (36 tokens each on
average), batch 32 (`pixi run bench [model] [count] [batch]`), on 2026-11-09:

| Machine | bge-small-en-v1.5 | all-MiniLM-L6-v2 | Peak RSS (bge) |
| --- | --- | --- | --- |
| Apple M4 (10 cores) | 442 texts/s | 847 texts/s | 420 MB |
| same, sentence-transformers on PyTorch 2.14 CPU (4 threads, its default) | 161 texts/s | 315 texts/s | |
| linux-aarch64, 8 vCPU container on the M4 | 196 texts/s | 377 texts/s | 310 MB |
| same, 4 cores (a Hetzner CAX21 has 4 vCPU) | 180 texts/s | | |
| same, 2 cores | 115 texts/s | | |

How it runs:
- **Linear layers:** max-core's `linalg.matmul` over every token of a batch
  at once. Sequences are packed with no padding.
- **Attention:** a SIMD loop per (sequence, head), run in parallel. For short
  texts this is cheaper than a matmul call per head.
- **Bias, GELU, residual and layernorm:** vectorized and parallel over rows.
- **Not done yet:** the GPU path, and tuning for Linux. Throughput goes from
  4 to 8 cores barely moves.

```sh
pixi run test     # tokenizer, safetensors and embedding parity (downloads ~220 MB of models)
pixi run bench
tools/linux-arm64.sh test   # the same on linux-aarch64, in a container
```

## Smoke test: max-core from a plain Mojo binary

[`smoke/`](smoke/) checks that `max-core`'s kernel packages work from an ordinary
`mojo build`, without the Python MAX packages
([#1](https://github.com/magmalake/encoder.mojo/issues/1)).

```sh
pixi run smoke-cpu            # this machine's CPU
tools/linux-arm64.sh smoke-cpu # linux-aarch64 in a container (needs docker)
pixi run smoke-gpu            # GPU (Apple silicon: needs Xcode's Metal Toolchain)
```

Results on 2026-11-09 with max-core 26.7.0 and mojo 1.2.0, at bge-small shapes
(hidden 384, FFN 1536, 128 tokens). Every kernel was checked against a naive
loop. Linux shows the range over three runs, each a single timed call after a
warm-up.

| Kernel (entry point) | osx-arm64, M4 CPU | linux-aarch64, 8 vCPU container on the M4 |
| --- | --- | --- |
| `linalg.matmul.matmul[transpose_b=True]`, 128×384 · 384×384 | 0.04 ms | 0.26–0.44 ms (90–145 GFLOP/s) |
| same, 128×384 · 1536×384 | 0.12 ms | 0.63–0.87 ms (175–240 GFLOP/s) |
| same, 128×1536 · 384×1536 | 0.12 ms | 0.54–0.87 ms (175–280 GFLOP/s) |
| `builtin_kernels.rowwise_normalization.layer_norm`, 128×384 | 0.15 ms, err 6e-7 | 0.05–0.06 ms, err 6e-7 |
| `nn.softmax.softmax_2_pass`, 128 | err 1.5e-8 | err 1.5e-8 |
| `nn.activations.gelu` (exact, erf) | err 4.8e-7 | err 4.8e-7 |

GPU, `linalg.matmul.matmul[transpose_b=True]` through `DeviceContext` on the
Apple M4. Each figure is the average of 20 calls. The error is against a
float64 loop and stays at 8e-5 or below:

| Shape | Time | GFLOP/s |
| --- | --- | --- |
| 128×384 · 384×384 | 0.06 ms | 650 |
| 128×384 · 1536×384 | 0.23 ms | 640 |
| 128×1536 · 384×1536 | 0.18 ms | 845 |
| 4096×384 · 1536×384 | 1.77 ms | 2,730 |

At one 128-token sequence the M4's GPU is slower than its CPU, which runs
matmul through Accelerate. It pays off only on large batches (4,096 rows
above).

What this means for the encoder:

- **No Python at runtime.** The linux-aarch64 binary is 420 KB. Besides libc it
  needs about 4 MB of shared libraries from the env: `libKGENCompilerRTShared`,
  `libAsyncRTMojoBindings`, `libAsyncRTRuntimeGlobals`, `libMSupportGlobals`,
  `libgcc_s` and `libstdc++`.
- **Linux needs a C compiler and glibc ≥ 2.34 at build time.** `pixi.toml` adds
  conda-forge `gcc` and `sysroot_linux-*` ≥ 2.34. With conda's default sysroot,
  the link fails on `dlerror@GLIBC_2.34`.
- **The GPU path works on Metal.** It needs Xcode's Metal Toolchain at build
  time (`xcodebuild -downloadComponent MetalToolchain`). Without it, `mojo build`
  stops with "missing Metal Toolchain".

- **Pin stable.** The nightly (26.8.0.dev2026110905) breaks these entry points
  week to week:
  - `target` parameters were removed from `matmul` and `layer_norm`, and are
    now taken from the `DeviceContext` type.
  - `softmax_2_pass` is gone.
  - `UnsafePointer` and the `alloc(count, alignment=)` form were removed from
    the stdlib.

  The `nightly` env is kept to track the next port.
- **macOS CPU matmul** is fast because on Apple silicon it goes through
  Accelerate (`linalg/matmul/cpu/apple_accelerate.mojo`). Linux gets max-core's
  own NEON kernels.
