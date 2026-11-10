# encoder.mojo

BERT-family text encoders in Mojo, on max-core (CPU and GPU).

The goal is a library a Mojo pipeline binary links to turn text into sentence
embeddings, with no Python at runtime and the weights kept outside the image
(see [#2](https://github.com/magmalake/encoder.mojo/issues/2)). The first
consumer is mojo-census's `mojo_embed` pipeline.

## Smoke test: max-core from a plain Mojo binary

[`smoke/`](smoke/) checks that `max-core`'s kernel packages work from an ordinary
`mojo build`, without the Python MAX packages
([#1](https://github.com/magmalake/encoder.mojo/issues/1)).

```sh
pixi run smoke-cpu            # this machine's CPU
smoke/linux-arm64.sh          # linux-aarch64 in a container (needs docker)
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

What this means for the encoder:

- **No Python at runtime.** The linux-aarch64 binary is 420 KB. Besides libc it
  needs about 4 MB of shared libraries from the env: `libKGENCompilerRTShared`,
  `libAsyncRTMojoBindings`, `libAsyncRTRuntimeGlobals`, `libMSupportGlobals`,
  `libgcc_s` and `libstdc++`.
- **Linux needs a C compiler and glibc ≥ 2.34 at build time.** `pixi.toml` adds
  conda-forge `gcc` and `sysroot_linux-*` ≥ 2.34. With conda's default sysroot,
  the link fails on `dlerror@GLIBC_2.34`.
- **The GPU path needs the Metal Toolchain.** `mojo build` of
  [`smoke/gpu.mojo`](smoke/gpu.mojo) stops with "missing Metal Toolchain"
  (`xcodebuild -downloadComponent MetalToolchain`), on stable and on the nightly.
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
