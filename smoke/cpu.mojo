# Smoke test: max-core's CPU kernels from a plain Mojo binary (no Python).
# Each kernel runs at an encoder-sized shape and is checked against a naive
# loop. Prints one line per kernel; raises on the first mismatch.

from std.math import erf, exp, rsqrt, sqrt
from std.random import random_float64, seed
from std.sys.info import simd_width_of
from std.time import perf_counter_ns

from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul
from nn.activations import gelu
from nn.softmax import softmax_2_pass
from builtin_kernels.rowwise_normalization import layer_norm

from std.utils.index import Index, IndexList

comptime F32 = DType.float32


def random_list(n: Int, lo: Float64, hi: Float64) -> List[Float32]:
    var out = List(length=n, fill=Float32(0))
    for i in range(n):
        out[i] = Float32(random_float64(lo, hi))
    return out^


def check(name: String, err: Float32, tol: Float32, ns: Int) raises:
    print(t"{name}: max |err| = {err}, {Float64(ns) / 1e6} ms")
    if err > tol:
        raise Error(t"{name}: error {err} above {tol}")


def smoke_matmul(m: Int, k: Int, n: Int) raises:
    """C[m, n] = A[m, k] @ B[n, k]^T, the shape of a BERT linear layer
    (safetensors store weights as [out, in])."""
    var a_l = random_list(m * k, -1.0, 1.0)
    var b_l = random_list(n * k, -1.0, 1.0)
    var c_l = List(length=m * n, fill=Float32(0))
    var a = TileTensor(a_l, row_major(m, k))
    var b = TileTensor(b_l, row_major(n, k))
    var c = TileTensor(c_l, row_major(m, n))

    matmul[transpose_b=True](c, a, b)  # warm-up
    var t0 = perf_counter_ns()
    matmul[transpose_b=True](c, a, b)
    var ns = Int(perf_counter_ns() - t0)

    var worst: Float32 = 0
    for i in range(m):
        for j in range(n):
            var acc: Float32 = 0
            for p in range(k):
                acc += a_l[i * k + p] * b_l[j * k + p]
            worst = max(worst, abs(acc - c_l[i * n + j]))

    var gflops = 2.0 * Float64(m * n * k) / Float64(ns)
    check(
        String(t"matmul {m}x{k} @ ({n}x{k})^T ({gflops} GFLOP/s)"),
        worst,
        1e-3,
        ns,
    )


def smoke_layer_norm(rows: Int, cols: Int) raises:
    var in_l = random_list(rows * cols, -3.0, 3.0)
    var out_l = List(length=rows * cols, fill=Float32(0))
    var gamma_l = random_list(cols, 0.5, 1.5)
    var beta_l = random_list(cols, -0.5, 0.5)
    var eps: Float32 = 1e-12  # BERT's layer_norm_eps

    var shape = Index(rows, cols)
    var input_buf = TileTensor(in_l, row_major(Coord(shape)))
    var output_buf = TileTensor(out_l, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_l, row_major(Coord(IndexList[1](cols))))
    var beta = TileTensor(beta_l, row_major(Coord(IndexList[1](cols))))

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var input_buf} -> SIMD[F32, width]:
        var idx = input_buf.layout(coords)
        return input_buf.raw_load[width=width, alignment=alignment](idx)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[F32, width]) {var output_buf}:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[F32, width]](val)
        )

    var t0 = perf_counter_ns()
    layer_norm[F32, 2, target="cpu"](
        input_fn, output_fn, Coord(shape), cols, gamma, beta, eps
    )
    var ns = Int(perf_counter_ns() - t0)

    var worst: Float32 = 0
    for r in range(rows):
        var mean: Float64 = 0
        for c in range(cols):
            mean += Float64(in_l[r * cols + c])
        mean /= Float64(cols)
        var var_: Float64 = 0
        for c in range(cols):
            var d = Float64(in_l[r * cols + c]) - mean
            var_ += d * d
        var_ /= Float64(cols)
        var inv = 1.0 / sqrt(var_ + Float64(eps))
        for c in range(cols):
            var want = (Float64(in_l[r * cols + c]) - mean) * inv * Float64(
                gamma_l[c]
            ) + Float64(beta_l[c])
            worst = max(worst, abs(Float32(want) - out_l[r * cols + c]))
    check(String(t"layer_norm {rows}x{cols}"), worst, 1e-4, ns)


def smoke_softmax(n: Int) raises:
    """One attention row: softmax over n scores."""
    var in_l = random_list(n, -8.0, 8.0)
    var out_l = List(length=n, fill=Float32(0))
    var in_tt = TileTensor(in_l, row_major(n))
    var out_tt = TileTensor(out_l, row_major(n))

    var t0 = perf_counter_ns()
    softmax_2_pass[simd_width_of[F32](), F32](out_tt, in_tt)
    var ns = Int(perf_counter_ns() - t0)

    var mx = in_l[0]
    for i in range(n):
        mx = max(mx, in_l[i])
    var total: Float64 = 0
    for i in range(n):
        total += Float64(exp(in_l[i] - mx))
    var worst: Float32 = 0
    for i in range(n):
        var want = Float64(exp(in_l[i] - mx)) / total
        worst = max(worst, abs(Float32(want) - out_l[i]))
    check(String(t"softmax {n}"), worst, 1e-6, ns)


def smoke_gelu(n: Int) raises:
    comptime W = simd_width_of[F32]()
    var worst: Float32 = 0
    var t0 = perf_counter_ns()
    for i in range(0, n, W):
        var x = SIMD[F32, W]()
        for l in range(W):
            x[l] = Float32(-6.0 + 12.0 * Float64(i + l) / Float64(n))
        var y = gelu(x)
        for l in range(W):
            var xv = Float64(x[l])
            var want = 0.5 * xv * (1.0 + erf(xv / 1.4142135623730951))
            worst = max(worst, abs(Float32(want) - y[l]))
    var ns = Int(perf_counter_ns() - t0)
    check(String(t"gelu {n}"), worst, 1e-5, ns)


def main() raises:
    seed(42)
    # bge-small: hidden 384, FFN 1536, 128 tokens is a typical batch row.
    smoke_matmul(128, 384, 384)
    smoke_matmul(128, 384, 1536)
    smoke_matmul(128, 1536, 384)
    smoke_layer_norm(128, 384)
    smoke_softmax(128)
    smoke_gelu(128 * 1536)
    print("cpu smoke: ok")
