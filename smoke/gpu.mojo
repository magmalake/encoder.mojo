# Smoke test: max-core's GPU matmul from a plain Mojo binary (Metal on Apple
# silicon). Same BERT linear-layer shapes as cpu.mojo, checked against a
# float64 loop on the host.

from std.random import random_float64, seed
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from linalg.matmul import matmul

comptime F32 = DType.float32


def smoke_matmul[K: Int, N: Int](ctx: DeviceContext, m: Int) raises:
    """C[m, N] = A[m, K] @ B[N, K]^T on the GPU."""
    var a_h = ctx.enqueue_create_host_buffer[F32](m * K)
    var b_h = ctx.enqueue_create_host_buffer[F32](N * K)
    var c_h = ctx.enqueue_create_host_buffer[F32](m * N)
    ctx.synchronize()
    for i in range(m * K):
        a_h[i] = Float32(random_float64(-1.0, 1.0))
    for i in range(N * K):
        b_h[i] = Float32(random_float64(-1.0, 1.0))

    var a_d = ctx.enqueue_create_buffer[F32](m * K)
    var b_d = ctx.enqueue_create_buffer[F32](N * K)
    var c_d = ctx.enqueue_create_buffer[F32](m * N)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)

    var a = TileTensor(a_d, row_major(m, Idx[K]))
    var b = TileTensor(b_d, row_major(Idx[N], Idx[K]))
    var c = TileTensor(c_d, row_major(m, Idx[N]))

    matmul[transpose_b=True, target="gpu"](c, a, b, ctx)  # warm-up + compile
    ctx.synchronize()
    comptime REPS = 20
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        matmul[transpose_b=True, target="gpu"](c, a, b, ctx)
    ctx.synchronize()
    var ns = Int(perf_counter_ns() - t0) // REPS

    ctx.enqueue_copy(c_h, c_d)
    ctx.synchronize()

    var worst: Float64 = 0
    for i in range(m):
        for j in range(N):
            var acc: Float64 = 0
            for p in range(K):
                acc += Float64(a_h[i * K + p]) * Float64(b_h[j * K + p])
            worst = max(worst, abs(acc - Float64(c_h[i * N + j])))

    var gflops = 2.0 * Float64(m * N * K) / Float64(ns)
    print(
        t"gpu matmul {m}x{K} @ ({N}x{K})^T: max |err| = {worst},"
        t" {Float64(ns) / 1e6} ms ({gflops} GFLOP/s)"
    )
    if worst > 1e-2:
        raise Error(t"gpu matmul error {worst} above 1e-2")


def main() raises:
    seed(42)
    with DeviceContext() as ctx:
        print("device:", ctx.name())
        smoke_matmul[384, 384](ctx, 128)
        smoke_matmul[384, 1536](ctx, 128)
        smoke_matmul[1536, 384](ctx, 128)
        smoke_matmul[384, 1536](ctx, 4096)
    print("gpu smoke: ok")
