# Embeddings match sentence-transformers (cosine ≥ 0.999) on every fixture
# input, for each model. Needs build/models/ (tools/make_fixtures.py).

from std.math import sqrt
from std.memory import bitcast
from std.time import perf_counter_ns

from encoder import Encoder


def read_lines(path: String) raises -> List[String]:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var out = List[String]()
    for line in text.split("\n"):
        out.append(String(line))
    if len(out) > 0 and out[len(out) - 1].byte_length() == 0:
        _ = out.pop()
    return out^


def read_f32(path: String) raises -> List[Float32]:
    var b: List[UInt8]
    with open(path, "r") as f:
        b = f.read_bytes()
    var out = List[Float32](capacity=len(b) // 4)
    for i in range(0, len(b), 4):
        var bits = (
            UInt32(b[i])
            | (UInt32(b[i + 1]) << 8)
            | (UInt32(b[i + 2]) << 16)
            | (UInt32(b[i + 3]) << 24)
        )
        out.append(bitcast[DType.float32](bits))
    return out^


def check_model(name: String) raises -> Int:
    var t0 = perf_counter_ns()
    var enc = Encoder(String("build/models/") + name)
    var load_ms = Float64(perf_counter_ns() - t0) / 1e6
    var texts = read_lines(String("tests/fixtures/") + name + "/texts.txt")
    var want = read_f32(String("tests/fixtures/") + name + "/embeddings.f32")
    var d = enc.dim()
    t0 = perf_counter_ns()
    var got = enc.encode(texts)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var bad = 0
    var worst: Float64 = 1
    for i in range(len(texts)):
        var dot: Float64 = 0
        var na: Float64 = 0
        var nb: Float64 = 0
        for j in range(d):
            var a = Float64(got[i * d + j])
            var b = Float64(want[i * d + j])
            dot += a * b
            na += a * a
            nb += b * b
        var cos = dot / sqrt(na * nb)
        worst = min(worst, cos)
        if cos < 0.999:
            bad += 1
            if bad <= 5:
                print("LOW COSINE", name, "#", i, cos, repr(texts[i]))
    print(
        t"{name}: {len(texts) - bad}/{len(texts)} cosine >= 0.999,"
        t" min {worst}; load {load_ms} ms, encode {ms} ms"
    )
    return bad


def main() raises:
    var bad = 0
    bad += check_model("bge-small-en-v1.5")
    bad += check_model("all-MiniLM-L6-v2")
    if bad > 0:
        raise Error(bad, " embeddings below cosine 0.999")
