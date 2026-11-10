# Throughput on repo-description-sized texts (about 15-40 tokens).
#   pixi run bench [model] [count] [batch]

from std.sys import argv
from std.time import perf_counter_ns

from encoder import Encoder


def texts(n: Int) -> List[String]:
    var topics: List[String] = [
        "inference engine",
        "SQLite bindings",
        "package manager",
        "web server",
        "game engine",
        "data frame library",
        "tokenizer",
        "HTTP client",
        "linear algebra kernels",
        "image processing",
        "audio DSP",
        "CLI parser",
        "testing framework",
        "async runtime",
        "JSON parser",
        "compression codec",
    ]
    var adjectives: List[String] = [
        "fast",
        "simple",
        "experimental",
        "pure Mojo",
        "minimal",
        "GPU-accelerated",
        "zero-copy",
        "type-safe",
    ]
    var out = List[String](capacity=n)
    for i in range(n):
        out.append(
            String(
                "A ",
                adjectives[(i * 7) % len(adjectives)],
                " ",
                topics[i % len(topics)],
                " for Mojo with examples, benchmarks and docs; supports ",
                "Linux and macOS, version 0.",
                i % 10,
                ".",
                i % 7,
                " (issue #",
                i,
                ")",
            )
        )
    return out^


def main() raises:
    var args = argv()
    var model = String(args[1]) if len(args) > 1 else String(
        "bge-small-en-v1.5"
    )
    var n = Int(String(args[2])) if len(args) > 2 else 2000
    var batch = Int(String(args[3])) if len(args) > 3 else 32
    var enc = Encoder(String("build/models/") + model)
    var inputs = texts(n)
    var tokens = 0
    for t in inputs:
        tokens += len(enc.tokenizer.encode(t))
    _ = enc.encode(texts(batch), batch)  # warm-up
    var t0 = perf_counter_ns()
    var out = enc.encode(inputs, batch)
    var s = Float64(perf_counter_ns() - t0) / 1e9
    print(
        t"{model}: {n} texts ({Float64(tokens) / Float64(n)} tokens avg),"
        t" batch {batch}: {s} s, {Float64(n) / s} texts/s; dim {len(out) // n}"
    )
