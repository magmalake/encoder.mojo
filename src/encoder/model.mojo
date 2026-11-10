"""BERT encoder forward pass on the CPU, float32, as HF `BertModel` runs it
in eval mode, plus sentence-transformers pooling and L2 normalization.

Sequences are packed: the token rows of a batch sit back to back in one
[tokens, hidden] matrix, so linear layers are one max-core matmul over the
whole batch and attention only ever looks inside one sequence. Nothing is
padded.
"""

from std.math import erf, exp, sqrt
from std.sys.info import simd_width_of

from max.algorithm import parallelize

from layout import TileTensor, row_major
from linalg.matmul import matmul
from nn.activations import gelu

from .json import JsonCursor
from .safetensors import SafeTensors


@fieldwise_init
struct BertConfig(Copyable, Movable, Writable):
    var vocab: Int
    var hidden: Int
    var layers: Int
    var heads: Int
    var ffn: Int
    var max_positions: Int
    var eps: Float32
    var mean_pooling: Bool  # else the [CLS] token

    @staticmethod
    def load(dir: String) raises -> Self:
        var cfg = BertConfig(0, 0, 0, 0, 0, 0, 1e-12, False)
        var act = String()
        var kind = String("absolute")
        with open(dir + "/config.json", "r") as f:
            var j = JsonCursor(f.read())
            j.expect(ord("{"))
            var first = True
            while True:
                var key = j.next_key(first)
                if not key:
                    break
                var k = key.value()
                if k == "vocab_size":
                    cfg.vocab = j.int()
                elif k == "hidden_size":
                    cfg.hidden = j.int()
                elif k == "num_hidden_layers":
                    cfg.layers = j.int()
                elif k == "num_attention_heads":
                    cfg.heads = j.int()
                elif k == "intermediate_size":
                    cfg.ffn = j.int()
                elif k == "max_position_embeddings":
                    cfg.max_positions = j.int()
                elif k == "layer_norm_eps":
                    cfg.eps = Float32(j.number())
                elif k == "hidden_act":
                    act = j.string()
                elif k == "position_embedding_type":
                    kind = j.string()
                else:
                    j.skip_value()
        if act != "gelu":
            raise Error("hidden_act ", act, " not supported (only gelu)")
        if kind != "absolute":
            raise Error("position_embedding_type ", kind, " not supported")
        if cfg.hidden == 0 or cfg.heads == 0 or cfg.hidden % cfg.heads != 0:
            raise Error("config.json: bad hidden_size/num_attention_heads")
        if cfg.hidden % W != 0:
            raise Error("hidden_size must be a multiple of ", W)
        # sentence-transformers pooling; [CLS] if the model has none.
        try:
            with open(dir + "/1_Pooling/config.json", "r") as f:
                var j = JsonCursor(f.read())
                j.expect(ord("{"))
                var first = True
                while True:
                    var key = j.next_key(first)
                    if not key:
                        break
                    if key.value() == "pooling_mode_mean_tokens":
                        cfg.mean_pooling = j.bool()
                    else:
                        j.skip_value()
        except:
            pass
        return cfg^


struct Layer(Copyable, Movable):
    var qkv_w: List[Float32]  # [3H, H]: query, key, value stacked
    var qkv_b: List[Float32]
    var out_w: List[Float32]  # [H, H]
    var out_b: List[Float32]
    var ln1_g: List[Float32]
    var ln1_b: List[Float32]
    var ff1_w: List[Float32]  # [F, H]
    var ff1_b: List[Float32]
    var ff2_w: List[Float32]  # [H, F]
    var ff2_b: List[Float32]
    var ln2_g: List[Float32]
    var ln2_b: List[Float32]

    def __init__(out self, st: SafeTensors, i: Int, h: Int, f: Int) raises:
        var p = String("encoder.layer.", i, ".")
        self.qkv_w = st.f32(p + "attention.self.query.weight", [h, h])
        self.qkv_w.extend(st.f32(p + "attention.self.key.weight", [h, h]))
        self.qkv_w.extend(st.f32(p + "attention.self.value.weight", [h, h]))
        self.qkv_b = st.f32(p + "attention.self.query.bias", [h])
        self.qkv_b.extend(st.f32(p + "attention.self.key.bias", [h]))
        self.qkv_b.extend(st.f32(p + "attention.self.value.bias", [h]))
        self.out_w = st.f32(p + "attention.output.dense.weight", [h, h])
        self.out_b = st.f32(p + "attention.output.dense.bias", [h])
        self.ln1_g = st.f32(p + "attention.output.LayerNorm.weight", [h])
        self.ln1_b = st.f32(p + "attention.output.LayerNorm.bias", [h])
        self.ff1_w = st.f32(p + "intermediate.dense.weight", [f, h])
        self.ff1_b = st.f32(p + "intermediate.dense.bias", [f])
        self.ff2_w = st.f32(p + "output.dense.weight", [h, f])
        self.ff2_b = st.f32(p + "output.dense.bias", [h])
        self.ln2_g = st.f32(p + "output.LayerNorm.weight", [h])
        self.ln2_b = st.f32(p + "output.LayerNorm.bias", [h])


comptime W = simd_width_of[DType.float32]()
comptime ROWS_PER_TASK = 16


def _linear[
    with_gelu: Bool = False
](
    x: List[Float32], rows: Int, k: Int, w: List[Float32], b: List[Float32]
) raises -> List[Float32]:
    """x[rows, k] @ w[n, k]^T + b, with n = len(b), then exact GELU if
    `with_gelu`."""
    var n = len(b)
    var y = List(length=rows * n, fill=Float32(0))
    var c = TileTensor(y, row_major(rows, n))
    matmul[transpose_b=True](
        c, TileTensor(x, row_major(rows, k)), TileTensor(w, row_major(n, k))
    )
    var yp = y.unsafe_ptr()
    var bp = b.unsafe_ptr()

    def task(t: Int) {imm yp, imm bp, imm rows, imm n}:
        for r in range(t * ROWS_PER_TASK, min((t + 1) * ROWS_PER_TASK, rows)):
            var o = r * n
            var j = 0
            while j + W <= n:
                var v = yp.unsafe_load[width=W](o + j) + bp.unsafe_load[
                    width=W
                ](j)
                comptime if with_gelu:
                    v = gelu(v)
                yp.unsafe_store(o + j, v)
                j += W
            while j < n:
                var v = yp[unsafe_offset=o + j] + bp[unsafe_offset=j]
                comptime if with_gelu:
                    v = gelu(v)
                yp[unsafe_offset=o + j] = v
                j += 1

    parallelize(task, (rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK)
    return y^


def _add_layer_norm(
    mut x: List[Float32],
    y: List[Float32],
    rows: Int,
    h: Int,
    g: List[Float32],
    b: List[Float32],
    eps: Float32,
):
    """x = LayerNorm(x + y), row by row. h must be a multiple of W."""
    var xp = x.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var gp = g.unsafe_ptr()
    var bp = b.unsafe_ptr()

    def task(t: Int) {imm xp, imm yp, imm gp, imm bp, imm rows, imm h, imm eps}:
        for r in range(t * ROWS_PER_TASK, min((t + 1) * ROWS_PER_TASK, rows)):
            var o = r * h
            var sum = SIMD[DType.float32, W](0)
            for j in range(0, h, W):
                var v = xp.unsafe_load[width=W](o + j) + yp.unsafe_load[
                    width=W
                ](o + j)
                xp.unsafe_store(o + j, v)
                sum += v
            var mean = sum.reduce_add() / Float32(h)
            var sq = SIMD[DType.float32, W](0)
            for j in range(0, h, W):
                var d = xp.unsafe_load[width=W](o + j) - mean
                sq += d * d
            var inv = 1 / sqrt(sq.reduce_add() / Float32(h) + eps)
            for j in range(0, h, W):
                var v = (xp.unsafe_load[width=W](o + j) - mean) * inv
                xp.unsafe_store(
                    o + j,
                    v * gp.unsafe_load[width=W](j) + bp.unsafe_load[width=W](j),
                )

    parallelize(task, (rows + ROWS_PER_TASK - 1) // ROWS_PER_TASK)


struct Bert(Movable):
    var cfg: BertConfig
    var word: List[Float32]  # [vocab, H]
    var position: List[Float32]  # [max_positions, H]
    var token_type0: List[Float32]  # [H], type 0 (single sentences)
    var emb_ln_g: List[Float32]
    var emb_ln_b: List[Float32]
    var layers: List[Layer]

    def __init__(out self, dir: String) raises:
        self.cfg = BertConfig.load(dir)
        var st = SafeTensors(dir + "/model.safetensors")
        var h = self.cfg.hidden
        self.word = st.f32(
            "embeddings.word_embeddings.weight", [self.cfg.vocab, h]
        )
        self.position = st.f32(
            "embeddings.position_embeddings.weight", [self.cfg.max_positions, h]
        )
        var types = st.info(
            "embeddings.token_type_embeddings.weight"
        ).shape.copy()
        var tt = st.f32("embeddings.token_type_embeddings.weight", types)
        self.token_type0 = List[Float32](capacity=h)
        for j in range(h):
            self.token_type0.append(tt[j])
        self.emb_ln_g = st.f32("embeddings.LayerNorm.weight", [h])
        self.emb_ln_b = st.f32("embeddings.LayerNorm.bias", [h])
        self.layers = List[Layer]()
        for i in range(self.cfg.layers):
            self.layers.append(Layer(st, i, h, self.cfg.ffn))

    def _attention(
        self, qkv: List[Float32], starts: List[Int], total: Int
    ) raises -> List[Float32]:
        """Self-attention of each sequence over its own tokens only, one
        parallel task per (sequence, head). Sequences here are short (tens
        of tokens), where a direct SIMD loop beats a matmul call per head."""
        var h = self.cfg.hidden
        var heads = self.cfg.heads
        var d = h // heads
        var scale = 1 / sqrt(Float32(d))
        var ctx = List(length=total * h, fill=Float32(0))
        var qp = qkv.unsafe_ptr()
        var cp = ctx.unsafe_ptr()
        var sp = starts.unsafe_ptr()

        def work(
            t: Int,
        ) {imm qp, imm cp, imm sp, imm h, imm heads, imm d, imm scale}:
            var s = t // heads
            var hd = t % heads
            var t0 = sp[unsafe_offset=s]
            var n = sp[unsafe_offset=s + 1] - t0
            var scores = List(length=n, fill=Float32(0))
            var acc = List(length=d, fill=Float32(0))
            var ap = acc.unsafe_ptr()
            for i in range(n):
                var q = (t0 + i) * 3 * h + hd * d
                var mx = Float32.MIN
                for j in range(n):
                    var k = (t0 + j) * 3 * h + h + hd * d
                    var dot = SIMD[DType.float32, W](0)
                    var e = 0
                    while e + W <= d:
                        dot += qp.unsafe_load[width=W](q + e) * qp.unsafe_load[
                            width=W
                        ](k + e)
                        e += W
                    var sc = dot.reduce_add()
                    while e < d:
                        sc += qp[unsafe_offset=q + e] * qp[unsafe_offset=k + e]
                        e += 1
                    sc *= scale
                    scores[j] = sc
                    mx = max(mx, sc)
                var sum: Float32 = 0
                for j in range(n):
                    var ex = exp(scores[j] - mx)
                    scores[j] = ex
                    sum += ex
                for e in range(d):
                    acc[e] = 0
                for j in range(n):
                    var w = scores[j] / sum
                    var v = (t0 + j) * 3 * h + 2 * h + hd * d
                    var e = 0
                    while e + W <= d:
                        ap.unsafe_store(
                            e,
                            ap.unsafe_load[width=W](e)
                            + w * qp.unsafe_load[width=W](v + e),
                        )
                        e += W
                    while e < d:
                        acc[e] += w * qp[unsafe_offset=v + e]
                        e += 1
                var o = (t0 + i) * h + hd * d
                for e in range(d):
                    cp[unsafe_offset=o + e] = acc[e]

        parallelize(work, (len(starts) - 1) * heads)
        return ctx^

    def forward(self, ids: List[List[Int]]) raises -> List[Float32]:
        """Pooled, L2-normalized embeddings, [len(ids), hidden] flattened."""
        var h = self.cfg.hidden
        var starts = List[Int](capacity=len(ids) + 1)
        var total = 0
        for seq in ids:
            if len(seq) > self.cfg.max_positions:
                raise Error("sequence longer than max_position_embeddings")
            starts.append(total)
            total += len(seq)
        starts.append(total)

        var x = List(length=total * h, fill=Float32(0))
        var zero = List(length=total * h, fill=Float32(0))
        for s in range(len(ids)):
            for i in range(len(ids[s])):
                var o = (starts[s] + i) * h
                var w = ids[s][i] * h
                var p = i * h
                for j in range(h):
                    x[o + j] = (
                        self.word[w + j]
                        + self.position[p + j]
                        + self.token_type0[j]
                    )
        _add_layer_norm(
            x, zero, total, h, self.emb_ln_g, self.emb_ln_b, self.cfg.eps
        )

        for ref layer in self.layers:
            var qkv = _linear(x, total, h, layer.qkv_w, layer.qkv_b)
            var ctx = self._attention(qkv, starts, total)
            var attn = _linear(ctx, total, h, layer.out_w, layer.out_b)
            _add_layer_norm(
                x, attn, total, h, layer.ln1_g, layer.ln1_b, self.cfg.eps
            )
            var mid = _linear[with_gelu=True](
                x, total, h, layer.ff1_w, layer.ff1_b
            )
            var ff = _linear(mid, total, self.cfg.ffn, layer.ff2_w, layer.ff2_b)
            _add_layer_norm(
                x, ff, total, h, layer.ln2_g, layer.ln2_b, self.cfg.eps
            )

        var out = List(length=len(ids) * h, fill=Float32(0))
        for s in range(len(ids)):
            var n = starts[s + 1] - starts[s]
            if self.cfg.mean_pooling:
                for i in range(n):
                    var o = (starts[s] + i) * h
                    for j in range(h):
                        out[s * h + j] += x[o + j]
                for j in range(h):
                    out[s * h + j] /= Float32(n)
            else:
                var o = starts[s] * h
                for j in range(h):
                    out[s * h + j] = x[o + j]
            var norm: Float32 = 0
            for j in range(h):
                norm += out[s * h + j] * out[s * h + j]
            var inv = 1 / max(sqrt(norm), 1e-12)
            for j in range(h):
                out[s * h + j] *= inv
        return out^
