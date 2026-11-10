# The loader reads tensors with the values numpy sees (values printed by
# safetensors.numpy for bge-small-en-v1.5).

from std.testing import assert_almost_equal, assert_equal, assert_raises

from encoder.safetensors import SafeTensors


def main() raises:
    var st = SafeTensors("build/models/bge-small-en-v1.5/model.safetensors")
    assert_equal(len(st.tensors), 200)
    var w = st.f32("embeddings.word_embeddings.weight", [30522, 384])
    assert_equal(w[0], -0.0199737548828125)
    assert_equal(w[1], -0.0034008026123046875)
    assert_equal(w[len(w) - 1], 0.0518798828125)
    var b = st.f32("encoder.layer.11.output.dense.bias", [384])
    var total: Float64 = 0
    for v in b:
        total += Float64(v)
    assert_almost_equal(total, 0.9493446350097656, atol=1e-9)
    with assert_raises(contains="shape"):
        _ = st.f32("encoder.layer.11.output.dense.bias", [383])
    with assert_raises(contains="no tensor"):
        _ = st.info("nope")
    print("safetensors: ok")
