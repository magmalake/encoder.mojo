# Token ids match HF `tokenizers` on every fixture input, for each model.
# Needs the model files in build/models/ (tools/make_fixtures.py).

from std.sys import argv

from encoder import WordPiece


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


def check_model(name: String) raises -> Int:
    var tok = WordPiece(String("build/models/") + name + "/vocab.txt")
    var texts = read_lines(String("tests/fixtures/") + name + "/texts.txt")
    var want = read_lines(String("tests/fixtures/") + name + "/ids.txt")
    if len(texts) != len(want):
        raise Error("fixture size mismatch: ", len(texts), " vs ", len(want))
    var bad = 0
    for i in range(len(texts)):
        var got = tok.encode(texts[i])
        var got_s = String()
        for j in range(len(got)):
            if j > 0:
                got_s += " "
            got_s += String(got[j])
        if got_s != want[i]:
            bad += 1
            if bad <= 5:
                print("MISMATCH", name, "#", i, repr(texts[i]))
                print("  want:", want[i])
                print("  got: ", got_s)
    print(name, ":", len(texts) - bad, "/", len(texts), "inputs match")
    return bad


def main() raises:
    var bad = 0
    bad += check_model("bge-small-en-v1.5")
    bad += check_model("all-MiniLM-L6-v2")
    if bad > 0:
        raise Error(bad, " inputs tokenized differently")
