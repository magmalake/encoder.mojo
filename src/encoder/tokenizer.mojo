"""BERT's uncased WordPiece tokenizer, as HF `tokenizers` runs it for
bge-small and MiniLM: BertNormalizer (clean text, pad CJK, strip accents,
lowercase), BertPreTokenizer (split on whitespace, isolate punctuation),
then greedy longest-match WordPiece with a `##` continuation prefix.

Special tokens written in the input text (a literal "[SEP]") are tokenized
as text, not matched as added tokens.
"""

from std.collections import Dict

from ._unicode import (
    DECOMPOSITIONS,
    LOWERCASE,
    MARK_RANGES,
    OTHER_RANGES,
    PUNCT_RANGES,
)

comptime CLS = "[CLS]"
comptime SEP = "[SEP]"
comptime UNK = "[UNK]"
comptime MAX_CHARS_PER_WORD = 100


def _hex(s: StringSlice) raises -> UInt32:
    var v: UInt32 = 0
    for b in s.as_bytes():
        var c = UInt32(b)
        var d: UInt32
        if c >= 48 and c <= 57:  # 0-9
            d = c - 48
        elif c >= 97 and c <= 102:  # a-f
            d = c - 87
        else:
            raise Error("bad hex digit in Unicode table: ", s)
        v = v * 16 + d
    return v


struct _Ranges(Movable):
    """Sorted, disjoint [lo, hi] code point ranges."""

    var lo: List[UInt32]
    var hi: List[UInt32]

    def __init__(out self, packed: StaticString) raises:
        self.lo = List[UInt32]()
        self.hi = List[UInt32]()
        for item in packed.split(","):
            var parts = item.split("-")
            self.lo.append(_hex(parts[0]))
            self.hi.append(_hex(parts[1]))

    def contains(self, cp: UInt32) -> Bool:
        var a = 0
        var b = len(self.lo)
        while a < b:
            var m = (a + b) // 2
            if self.hi[m] < cp:
                a = m + 1
            else:
                b = m
        return a < len(self.lo) and self.lo[a] <= cp


struct _CodeMap(Movable):
    """Code point → sequence of code points, sorted by key."""

    var keys: List[UInt32]
    var starts: List[Int]  # len(keys) + 1 offsets into values
    var values: List[UInt32]

    def __init__(out self, packed: StaticString) raises:
        self.keys = List[UInt32]()
        self.starts = List[Int]()
        self.values = List[UInt32]()
        for entry in packed.split(";"):
            var kv = entry.split(":")
            self.keys.append(_hex(kv[0]))
            self.starts.append(len(self.values))
            for v in kv[1].split(" "):
                self.values.append(_hex(v))
        self.starts.append(len(self.values))

    def find(self, cp: UInt32) -> Int:
        """Index of `cp` in keys, or -1."""
        var a = 0
        var b = len(self.keys)
        while a < b:
            var m = (a + b) // 2
            if self.keys[m] < cp:
                a = m + 1
            else:
                b = m
        if a < len(self.keys) and self.keys[a] == cp:
            return a
        return -1

    def append_mapped(self, cp: UInt32, mut out: List[UInt32]):
        """Append cp's mapping to out, or cp itself if it has none."""
        var i = self.find(cp)
        if i < 0:
            out.append(cp)
            return
        for j in range(self.starts[i], self.starts[i + 1]):
            out.append(self.values[j])


def _is_whitespace(cp: UInt32) -> Bool:
    """Rust's char::is_whitespace (the White_Space property)."""
    return (
        (cp >= 9 and cp <= 13)
        or cp == 32
        or cp == 0x85
        or cp == 0xA0
        or cp == 0x1680
        or (cp >= 0x2000 and cp <= 0x200A)
        or cp == 0x2028
        or cp == 0x2029
        or cp == 0x202F
        or cp == 0x205F
        or cp == 0x3000
    )


def _is_chinese(cp: UInt32) -> Bool:
    return (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B920 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    )


def _is_ascii_punct(cp: UInt32) -> Bool:
    return (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    )


def _decode(text: String) -> List[UInt32]:
    var out = List[UInt32](capacity=text.byte_length())
    for c in text.codepoints():
        out.append(c.to_u32())
    return out^


def _encode(
    cps: List[UInt32], start: Int, end: Int, continuation: Bool
) -> String:
    var s = String("##") if continuation else String()
    for i in range(start, end):
        s.write(Codepoint(unsafe_unchecked_codepoint=cps[i]))
    return s^


def _special(vocab: Dict[String, Int], token: StaticString) raises -> Int:
    var id = vocab.get(String(token))
    if not id:
        raise Error("vocab has no ", token)
    return id.value()


struct WordPiece(Movable):
    """Uncased BERT WordPiece tokenizer loaded from a `vocab.txt`."""

    var vocab: Dict[String, Int]
    var cls_id: Int
    var sep_id: Int
    var unk_id: Int
    var _other: _Ranges
    var _punct: _Ranges
    var _mark: _Ranges
    var _decomp: _CodeMap
    var _lower: _CodeMap

    def __init__(out self, vocab_path: String) raises:
        """Load `vocab.txt`: one token per line, its id is the line number."""
        self.vocab = Dict[String, Int]()
        var text: String
        with open(vocab_path, "r") as f:
            text = f.read()
        var id = 0
        for line in text.split("\n"):
            if line.byte_length() > 0:
                self.vocab[String(line)] = id
            id += 1
        self.cls_id = _special(self.vocab, CLS)
        self.sep_id = _special(self.vocab, SEP)
        self.unk_id = _special(self.vocab, UNK)
        self._other = _Ranges(OTHER_RANGES)
        self._punct = _Ranges(PUNCT_RANGES)
        self._mark = _Ranges(MARK_RANGES)
        self._decomp = _CodeMap(DECOMPOSITIONS)
        self._lower = _CodeMap(LOWERCASE)

    def normalize(self, text: String) -> List[UInt32]:
        """BertNormalizer with clean_text, handle_chinese_chars,
        strip_accents and lowercase all on."""
        var cps = _decode(text)
        # clean_text, then pad CJK ideographs with spaces.
        var clean = List[UInt32](capacity=len(cps))
        for cp in cps:
            if cp == 0 or cp == 0xFFFD:
                continue
            if cp != 9 and cp != 10 and cp != 13 and self._other.contains(cp):
                continue
            if _is_whitespace(cp):
                clean.append(32)
            elif _is_chinese(cp):
                clean.append(32)
                clean.append(cp)
                clean.append(32)
            else:
                clean.append(cp)
        # strip_accents: NFD, then drop nonspacing marks.
        var nfd = List[UInt32](capacity=len(clean))
        for cp in clean:
            if cp >= 0xAC00 and cp < 0xD7A4:  # Hangul syllable
                var s = cp - 0xAC00
                nfd.append(0x1100 + s // 588)
                nfd.append(0x1161 + (s % 588) // 28)
                if s % 28 != 0:
                    nfd.append(0x11A7 + s % 28)
            else:
                self._decomp.append_mapped(cp, nfd)
        var out = List[UInt32](capacity=len(nfd))
        for cp in nfd:
            if not self._mark.contains(cp):
                self._lower.append_mapped(cp, out)
        return out^

    def is_punct(self, cp: UInt32) -> Bool:
        return _is_ascii_punct(cp) or self._punct.contains(cp)

    def _word_pieces(self, word: List[UInt32], mut ids: List[Int]):
        if len(word) > MAX_CHARS_PER_WORD:
            ids.append(self.unk_id)
            return
        var pieces = List[Int]()
        var start = 0
        while start < len(word):
            var end = len(word)
            var found = -1
            while end > start:
                var piece = _encode(word, start, end, start > 0)
                var id = self.vocab.get(piece)
                if id:
                    found = id.value()
                    break
                end -= 1
            if found < 0:
                ids.append(self.unk_id)
                return
            pieces.append(found)
            start = end
        ids.extend(pieces^)

    def tokenize(self, text: String) -> List[Int]:
        """Token ids without [CLS]/[SEP] and without truncation."""
        var cps = self.normalize(text)
        var ids = List[Int]()
        var word = List[UInt32]()
        for cp in cps:
            if _is_whitespace(cp):
                if len(word) > 0:
                    self._word_pieces(word, ids)
                    word.clear()
            elif self.is_punct(cp):
                if len(word) > 0:
                    self._word_pieces(word, ids)
                    word.clear()
                self._word_pieces([cp], ids)
            else:
                word.append(cp)
        if len(word) > 0:
            self._word_pieces(word, ids)
        return ids^

    def encode(self, text: String, max_tokens: Int = 512) -> List[Int]:
        """[CLS] + ids + [SEP], with ids truncated so the total fits in
        max_tokens."""
        var ids = self.tokenize(text)
        var keep = min(len(ids), max_tokens - 2)
        var out = List[Int](capacity=keep + 2)
        out.append(self.cls_id)
        for i in range(keep):
            out.append(ids[i])
        out.append(self.sep_id)
        return out^
