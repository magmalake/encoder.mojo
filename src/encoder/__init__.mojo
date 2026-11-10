"""BERT-family text encoders on max-core: text in, normalized sentence
embeddings out, with no Python at runtime."""

from .model import Bert, BertConfig
from .tokenizer import WordPiece


struct Encoder(Movable):
    """A tokenizer and model loaded from a sentence-transformers directory
    (config.json, model.safetensors, vocab.txt, 1_Pooling/config.json)."""

    var tokenizer: WordPiece
    var model: Bert
    var max_tokens: Int

    def __init__(out self, dir: String, max_tokens: Int = 512) raises:
        self.tokenizer = WordPiece(dir + "/vocab.txt")
        self.model = Bert(dir)
        self.max_tokens = min(max_tokens, self.model.cfg.max_positions)

    def dim(self) -> Int:
        return self.model.cfg.hidden

    def encode(
        self, texts: List[String], batch: Int = 32
    ) raises -> List[Float32]:
        """Embeddings for texts, row i for texts[i], as [len(texts), dim]
        flattened. Each batch of up to `batch` texts runs as one forward."""
        var out = List[Float32](capacity=len(texts) * self.dim())
        var i = 0
        while i < len(texts):
            var ids = List[List[Int]]()
            for j in range(i, min(i + batch, len(texts))):
                ids.append(self.tokenizer.encode(texts[j], self.max_tokens))
            out.extend(self.model.forward(ids))
            i += batch
        return out^
