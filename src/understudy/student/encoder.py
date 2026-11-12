"""ONNX sentence encoder used by the student at training *and* serving time.

The exported graph includes mean pooling and L2 normalisation, so it maps token ids
straight to unit-length sentence embeddings. Only ``onnxruntime``, ``tokenizers`` and
``numpy`` are needed at runtime, not torch, which keeps the edge container small.
Training embeds with this same INT8 graph, so there is no train/serve skew.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ENCODER_FILE = "encoder.onnx"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "encoder_meta.json"


class OnnxEncoder:
    """Tokenizer + ONNX Runtime session. ``encode`` returns L2-normalised float32 vectors."""

    def __init__(self, model_dir: str | Path, threads: int = 1, batch_size: int = 64) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        self.dim = int(self.meta["dim"])
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(d / TOKENIZER_FILE))
        self.tokenizer.enable_truncation(max_length=int(self.meta["max_seq_len"]))
        self.tokenizer.enable_padding(pad_id=int(self.meta.get("pad_id", 0)), pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(d / ENCODER_FILE), opts, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` (batched)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch(texts[i : i + self.batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
            }
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            out.append(self.session.run(None, feeds)[0])
        emb = np.concatenate(out).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms
