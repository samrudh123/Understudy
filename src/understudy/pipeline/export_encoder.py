"""``export_encoder`` stage: MiniLM -> ONNX (pooling + L2 norm in the graph) -> dynamic 8-bit.

Per-channel UINT8 weights: on CLINC probes it keeps cosine ~0.99 to fp32 (per-tensor
INT8 dropped to ~0.96 and changed 20% of nearest neighbours).

Outputs ``models/encoder/`` (``encoder.onnx``, ``tokenizer.json``, ``encoder_meta.json``)
and ``reports/encoder_export.json``. The report covers fp32 vs INT8 size, cosine agreement
with the original PyTorch model and single-query latency on one thread.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.student.encoder import ENCODER_FILE, META_FILE, TOKENIZER_FILE, OnnxEncoder

log = get_logger(__name__)

OUT_DIR = "models/encoder"
REPORT = "reports/encoder_export.json"
PROBES = [
    "how do i reset my password",
    "what's my checking account balance",
    "book me a flight to delhi next friday",
    "how long should i boil an egg",
    "set a timer for ten minutes",
    "my card was stolen what do i do",
    "is there traffic on the way to work",
    "tell me a joke",
]


def _probe_texts() -> list[str]:
    """CLINC test questions (every 20th) if available, else a built-in list."""
    try:
        from understudy.data.clinc import load_clinc

        return [t for t, _ in load_clinc(resolve("data/raw/clinc150")).splits["test"]][::20]
    except FileNotFoundError:
        return PROBES


def _export_fp32(model_name: str, revision: str, max_len: int, dest: Path) -> tuple[Path, int]:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, revision=revision, device="cpu")
    hf = st[0].auto_model.eval()

    class Pooled(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            h = self.m(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    ids = torch.ones((2, 8), dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(
            Pooled(hf),
            (ids, torch.ones_like(ids), torch.zeros_like(ids)),
            str(dest),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    st.tokenizer.save_pretrained(str(dest.parent / "hf_tokenizer"))
    return dest, int(st.get_sentence_embedding_dimension())


def export(params: dict, out_dir: str | Path = OUT_DIR) -> dict:
    """Run the export and return the report."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    enc_cfg = params["encoder"]
    name, rev = params["mine"]["embedding_model"], params["mine"]["embedding_revision"]
    out = resolve(out_dir)
    with stage_run("export_encoder") as rep, tempfile.TemporaryDirectory() as tmp:
        fp32, dim = _export_fp32(name, rev, int(enc_cfg["max_seq_len"]), Path(tmp) / "fp32.onnx")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        quantize_dynamic(
            str(fp32), str(out / ENCODER_FILE), weight_type=QuantType.QUInt8, per_channel=True
        )
        shutil.copy(Path(tmp) / "hf_tokenizer" / "tokenizer.json", out / TOKENIZER_FILE)
        meta = {
            "source_model": name,
            "source_revision": rev,
            "dim": dim,
            "max_seq_len": int(enc_cfg["max_seq_len"]),
            "quantization": "dynamic-uint8-per-channel",
            "pad_id": 0,
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        # Quality: INT8 graph vs the original PyTorch model.
        from understudy.common.embeddings import SentenceTransformerEncoder

        probes = _probe_texts()
        ref = SentenceTransformerEncoder(name, revision=rev).encode(probes)
        enc = OnnxEncoder(out, threads=1)
        got = enc.encode(probes)
        agree = (ref * got).sum(1)
        nn_ref = np.argsort(-(ref @ ref.T), 1)[:, 1]
        nn_got = np.argsort(-(got @ got.T), 1)[:, 1]
        rank_ok = round(float(np.mean(nn_ref == nn_got)), 4)
        lat = []
        for q in PROBES * 5:
            t0 = time.perf_counter()
            enc.encode([q])
            lat.append((time.perf_counter() - t0) * 1000)
        report = {
            **meta,
            "fp32_mb": round(fp32.stat().st_size / 2**20, 2),
            "int8_mb": round((out / ENCODER_FILE).stat().st_size / 2**20, 2),
            "cosine_vs_fp32_min": round(float(agree.min()), 4),
            "cosine_vs_fp32_mean": round(float(agree.mean()), 4),
            "probe_sentences": len(probes),
            "nearest_neighbour_agreement": rank_ok,
            "latency_ms_p50_1thread": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95_1thread": round(float(np.percentile(lat, 95)), 2),
        }
        rep.rows_out = 1
        rep.extra.update({k: report[k] for k in ("int8_mb", "cosine_vs_fp32_min")})
    resolve(REPORT).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("encoder exported", extra=report)
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(export(load_params()), indent=2))


if __name__ == "__main__":
    main()
