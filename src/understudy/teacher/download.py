"""Fetch the teacher's GGUF weights from a pinned Hugging Face commit and verify the SHA-256.

``python -m understudy.teacher.download``  (idempotent; skips the download if the file is
already present and its checksum matches)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from understudy.common.config import load_params, resolve
from understudy.common.lineage import file_sha256
from understudy.common.logs import get_logger
from understudy.data.ingest import download

log = get_logger(__name__)

HF_URL = "https://huggingface.co/{repo}/resolve/{revision}/{file}"


def model_path(teacher: dict) -> Path:
    """Local path of the pinned GGUF file."""
    return resolve(teacher["model_dir"]) / teacher["model_file"]


def ensure_model(teacher: dict) -> Path:
    """Download the pinned weights unless an identical copy is already on disk."""
    dest = model_path(teacher)
    if dest.is_file() and file_sha256(dest) == teacher["model_sha256"]:
        log.info("teacher weights present", extra={"path": str(dest)})
        return dest
    url = HF_URL.format(
        repo=teacher["model_repo"], revision=teacher["model_revision"], file=teacher["model_file"]
    )
    log.info("downloading teacher weights", extra={"url": url})
    return download(url, dest, teacher["model_sha256"], timeout=600.0)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(ensure_model(load_params()["teacher"]))


if __name__ == "__main__":
    main()
