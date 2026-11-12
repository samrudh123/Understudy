"""Loader for the raw CLINC150 files written by the ``ingest_dataset`` stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = "data/raw/clinc150"
DATA_FILE = "data_full.json"
DOMAINS_FILE = "domains.json"
OOS_LABEL = "oos"
IN_SCOPE_SPLITS = ("train", "val", "test")
OOS_SPLITS = ("oos_train", "oos_val", "oos_test")


@dataclass
class Clinc:
    """Raw CLINC150: ``splits[name] -> [(text, intent)]`` and ``domains[domain] -> [intent]``."""

    splits: dict[str, list[tuple[str, str]]]
    domains: dict[str, list[str]]

    @property
    def intent_to_domain(self) -> dict[str, str]:
        """Map every intent to its domain."""
        return {intent: d for d, intents in self.domains.items() for intent in intents}

    def utterances(self, splits: tuple[str, ...]) -> dict[str, list[str]]:
        """Group utterances from ``splits`` by intent label."""
        out: dict[str, list[str]] = {}
        for s in splits:
            for text, label in self.splits.get(s, []):
                out.setdefault(label, []).append(text)
        return out


def load_clinc(raw_dir: str | Path) -> Clinc:
    """Load ``data_full.json`` and ``domains.json`` from ``raw_dir``."""
    raw_dir = Path(raw_dir)
    with (raw_dir / DATA_FILE).open(encoding="utf-8") as fh:
        data = json.load(fh)
    with (raw_dir / DOMAINS_FILE).open(encoding="utf-8") as fh:
        domains = json.load(fh)
    splits = {name: [(row[0], row[1]) for row in rows] for name, rows in data.items()}
    return Clinc(splits=splits, domains=domains)
