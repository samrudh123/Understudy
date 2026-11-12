"""Real llama.cpp teacher (skipped when the GGUF weights are not downloaded)."""

import pytest

from understudy.common.config import load_params
from understudy.teacher.download import model_path
from understudy.teacher.prompts import DISTIL_SCHEMA, load_prompts

pytestmark = pytest.mark.integration

TEACHER = load_params()["teacher"]
needs_model = pytest.mark.skipif(
    not model_path(TEACHER).is_file(), reason="teacher weights not downloaded"
)


@pytest.fixture(scope="module")
def engine():
    from understudy.teacher.engine import LlamaEngine

    return LlamaEngine(model_path(TEACHER), n_ctx=TEACHER["n_ctx"], n_threads=8)


@needs_model
def test_real_teacher_streams_an_answer(engine) -> None:
    p = load_prompts(TEACHER["prompt_file"])
    pieces = list(engine.stream(p.answer("how do i report a lost card"), 64, 0.0))
    assert len(pieces) > 3
    assert "card" in "".join(pieces).lower()


@needs_model
def test_real_teacher_distil_json_matches_schema(engine) -> None:
    p = load_prompts(TEACHER["prompt_file"])
    out = engine.complete_json(
        p.distil(["how much gas is in my tank", "fuel level please", "is my tank full"]),
        DISTIL_SCHEMA,
        **p.settings("distil"),
    )
    assert set(DISTIL_SCHEMA["required"]) <= out.keys()
    assert out["answer"].strip() and len(out["paraphrases"]) >= 4
