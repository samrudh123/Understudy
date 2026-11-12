"""Data contracts shared across services and pipeline stages.

All on-disk records (query logs, feedback, mined candidates, approvals) are
validated against these models. ``validate`` stops the pipeline when too many
rows break the contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QUERYLOG_SCHEMA_VERSION = 1


class Route(StrEnum):
    """Which tier produced the answer for a question."""

    STUDENT = "student"  # instant answer from the edge student
    TEACHER = "teacher"  # fallback answered by the server-side teacher
    UNAVAILABLE = "unavailable"  # fallback needed but the server could not answer
    REJECTED = "rejected"  # blocked by input validation / guardrails


class QueryLogRecord(BaseModel):
    """One question and its routing decision. Append-only and PII-scrubbed."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = QUERYLOG_SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=64)
    ts: datetime
    text: str = Field(min_length=1, max_length=2000)
    route: Route
    matched_faq_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    student_version: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    teacher_ttft_ms: float | None = Field(default=None, ge=0.0)
    # "test" = load tests / drills: counted in metrics, excluded from mining and training.
    source: Literal["live", "simulated", "test"] = "live"
    # Simulation-only ground truth. Used for offline evaluation, never for routing.
    sim_week: int | None = Field(default=None, ge=0)
    sim_label: str | None = None
    sim_domain: str | None = None

    @field_validator("ts")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("ts must be timezone-aware")
        return v


class FeedbackRecord(BaseModel):
    """A helpful / not-helpful vote on a previously logged answer."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    ts: datetime
    helpful: bool
    source: Literal["live", "simulated"] = "live"


class Candidate(BaseModel):
    """A cluster of fallback questions that has become frequent (output of ``mine_frequent``)."""

    candidate_id: str
    status: Literal["new", "known_but_missed"]
    representative_question: str
    support: int = Field(ge=1)  # number of fallback queries in the cluster
    n_unique: int = Field(ge=1)  # distinct normalised questions
    share_of_fallbacks: float = Field(ge=0.0, le=1.0)
    cohesion: float  # mean cosine similarity to the centroid
    examples: list[str]
    # Evaluation only (simulation ground truth): how pure the cluster is.
    dominant_sim_label: str | None = None
    purity: float | None = None


class ApprovalRecord(BaseModel):
    """An operator's decision on a teacher draft. One file per FAQ under ``data/approvals/``."""

    model_config = ConfigDict(extra="forbid")

    faq_id: str
    candidate_id: str
    decision: Literal["approved", "rejected"]
    question: str
    answer: str
    paraphrases: list[str] = Field(default_factory=list)
    reviewer: str
    decided_at: datetime
    teacher_model: str
    prompt_version: str
    log_examples: list[str] = Field(default_factory=list)  # real user phrasings from the cluster
    merged_candidates: list[str] = Field(default_factory=list)  # later clusters merged into it
    support: int | None = None
    source_intent: str | None = None  # evaluation only


class Draft(BaseModel):
    """A teacher-written FAQ entry for one mined candidate, waiting for human review."""

    candidate_id: str
    question: str
    answer: str
    paraphrases: list[str]
    examples: list[str]  # the real user questions the teacher saw
    support: int
    share_of_fallbacks: float
    cohesion: float
    teacher_model: str
    prompt_version: str
    generated_at: datetime
    latency_s: float
    cached: bool = False
    dominant_sim_label: str | None = None  # evaluation only
    purity: float | None = None  # evaluation only
