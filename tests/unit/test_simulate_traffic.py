import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from understudy.common.schemas import ApprovalRecord, QueryLogRecord, Route
from understudy.data.clinc import load_clinc
from understudy.data.simulate_traffic import (
    domains_active,
    generate_week,
    parse_weeks,
    plan_week,
    simulate_offline,
)

pytestmark = pytest.mark.unit


def test_parse_weeks() -> None:
    assert parse_weeks("3") == [3]
    assert parse_weeks("0-2") == [0, 1, 2]
    assert parse_weeks("4,0-1") == [0, 1, 4]
    with pytest.raises(ValueError):
        parse_weeks("")


def test_domain_schedule_and_surge(params: dict) -> None:
    sim = params["simulation"]
    assert domains_active(sim, 1) == (["d1", "d2"], [])
    assert domains_active(sim, 2) == (["d1", "d2", "d3"], ["d3"])
    assert domains_active(sim, 3) == (["d1", "d2", "d3"], ["d3"])
    assert domains_active(sim, 4) == (["d1", "d2", "d3"], [])


def test_plan_weights_sum_to_one_and_surge_share(tiny_raw: Path, params: dict) -> None:
    clinc = load_clinc(tiny_raw)
    w1 = plan_week(clinc, params["simulation"], params["seed"], 1)
    assert pytest.approx(sum(w1.weights.values())) == 1.0
    assert not {"recipe", "timer", "alarm"} & w1.weights.keys()
    w2 = plan_week(clinc, params["simulation"], params["seed"], 2)
    d3_share = sum(w2.weights[i] for i in ("recipe", "timer", "alarm"))
    assert d3_share == pytest.approx(params["simulation"]["new_domain_share"])


def test_unknown_domain_rejected(tiny_raw: Path, params: dict) -> None:
    params["simulation"]["initial_domains"] = ["nope"]
    with pytest.raises(ValueError):
        plan_week(load_clinc(tiny_raw), params["simulation"], params["seed"], 0)


def test_generate_week_is_deterministic_and_time_ordered(tiny_raw: Path, params: dict) -> None:
    clinc = load_clinc(tiny_raw)
    a = generate_week(clinc, params["simulation"], params["seed"], 1, {})
    b = generate_week(clinc, params["simulation"], params["seed"], 1, {})
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    assert len(a) == params["simulation"]["queries_per_week"]
    assert all(x.ts <= y.ts for x, y in zip(a, a[1:], strict=False))
    assert all(r.route == Route.TEACHER for r in a)  # nothing approved yet -> all fallback


def test_never_replays_test_utterances(tiny_raw: Path, params: dict) -> None:
    clinc = load_clinc(tiny_raw)
    test_texts = {t for t, _ in clinc.splits["test"]} | {t for t, _ in clinc.splits["oos_test"]}
    recs = generate_week(clinc, params["simulation"], params["seed"], 2, {})
    assert not any(r.text in test_texts for r in recs)


def test_oracle_routing_uses_approvals(tiny_raw: Path, params: dict, root: Path) -> None:
    approvals = root / "data/approvals"
    approvals.mkdir(parents=True)
    rec = ApprovalRecord(
        faq_id="faq-1",
        candidate_id="c1",
        decision="approved",
        question="reset password",
        answer="Use the portal.",
        reviewer="op",
        decided_at=datetime.now(UTC),
        teacher_model="t",
        prompt_version="v1",
        source_intent="reset_password",
    )
    (approvals / "faq-1.json").write_text(rec.model_dump_json())
    summary = simulate_offline(params, [0, 1], tiny_raw, root / "data/querylogs", approvals)
    assert set(summary) == {0, 1}
    lines = (root / "data/querylogs/sim-week-01.jsonl").read_text().splitlines()
    recs = [QueryLogRecord.model_validate_json(x) for x in lines]
    student = [r for r in recs if r.route == Route.STUDENT]
    assert student and all(
        r.sim_label == "reset_password" and r.matched_faq_id == "faq-1" for r in student
    )
    assert 0 < summary[1]["fallback_ratio"] < 1
    # Re-running a week overwrites rather than appends.
    simulate_offline(params, [1], tiny_raw, root / "data/querylogs", approvals)
    assert len((root / "data/querylogs/sim-week-01.jsonl").read_text().splitlines()) == len(lines)
    # PII was injected and scrubbed before hitting disk.
    assert "@example.com" not in "\n".join(lines)
    assert json.dumps(summary[1]["pii_scrubbed"]) != "{}"
