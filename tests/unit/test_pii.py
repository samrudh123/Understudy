import pytest

from understudy.common.pii import scrub, scrub_text

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("my email is asha.r@example.com thanks", "EMAIL"),
        ("call me on +91 98765 43210", "PHONE"),
        ("call 9876543210 please", "PHONE"),
        ("card 4111 1111 1111 1111 was charged", "CARD"),
        ("my roll number is DA24B025", "ID"),
        ("server 192.168.1.20 is down", "IP"),
        ("account 123456789 is locked", "ACCOUNT"),
    ],
)
def test_detects_pii(text: str, label: str) -> None:
    result = scrub(text)
    assert f"<{label}>" in result.text
    assert result.counts[label] == 1


@pytest.mark.parametrize(
    "text",
    [
        "transfer 500 dollars to savings",
        "what is on 2026-08-03",
        "set an alarm for 7:30 am",
        "is flight ai 302 on time",
        "what's 25 times 4",
    ],
)
def test_leaves_ordinary_questions_alone(text: str) -> None:
    result = scrub(text)
    assert result.text == text
    assert not result.found


def test_multiple_spans_and_wrapper() -> None:
    text = "mail a@b.co or b@c.org, phone 9876543210"
    assert scrub_text(text) == "mail <EMAIL> or <EMAIL>, phone <PHONE>"


def test_non_luhn_long_number_is_not_card() -> None:
    assert scrub_text("ref 1234 5678 9012 3456") != "ref <CARD>"
