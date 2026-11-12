import numpy as np
import pytest

from understudy.common.embeddings import HashingEncoder, get_encoder
from understudy.common.text import normalize

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  When is   Assignment 4 DUE? ", "when is assignment 4 due?"),
        ("what’s the “wifi” password", 'what\'s the "wifi" password'),
        ("tab\tand\nnewline", "tab and newline"),
        ("ｆｕｌｌｗｉｄｔｈ", "fullwidth"),
        ("", ""),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    s = "  Héllo   WORLD’s "
    assert normalize(normalize(s)) == normalize(s)


def test_hashing_encoder_normalised_and_deterministic() -> None:
    enc = HashingEncoder(128)
    a = enc.encode(["book a flight to delhi", "reset my password"])
    b = HashingEncoder(128).encode(["book a flight to delhi", "reset my password"])
    assert a.shape == (2, 128)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a, b)


def test_hashing_encoder_similarity_is_meaningful() -> None:
    enc = get_encoder("hash:512")
    v = enc.encode(
        ["book a flight to delhi", "please book a flight to mumbai", "reset my password"]
    )
    assert v[0] @ v[1] > v[0] @ v[2]


def test_get_encoder_is_cached() -> None:
    assert get_encoder("hash:64") is get_encoder("hash:64")
