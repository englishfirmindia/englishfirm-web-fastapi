import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from unittest.mock import patch

from services.trainer_review import _presign_in_place


def _fake_presign(url, expires_in=None):
    return f"PRESIGNED({url})"


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_presigns_image_url_when_present(_):
    out = _presign_in_place({"image_url": "https://b.s3.amazonaws.com/x.jpg"})
    assert out["image_url"] == "PRESIGNED(https://b.s3.amazonaws.com/x.jpg)"


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_presigns_audio_url(_):
    out = _presign_in_place({"audio_url": "https://b.s3.amazonaws.com/a.aac"})
    assert out["audio_url"] == "PRESIGNED(https://b.s3.amazonaws.com/a.aac)"


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_normalizes_s3_url_to_presigned_image_url(_):
    """Legacy describe_image shape: image stored under `s3_url`. Trainer
    payload must still expose a presigned `image_url` so the frontend
    Image.network() succeeds."""
    out = _presign_in_place({"s3_url": "https://b.s3.amazonaws.com/x.jpg"})
    assert out["image_url"] == "PRESIGNED(https://b.s3.amazonaws.com/x.jpg)"
    assert out["s3_url"] == "https://b.s3.amazonaws.com/x.jpg"


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_image_url_takes_precedence_when_both_present(_):
    out = _presign_in_place({
        "image_url": "https://b.s3.amazonaws.com/canonical.jpg",
        "s3_url": "https://b.s3.amazonaws.com/legacy.jpg",
    })
    assert out["image_url"] == "PRESIGNED(https://b.s3.amazonaws.com/canonical.jpg)"


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_handles_none_and_primitives(_):
    assert _presign_in_place(None) is None
    assert _presign_in_place("plain") == "plain"
    assert _presign_in_place(42) == 42


@patch("services.trainer_review.generate_presigned_url", side_effect=_fake_presign)
def test_recurses_into_nested_lists_and_dicts(_):
    out = _presign_in_place({
        "items": [
            {"image_url": "https://b.s3.amazonaws.com/1.jpg"},
            {"s3_url": "https://b.s3.amazonaws.com/2.jpg"},
        ]
    })
    assert out["items"][0]["image_url"] == "PRESIGNED(https://b.s3.amazonaws.com/1.jpg)"
    assert out["items"][1]["image_url"] == "PRESIGNED(https://b.s3.amazonaws.com/2.jpg)"


@patch("services.trainer_review.generate_presigned_url", side_effect=ValueError("bad url"))
def test_falls_back_to_raw_url_on_presign_failure(_):
    out = _presign_in_place({"s3_url": "not-an-s3-url"})
    assert out["image_url"] == "not-an-s3-url"
