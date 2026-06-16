"""Regression tests for the per-item `download_only` flag on the learning-
resources catalogue (shipped 2026-06-17 after the Class Videos Links.docx
gview-rendering bug).

Background: gview rendered Class Videos Links.docx with page 1 mostly blank
because of empty-paragraph spacers. Office Online would render it fine, but
the file is a hyperlink index by nature — students want the YouTube URLs,
not a Word preview. The cleanest fix: bypass both viewers for this single
file and offer the user a direct download. A per-item flag in the catalogue
gives us an extensibility point for any future docx that hits the same
class of rendering bug.

These tests pin:
  1. Class Videos Links carries `download_only: True` in the API response
  2. Every other item in the catalogue does NOT carry the flag (kept off
     the response to keep payloads compact)
  3. The flag is only emitted when truthy — defaulting to absent means
     older frontends that don't know about the field treat all items as
     previewable (the historical default).
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from unittest.mock import patch


def test_class_videos_links_has_download_only_flag():
    """The specific file the bug hit must surface the flag."""
    # Stub out S3 presigning — _url() hits boto3 by default.
    with patch("services.resources_service._url", return_value="https://example.com/x"):
        from services.resources_service import get_resources
        # Force re-import so the patched _url is used.
        import importlib, services.resources_service as svc
        importlib.reload(svc)
        with patch.object(svc, "_url", return_value="https://example.com/x"):
            cats = svc.get_resources()
    target = None
    for cat in cats:
        for sub in cat["subsections"]:
            for item in sub["items"]:
                if item["title"] == "Class Videos Links":
                    target = item
                    break
    assert target is not None, "Class Videos Links must exist in the catalogue"
    assert target.get("download_only") is True, (
        f"Class Videos Links must carry download_only=True; got {target}"
    )


def test_other_items_do_NOT_carry_download_only():
    """The flag is per-item — every other resource must come back without
    the key so the frontend continues to use the preview modal for them."""
    import importlib, services.resources_service as svc
    importlib.reload(svc)
    with patch.object(svc, "_url", return_value="https://example.com/x"):
        cats = svc.get_resources()
    offenders = []
    for cat in cats:
        for sub in cat["subsections"]:
            for item in sub["items"]:
                if item["title"] == "Class Videos Links":
                    continue
                # Either absent (preferred — compact) OR explicitly False
                if item.get("download_only"):
                    offenders.append(item["title"])
    assert offenders == [], f"Unexpected download_only flag on: {offenders}"


def test_download_only_absent_not_false_for_compactness():
    """The serializer emits the flag only when truthy — items without it
    should not have the key in the response at all (saves bytes for the
    ~56 items that don't need it)."""
    import importlib, services.resources_service as svc
    importlib.reload(svc)
    with patch.object(svc, "_url", return_value="https://example.com/x"):
        cats = svc.get_resources()
    sample_item = None
    for cat in cats:
        for sub in cat["subsections"]:
            for item in sub["items"]:
                if item["title"] != "Class Videos Links":
                    sample_item = item
                    break
            if sample_item: break
        if sample_item: break
    assert sample_item is not None
    assert "download_only" not in sample_item, (
        f"Non-flagged items should not even have the key. Got: {sample_item.keys()}"
    )


def test_response_shape_otherwise_unchanged():
    """Existing fields (title, url, type) must continue to be present on
    every item — the new flag is additive, never replacing anything."""
    import importlib, services.resources_service as svc
    importlib.reload(svc)
    with patch.object(svc, "_url", return_value="https://example.com/x"):
        cats = svc.get_resources()
    for cat in cats:
        assert {"id", "label", "icon", "color", "subsections"} <= set(cat.keys())
        for sub in cat["subsections"]:
            assert {"id", "label", "color", "items"} <= set(sub.keys())
            for item in sub["items"]:
                # The three legacy fields must always be present.
                assert {"title", "url", "type"} <= set(item.keys()), (
                    f"Missing legacy field on {item}"
                )
