"""Regression tests for services.email_normaliser.normalise_email
(shipped 2026-09-05 alongside Google + Apple social login on mobile).

Prevents accidental account duplication caused by trivial casing / alias
differences. See the module docstring for the rule set. These tests pin:

  1. Casing normalised (uppercase → lowercase)
  2. Gmail + Googlemail `+alias` stripped (Gmail's documented
     sub-addressing convention)
  3. `+alias` NOT stripped on other providers (Outlook, Fastmail,
     etc. don't guarantee the same semantics)
  4. Whitespace trimmed at ends
  5. Empty / malformed input passes through unchanged so downstream
     validation still fires (the normaliser is best-effort, not a
     validator)
  6. Dots in Gmail local parts NOT stripped (users deliberately style
     them; silent stripping would surprise)
"""
from services.email_normaliser import normalise_email


def test_lowercases_the_whole_address():
    assert normalise_email("Foo@Gmail.com") == "foo@gmail.com"
    assert normalise_email("USER@EXAMPLE.COM") == "user@example.com"


def test_strips_gmail_plus_alias():
    assert normalise_email("foo+work@gmail.com") == "foo@gmail.com"
    assert normalise_email("foo+anything.at.all@gmail.com") == "foo@gmail.com"


def test_strips_googlemail_plus_alias():
    """@googlemail.com is Gmail's original UK/DE domain — same inbox,
    same +alias semantics. Historical addresses still exist in the wild."""
    assert normalise_email("user+xxx@GOOGLEMAIL.COM") == "user@googlemail.com"


def test_does_not_strip_plus_alias_on_non_gmail():
    """Outlook / Fastmail / iCloud etc. don't guarantee +alias delivers
    to the same mailbox. Leaving intact means at worst two accounts stay
    separate — never accidentally-merged."""
    assert normalise_email("foo+xxx@outlook.com") == "foo+xxx@outlook.com"
    assert normalise_email("foo+xxx@fastmail.com") == "foo+xxx@fastmail.com"
    assert normalise_email("foo+xxx@icloud.com") == "foo+xxx@icloud.com"


def test_trims_leading_and_trailing_whitespace():
    assert normalise_email("  foo@bar.com  ") == "foo@bar.com"
    assert normalise_email("\tfoo@bar.com\n") == "foo@bar.com"


def test_empty_input_passes_through():
    """Best-effort — return empty rather than raise. Downstream validator
    handles the rejection."""
    assert normalise_email("") == ""


def test_malformed_no_at_sign_passes_through():
    """Downstream validator handles rejection. Normaliser doesn't
    inspect validity beyond splitting on '@'."""
    assert normalise_email("nope") == "nope"


def test_malformed_multiple_at_signs_passes_through():
    """Multiple @ is invalid but ambiguous which to split on. Punt to
    the validator."""
    got = normalise_email("a@b@c.com")
    assert got == "a@b@c.com".lower()


def test_dots_in_gmail_local_part_preserved():
    """Gmail treats `f.o.o@gmail.com` == `foo@gmail.com` server-side,
    but users type dots deliberately for legibility. Silently stripping
    would surprise them. Two accounts staying separate is the safer
    failure mode than accidentally-merging strangers who happen to
    share dots."""
    assert normalise_email("f.o.o@gmail.com") == "f.o.o@gmail.com"


def test_gmail_alias_then_dots_preserved_after_strip():
    """Combination: strip +alias, keep dots."""
    assert normalise_email("f.o.o+work@gmail.com") == "f.o.o@gmail.com"


def test_empty_local_after_alias_strip_falls_back():
    """`+work@gmail.com` has empty local part after strip — that's not a
    real Gmail address anyway. Pass through the lowercased original so
    the validator rejects downstream instead of returning `@gmail.com`."""
    got = normalise_email("+work@gmail.com")
    assert "@gmail.com" in got  # some form of lowercased input returned
    assert got != "@gmail.com"  # NOT the empty-local-part garbage


def test_idempotent_on_already_canonical_input():
    """Running the normaliser on its own output must be a no-op —
    otherwise repeated calls could drift."""
    canonical = "foo@gmail.com"
    assert normalise_email(canonical) == canonical
    assert normalise_email(normalise_email("Foo+xxx@Gmail.com")) == canonical
