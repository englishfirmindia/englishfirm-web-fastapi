"""
Canonical email normalisation for account lookup.

Prevents accidental duplicate accounts caused by trivial casing / alias
differences. Called by every auth entry point (signup, /auth/google,
/auth/apple, password reset lookup) BEFORE the DB lookup / insert.

Rules
-----
1. Lowercase the whole address (Gmail + most providers treat casing as
   case-insensitive; keeping mixed case created "Foo@Gmail.com" and
   "foo@gmail.com" as separate accounts).

2. Strip Gmail's "+alias" tag from the local part. `foo+work@gmail.com`
   and `foo@gmail.com` deliver to the same inbox — Gmail specifically
   documents `+` as sub-addressing. Applied only for @gmail.com and
   @googlemail.com because other providers (e.g. Outlook, Fastmail)
   don't guarantee the same semantics.

Intentionally NOT normalised
----------------------------
- Dots in Gmail local parts (`f.o.o@gmail.com` == `foo@gmail.com`
  server-side). Users type these deliberately for aesthetic reasons and
  stripping them silently would surprise them; leaving intact means at
  worst two accounts stay separate, never accidentally-merged.
- Whitespace inside the address (invalid email — let the validator reject).
- Unicode normalisation (NFC/NFKC) — Latin-alphabet emails dominate our
  user base; add later if we ever see IDN duplicates in production.

Safety
------
Returns the ORIGINAL string on any parsing failure (empty, no `@`,
multiple `@`). Callers should always run their own email-format
validator downstream; this helper is best-effort canonicalisation, not
input validation.
"""

_GMAIL_HOSTS = frozenset({"gmail.com", "googlemail.com"})


def normalise_email(raw: str) -> str:
    """Return the canonical form of `raw` for DB lookup. See module docstring."""
    if not raw:
        return raw
    e = raw.strip().lower()
    if e.count("@") != 1:
        return e  # let validator reject downstream
    local, host = e.split("@", 1)
    if host in _GMAIL_HOSTS and "+" in local:
        local = local.split("+", 1)[0]
    if not local:
        return e  # empty local part after strip → let validator reject
    return f"{local}@{host}"
