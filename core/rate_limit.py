"""Shared rate limiter — keyed on remote IP.

Defined here so routers and `main.py` import the same instance. Wire-up:

  main.py:
    from core.rate_limit import limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

  any router (must accept `request: Request`):
    from core.rate_limit import limiter

    @router.post("/login")
    @limiter.limit("10/minute")
    def login(request: Request, ...):
        ...
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
