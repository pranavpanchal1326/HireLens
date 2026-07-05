"""Security and Authentication Core.

Implements HTTP Basic Authentication and Recruiter Account resolution.

PRD §9 specifies *Basic auth* on recruiter endpoints — this module honors that
wording literally (HTTP Basic) rather than inventing JWT/OAuth scope the PRD does
not name. What it hardens vs. a naive implementation:

  - Passwords are never stored or compared in plaintext. Each account keeps a
    salted PBKDF2-HMAC-SHA256 hash (stdlib ``hashlib`` — no third-party crypto
    dependency), verified in constant time via ``hmac.compare_digest``.
  - Unknown usernames still pay the full hash cost (a dummy verification runs),
    so response timing does not leak which usernames exist (user-enumeration
    defense).
  - Demo credentials live in ``_DEMO_ACCOUNTS`` for zero-setup local dev, but can
    be overridden from the environment (``RECRUITER_CREDENTIALS``) so real
    secrets never live in source.

What this module deliberately does NOT do: issue tokens, manage sessions, handle
signup/refresh, or touch scoring/parsing/RAG logic.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)

security = HTTPBasic()

# PBKDF2 parameters. 200k iterations is a reasonable work factor for a POC; the
# salt is per-account so identical passwords never share a hash.
_PBKDF2_ROUNDS = 200_000
_PBKDF2_DIGEST = "sha256"


class RecruiterAccount(BaseModel):
    """Secure model conveying recruiter identity and account isolation boundaries."""

    account_id: str
    recruiter_id: str


def _hash_password(password: str, salt: bytes) -> bytes:
    """Derive a salted PBKDF2-HMAC hash for ``password``."""
    return hashlib.pbkdf2_hmac(
        _PBKDF2_DIGEST, password.encode("utf-8"), salt, _PBKDF2_ROUNDS
    )


class _StoredCredential(BaseModel):
    """One account's non-reversible credential material + isolation boundary."""

    account_id: str
    salt: bytes
    password_hash: bytes


# Demo accounts for zero-setup local dev. These match the credentials the Phase
# 7.8 isolation tests exercise; they are NON-SECRET fixtures, deliberately obvious.
# Override in any real deployment via the RECRUITER_CREDENTIALS env var (JSON:
# {"username": {"password": "...", "account_id": "..."}}).
_DEMO_ACCOUNTS: dict[str, dict[str, str]] = {
    "recruiter_one": {"password": "password123", "account_id": "company_a"},
    "recruiter_two": {"password": "password456", "account_id": "company_b"},
    "recruiter_three": {"password": "password789", "account_id": "company_c"},
}


def _load_account_source() -> dict[str, dict[str, str]]:
    """Load account definitions from the environment, falling back to demo accounts."""
    raw = os.environ.get("RECRUITER_CREDENTIALS")
    if not raw:
        return _DEMO_ACCOUNTS
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("RECRUITER_CREDENTIALS must be a JSON object.")
        return parsed
    except (ValueError, TypeError) as exc:
        logger.error("Invalid RECRUITER_CREDENTIALS env var, ignoring it: %s", exc)
        return _DEMO_ACCOUNTS


def _build_store() -> dict[str, _StoredCredential]:
    """Hash every configured account's password once, at import time."""
    store: dict[str, _StoredCredential] = {}
    for username, info in _load_account_source().items():
        salt = secrets.token_bytes(16)
        store[username] = _StoredCredential(
            account_id=info["account_id"],
            salt=salt,
            password_hash=_hash_password(info["password"], salt),
        )
    return store


# Non-reversible credential store, built once at import.
_CREDENTIAL_STORE: dict[str, _StoredCredential] = _build_store()

# Fixed dummy salt/hash used to run a constant-cost verification for unknown
# usernames, so a missing user takes the same wall-clock time as a wrong password.
_DUMMY_SALT = secrets.token_bytes(16)
_DUMMY_HASH = _hash_password("dummy-password-never-matches", _DUMMY_SALT)


def get_current_recruiter(
    credentials: HTTPBasicCredentials = Depends(security),
) -> RecruiterAccount:
    """Dependency that extracts and validates basic auth credentials.

    Verification is constant-time and does not reveal whether the failure was a
    bad username or a bad password.

    Raises:
        HTTPException: 401 on missing, invalid, or mismatched credentials.
    """
    username = credentials.username
    stored = _CREDENTIAL_STORE.get(username)

    if stored is None:
        # Spend the same work as a real verification to avoid timing-based
        # username enumeration, then fail.
        hmac.compare_digest(
            _hash_password(credentials.password, _DUMMY_SALT), _DUMMY_HASH
        )
        authenticated = False
        account_id = ""
    else:
        candidate = _hash_password(credentials.password, stored.salt)
        authenticated = hmac.compare_digest(candidate, stored.password_hash)
        account_id = stored.account_id

    if not authenticated:
        logger.warning("Failed authentication attempt for user: '%s'", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect recruiter credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

    logger.info(
        "Successfully authenticated recruiter '%s' (account: '%s')",
        username,
        account_id,
    )
    return RecruiterAccount(account_id=account_id, recruiter_id=username)
