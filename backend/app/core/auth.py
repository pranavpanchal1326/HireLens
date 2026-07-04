"""Security and Authentication Core.

Implements HTTP Basic Authentication and Recruiter Account resolution.
"""

from __future__ import annotations

import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)

security = HTTPBasic()


class RecruiterAccount(BaseModel):
    """Secure model conveying recruiter identity and account isolation boundaries."""

    account_id: str
    recruiter_id: str


# Static in-memory database of authorized recruiter accounts for early stages (PRD §9).
VALID_RECRUITERS = {
    "recruiter_one": {"password": "password123", "account_id": "company_a"},
    "recruiter_two": {"password": "password456", "account_id": "company_b"},
    "recruiter_three": {"password": "password789", "account_id": "company_c"},
}


def get_current_recruiter(credentials: HTTPBasicCredentials = Depends(security)) -> RecruiterAccount:
    """Dependency that extracts and validates basic auth credentials.

    Raises:
        HTTPException: 401 on missing, invalid, or mismatched credentials.
    """
    username = credentials.username
    password = credentials.password

    user_info = VALID_RECRUITERS.get(username)
    if not user_info or user_info["password"] != password:
        logger.warning(f"Failed authentication attempt for user: '{username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect recruiter credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )

    logger.info(f"Successfully authenticated recruiter '{username}' (account: '{user_info['account_id']}')")
    return RecruiterAccount(
        account_id=user_info["account_id"],
        recruiter_id=username,
    )
