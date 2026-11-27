"""FastAPI dependency injection helpers for database sessions and authenticated users."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.auth.security import (
    SESSION_COOKIE_NAME,
    validate_csrf_origin,
    verify_session_token,
)
from crypto_quant_lab.database.models import User
from crypto_quant_lab.database.session import get_db


def _extract_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.cookies.get(SESSION_COOKIE_NAME)


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = _extract_token(request)
    payload = verify_session_token(token)
    if not payload:
        return None
    user_id = payload.get("uid")
    if not user_id:
        return None
    user = db.get(User, int(user_id))
    if not user or not user.is_active:
        return None
    return user


def require_user(
    request: Request,
    user: User | None = Depends(get_optional_user),
) -> User:
    if not validate_csrf_origin(request):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid request origin")
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required")
    return user


def needs_first_run_setup(db: Session) -> bool:
    count = db.scalar(select(func.count(User.id))) or 0
    return count == 0
