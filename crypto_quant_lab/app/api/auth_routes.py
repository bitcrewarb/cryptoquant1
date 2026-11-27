"""Authentication routes: first-run admin claim, login, logout, and current-user inspection."""

from __future__ import annotations

import hmac
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from crypto_quant_lab.app.auth.security import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    hash_password,
    is_https_request,
    verify_password,
)
from crypto_quant_lab.app.config import get_settings
from crypto_quant_lab.app.dependencies import get_optional_user, needs_first_run_setup, require_user
from crypto_quant_lab.database.models import User
from crypto_quant_lab.database.session import get_db

router = APIRouter()
templates = Jinja2Templates(directory="crypto_quant_lab/app/templates")


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    https = is_https_request(request)
    kwargs: dict[str, object] = {
        "key": SESSION_COOKIE_NAME,
        "value": token,
        "max_age": SESSION_TTL_SECONDS,
        "httponly": True,
        "path": "/",
        "secure": https,
        "samesite": "none" if https else "lax",
    }
    response.set_cookie(**kwargs)  # type: ignore[arg-type]
    if https:
        # Add Partitioned attribute (CHIPS) for cross-site iframe preview compatibility
        existing = response.headers.get("set-cookie", "")
        if existing and "Partitioned" not in existing:
            response.headers["set-cookie"] = f"{existing}; Partitioned"


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
) -> Response:
    if user is not None:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    settings = get_settings()
    first_run = needs_first_run_setup(db) and not bool(settings.admin_passcode.strip())
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "first_run": first_run,
            "has_env_passcode": bool(settings.admin_passcode.strip()),
            "error": request.query_params.get("error", ""),
            "cookie_blocked": request.query_params.get("cookie_blocked") == "1",
        },
    )


@router.post("/auth/claim-admin")
def claim_admin(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(""),
    db: Session = Depends(get_db),
) -> Response:
    settings = get_settings()
    if settings.admin_passcode.strip():
        raise HTTPException(status_code=403, detail="Admin passcode is managed via environment configuration")
    user_count = db.scalar(select(func.count(User.id))) or 0
    if user_count > 0:
        raise HTTPException(status_code=403, detail="Administrator account already initialized")

    uname = username.strip()
    if len(uname) < 3 or len(password) < 8:
        return RedirectResponse(
            url="/login?error=Username+must+be+3%2B+chars+and+password+8%2B+chars",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    admin = User(
        username=uname,
        email=email.strip() or None,
        password_hash=hash_password(password),
        role="admin",
        last_login_at=datetime.now(timezone.utc),
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    token = create_session_token(admin.id, admin.username, admin.role)
    resp = RedirectResponse(url="/?cql_chk=1", status_code=status.HTTP_303_SEE_OTHER)
    _set_auth_cookie(resp, request, token)
    return resp


def _find_user_by_identifier(db: Session, identifier: str) -> User | None:
    ident = identifier.strip().lower()
    if not ident:
        return None
    return db.scalar(
        select(User).where(
            or_(
                func.lower(User.username) == ident,
                func.lower(User.email) == ident,
            )
        )
    )


@router.post("/auth/login")
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> Response:
    settings = get_settings()
    uname = username.strip()
    user = _find_user_by_identifier(db, uname)

    authenticated_user: User | None = None
    env_pass = settings.admin_passcode.strip()

    if env_pass and hmac.compare_digest(password, env_pass):
        if user is None:
            user = User(
                username=uname or "admin",
                password_hash=hash_password(password),
                role="admin",
            )
            db.add(user)
            db.flush()
        authenticated_user = user
    elif user is not None and verify_password(password, user.password_hash):
        authenticated_user = user

    if authenticated_user is None or not authenticated_user.is_active:
        first_run = needs_first_run_setup(db) and not bool(env_pass)
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "first_run": first_run,
                "has_env_passcode": bool(env_pass),
                "error": "Invalid username/email or password. Check your spelling and try again.",
                "cookie_blocked": False,
            },
        )

    authenticated_user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_session_token(authenticated_user.id, authenticated_user.username, authenticated_user.role)
    resp = RedirectResponse(url="/?cql_chk=1", status_code=status.HTTP_303_SEE_OTHER)
    _set_auth_cookie(resp, request, token)
    return resp


@router.post("/api/auth/token")
def issue_api_token(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    settings = get_settings()
    user = _find_user_by_identifier(db, username)
    env_pass = settings.admin_passcode.strip()
    ok = False
    if env_pass and hmac.compare_digest(password, env_pass):
        if user is None:
            user = User(username=username.strip() or "admin", password_hash=hash_password(password), role="admin")
            db.add(user)
            db.commit()
            db.refresh(user)
        ok = True
    elif user is not None and verify_password(password, user.password_hash):
        ok = True
    if not ok or user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_session_token(user.id, user.username, user.role), "token_type": "bearer"}


@router.post("/auth/logout")
def logout(request: Request) -> Response:
    https = is_https_request(request)
    resp = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=https,
        httponly=True,
        samesite="none" if https else "lax",
    )
    if https:
        existing = resp.headers.get("set-cookie", "")
        if existing and "Partitioned" not in existing:
            resp.headers["set-cookie"] = f"{existing}; Partitioned"
    return resp


@router.get("/api/auth/me")
def auth_me(user: User = Depends(require_user)) -> dict[str, object]:
    return {"id": user.id, "username": user.username, "role": user.role}
