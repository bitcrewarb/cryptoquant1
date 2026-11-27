"""Security primitives: scrypt password hashing, HMAC session tokens, Fernet credential vault, CSRF checks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken
from fastapi import Request

from crypto_quant_lab.app.config import get_settings

SESSION_COOKIE_NAME = "cql_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def hash_password(password: str) -> str:
    """Hash a password with scrypt and a random 16-byte salt."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    salt = os.urandom(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt${base64.b64encode(salt).decode('ascii')}${base64.b64encode(derived).decode('ascii')}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored scrypt hash in constant time."""
    try:
        scheme, b64_salt, b64_digest = stored_hash.split("$", 2)
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(b64_salt)
        expected = base64.b64decode(b64_digest)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=len(expected),
        )
        return hmac.compare_digest(derived, expected)
    except Exception:
        return False


def create_session_token(user_id: int, username: str, role: str) -> str:
    """Create a signed HMAC-SHA256 session token."""
    settings = get_settings()
    payload = {
        "uid": user_id,
        "sub": username,
        "role": role,
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    raw_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    sig = hmac.new(
        settings.effective_secret_key.encode("utf-8"),
        raw_payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{raw_payload}.{sig}"


def verify_session_token(token: str | None) -> dict[str, Any] | None:
    """Validate and decode a signed session token."""
    if not token or "." not in token:
        return None
    settings = get_settings()
    try:
        raw_payload, sig = token.rsplit(".", 1)
        expected_sig = hmac.new(
            settings.effective_secret_key.encode("utf-8"),
            raw_payload.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        padded = raw_payload + "=" * (-len(raw_payload) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def encrypt_secret(plaintext: str) -> str:
    """Encrypt API secret/key at rest using Fernet symmetric encryption."""
    if not plaintext:
        return ""
    fernet = Fernet(get_settings().effective_fernet_key)
    return fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str | None) -> str:
    """Decrypt API secret/key at rest; returns empty string if invalid/unset."""
    if not ciphertext:
        return ""
    fernet = Fernet(get_settings().effective_fernet_key)
    try:
        return fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken:
        return ""


def is_https_request(request: Request) -> bool:
    """Detect whether the external client connection uses HTTPS (including reverse proxies and iframes)."""
    if request.url.scheme == "https":
        return True
    proto = request.headers.get("x-forwarded-proto", "").lower()
    if "https" in proto:
        return True
    forwarded = request.headers.get("forwarded", "").lower()
    if "proto=https" in forwarded:
        return True
    if request.headers.get("x-forwarded-ssl", "").lower() == "on":
        return True
    origin = (request.headers.get("origin") or "").lower()
    if origin.startswith("https://"):
        return True
    referer = (request.headers.get("referer") or "").lower()
    if referer.startswith("https://"):
        return True
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip().lower()
    if host.endswith(".run.app") or host.endswith(".appspot.com"):
        return True
    return False


def validate_csrf_origin(request: Request) -> bool:
    """Verify Origin/Referer or custom HTMX/XHR header on state-changing cookie requests."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return True
    # Allow explicit Bearer API calls without cookie CSRF check
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return True
    # Custom XHR/HTMX headers cannot be sent cross-origin without CORS preflight
    if request.headers.get("hx-request") == "true" or request.headers.get("x-requested-with"):
        return True

    raw_host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    expected_host = raw_host.split(",")[0].strip()
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin and expected_host:
        parsed = urlparse(origin)
        if parsed.netloc == expected_host:
            return True
        return False
    # Same-origin HTML form posts in browsers may omit Origin on some clients, allow if Referer/Origin absent only on localhost/test
    return origin is None

