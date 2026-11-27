"""Application configuration via Pydantic Settings with secure environment handling."""

from __future__ import annotations

import base64
import hashlib
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ephemeral process-local fallback keys if env vars are not configured.
# Never hardcoded; regenerated per process if missing from environment.
_EPHEMERAL_SECRET_KEY = secrets.token_urlsafe(48)
_EPHEMERAL_FERNET_KEY = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="CRYPTO QUANT LAB")
    app_env: str = Field(default="development")
    app_url: str = Field(default="http://localhost:3000")
    log_level: str = Field(default="INFO")

    secret_key: str = Field(default="")
    credentials_encryption_key: str = Field(default="")
    admin_passcode: str = Field(default="")

    database_url: str = Field(default="")
    data_store_dir: str = Field(default="./data_store")

    redis_url: str = Field(default="")
    celery_broker_url: str = Field(default="")
    celery_result_backend: str = Field(default="")

    binance_spot_base_url: str = Field(default="https://api.binance.com")
    binance_futures_base_url: str = Field(default="https://fapi.binance.com")
    binance_ws_spot_url: str = Field(default="wss://stream.binance.com:9443/ws")
    binance_ws_futures_url: str = Field(default="wss://fstream.binance.com/ws")

    live_trading_enabled: bool = Field(default=False)

    @property
    def data_dir(self) -> Path:
        path = Path(self.data_store_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def parquet_dir(self) -> Path:
        path = self.data_dir / "parquet"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resolved_database_url(self) -> str:
        if self.database_url and self.database_url.strip():
            return self.database_url.strip()
        db_path = self.data_dir / "cql.db"
        return f"sqlite:///{db_path}"

    @property
    def effective_secret_key(self) -> str:
        if self.secret_key.strip():
            return self.secret_key.strip()
        key_file = self.data_dir / ".secret_key"
        try:
            if key_file.exists():
                cached = key_file.read_text(encoding="utf-8").strip()
                if len(cached) >= 32:
                    return cached
            key_file.write_text(_EPHEMERAL_SECRET_KEY, encoding="utf-8")
            key_file.chmod(0o600)
        except OSError:
            pass
        return _EPHEMERAL_SECRET_KEY

    @property
    def effective_fernet_key(self) -> bytes:
        raw = self.credentials_encryption_key.strip()
        if raw:
            # Derive valid 32-byte URL-safe base64 Fernet key deterministically from configured key
            digest = hashlib.sha256(raw.encode("utf-8")).digest()
            return base64.urlsafe_b64encode(digest)
        return _EPHEMERAL_FERNET_KEY.encode("ascii")

    @property
    def has_persistent_encryption_key(self) -> bool:
        return bool(self.credentials_encryption_key.strip())

    @property
    def celery_enabled(self) -> bool:
        return bool(self.celery_broker_url.strip() and self.redis_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
