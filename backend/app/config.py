"""Application Configuration Module.

Loads and validates settings from environment variables and .env file.
"""

from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project Root Directory (D:\StockMind-AI)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable bindings."""

    # 1. Application Metadata & Mode
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    APP_NAME: str = Field(default="StockMind-AI", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")
    API_V1_PREFIX: str = Field(default="/api/v1", description="Prefix for API v1 routes")

    # 2. Server Configuration
    BACKEND_HOST: str = Field(default="127.0.0.1", description="Backend host")
    BACKEND_PORT: int = Field(default=8000, description="Backend port")
    ALLOWED_ORIGINS: Union[List[str], str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )

    # 3. Security & JWT
    SECRET_KEY: str = Field(
        default="stockmind-dev-insecure-secret-key-change-in-production-only-32bytes",
        description="JWT cryptographic signing key"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="Token expiration in minutes")

    # 4. Database Configuration (PostgreSQL-ready with SQLite fallback)
    DATABASE_TYPE: str = Field(default="sqlite", description="Database engine type: sqlite or postgresql")
    SQLITE_DB_PATH: str = Field(default="database/stockmind.db", description="Relative SQLite DB file path")
    DATABASE_URL: str = Field(
        default="",
        description="Full database URL (defaults to SQLite async URL if empty)"
    )

    # 5. Redis / Cache (Optional for Phase 1)
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")

    # 6. Market Data Engine Configuration
    MARKET_DATA_MODE: str = Field(
        default="REPLAY",
        description="Market data mode: 'LIVE' or 'REPLAY'/'SIMULATED'"
    )
    MARKET_DATA_PROVIDER: str = Field(
        default="REPLAY",
        description="Market provider: 'REPLAY', 'ZERODHA', 'UPSTOX', 'FINVASIA', etc."
    )
    MARKET_API_KEY: str = Field(default="", description="Broker/market data API key")
    MARKET_API_SECRET: str = Field(default="", description="Broker/market data API secret")
    MARKET_ACCESS_TOKEN: str = Field(default="", description="Broker/market data session access token")
    MARKET_WS_URL: str = Field(default="", description="Broker WebSocket endpoint URL")
    MARKET_DEFAULT_EXCHANGE: str = Field(default="NSE", description="Default exchange (NSE/BSE)")

    # 7. Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_DIR: str = Field(default="logs", description="Logs directory")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def async_database_url(self) -> str:
        """Resolve async SQLAlchemy database connection string.

        Provides automatic driver bridging:
        - postgresql:// -> postgresql+asyncpg://
        - sqlite:// -> sqlite+aiosqlite:///
        Defaults to local SQLite on USB drive if not specified.
        """
        raw_url = self.DATABASE_URL.strip()

        if raw_url:
            if raw_url.startswith("postgresql://"):
                return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            if raw_url.startswith("sqlite://") and not raw_url.startswith("sqlite+aiosqlite://"):
                return raw_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
            return raw_url

        # SQLite fallback for USB local development
        db_path = BASE_DIR / self.SQLITE_DB_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Use 3 slashes with forward slashes for cross-platform SQLite URL
        return f"sqlite+aiosqlite:///{db_path.as_posix()}"

    def validate_configuration(self) -> dict:
        """Perform comprehensive configuration validation and USB environment audit."""
        issues = []
        warnings = []

        # 1. USB Storage Confinement Check
        sqlite_full = (BASE_DIR / self.SQLITE_DB_PATH).resolve()
        log_full = (BASE_DIR / self.LOG_DIR).resolve()
        
        # Ensure paths remain confined inside BASE_DIR (D:\StockMind-AI)
        try:
            sqlite_full.relative_to(BASE_DIR)
        except ValueError:
            issues.append(f"SQLITE_DB_PATH escapes USB root: {sqlite_full}")

        try:
            log_full.relative_to(BASE_DIR)
        except ValueError:
            issues.append(f"LOG_DIR escapes USB root: {log_full}")

        # 2. Market Mode & Live Credential Audit
        is_live = self.MARKET_DATA_MODE.upper() == "LIVE"
        if is_live:
            if not self.MARKET_API_KEY or "your_" in self.MARKET_API_KEY.lower():
                warnings.append(
                    "MARKET_DATA_MODE is 'LIVE' but MARKET_API_KEY is not configured. "
                    "Manual broker API credentials required in .env. Replay fallback active."
                )
            if not self.MARKET_ACCESS_TOKEN or "your_" in self.MARKET_ACCESS_TOKEN.lower():
                warnings.append(
                    "MARKET_ACCESS_TOKEN missing or placeholder for LIVE market streaming."
                )

        # 3. Security Audit
        if self.ENVIRONMENT == "production":
            if "insecure" in self.SECRET_KEY.lower() or len(self.SECRET_KEY) < 32:
                issues.append("Production environment detected with insecure or truncated SECRET_KEY.")

        status_str = "INVALID" if issues else ("WARNING" if warnings else "VALID")

        return {
            "status": status_str,
            "environment": self.ENVIRONMENT,
            "usb_root": str(BASE_DIR),
            "is_usb_confined": len(issues) == 0,
            "market_mode": self.MARKET_DATA_MODE,
            "market_provider": self.MARKET_DATA_PROVIDER,
            "database_type": self.DATABASE_TYPE,
            "database_path": str(sqlite_full),
            "issues": issues,
            "warnings": warnings,
        }


settings = Settings()
