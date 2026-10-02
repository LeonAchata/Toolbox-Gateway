from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # Optional shared secret. When set, every request except /health must send
    # "Authorization: Bearer <key>".
    gateway_api_key: str | None = None
    cors_origins: list[str] = ["*"]

    # Model used when a request does not name one. Empty means "the first
    # provider that has credentials".
    default_model: str = ""

    request_timeout_seconds: float = 120.0
    max_concurrent_requests_per_provider: int = 8

    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 1000

    # AWS Bedrock. Credentials are optional: without them boto3 falls back to
    # its default chain (env, shared profile, instance or task role).
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "us.amazon.nova-pro-v1:0"

    openai_api_key: str | None = None
    openai_org_id: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"

    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    @field_validator(
        "gateway_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "openai_api_key",
        "openai_org_id",
        "openai_base_url",
        "google_api_key",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value):
        # docker compose passes unset variables through as empty strings.
        if isinstance(value, str) and (not value.strip() or value.startswith("your_")):
            return None
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
