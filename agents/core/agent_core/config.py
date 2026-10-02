from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = """\
You are a precise assistant with access to a set of tools.

- Use a tool whenever the request involves arithmetic, counting, text \
transformation or the current date and time. Do not do these in your head.
- You may call several tools at once when the calls do not depend on each other.
- If a tool returns an error, read it, fix the arguments and try again once.
- Keep answers short and state the final result clearly.
- Reply in the same language the user writes in."""


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "agent"
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    toolbox_url: str = "http://toolbox:8000"
    llm_gateway_url: str = "http://llm-gateway:8003"
    gateway_api_key: str | None = None

    # Empty means "let the gateway pick its default".
    default_model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Upper bound on LLM -> tools -> LLM round trips for a single message.
    max_tool_rounds: int = 6
    # How many previous user turns are sent back to the model as context.
    history_turns: int = 8
    # Conversations kept in memory before the oldest is evicted.
    max_sessions: int = 500

    @field_validator("gateway_api_key", mode="before")
    @classmethod
    def _blank_is_none(cls, value):
        return value or None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value
