from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"
    # Set to false to skip mounting the native MCP endpoint at /mcp.
    mcp_endpoint_enabled: bool = True


settings = Settings()
