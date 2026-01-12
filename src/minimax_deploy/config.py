"""Configuration management using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MINIMAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # vLLM server configuration
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for vLLM OpenAI-compatible API",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM server (usually 'EMPTY' for local)",
    )
    model_name: str = Field(
        default="MiniMaxAI/MiniMax-M1-80k",
        description="Model name/path for vLLM",
    )

    # Benchmark configuration
    benchmark_num_requests: int = Field(
        default=100,
        description="Number of requests for benchmark",
    )
    benchmark_concurrency: int = Field(
        default=10,
        description="Concurrent requests during benchmark",
    )
    benchmark_max_tokens: int = Field(
        default=256,
        description="Max tokens to generate per request",
    )
    benchmark_warmup_requests: int = Field(
        default=5,
        description="Warmup requests before benchmark",
    )

    # W&B configuration
    wandb_project: str = Field(
        default="minimax-m2.1-baseline",
        description="W&B project name",
    )
    wandb_entity: str | None = Field(
        default=None,
        description="W&B entity (team/user)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="W&B run name (auto-generated if not set)",
    )
    wandb_tags: list[str] = Field(
        default_factory=lambda: ["baseline", "vllm", "minimax-m2.1"],
        description="W&B run tags",
    )

    # API server configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8080, description="API server port")


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
