"""Infisical secrets management for runtime secret injection."""

import os
from functools import lru_cache

from rich.console import Console

console = Console()


@lru_cache(maxsize=1)
def _get_infisical_client():
    """Get authenticated Infisical client (cached)."""
    try:
        from infisical_sdk import InfisicalSDKClient
    except ImportError:
        console.print("[yellow]infisicalsdk not installed, skipping Infisical[/yellow]")
        return None

    host = os.getenv("INFISICAL_HOST", "https://app.infisical.com")
    client_id = os.getenv("INFISICAL_CLIENT_ID")
    client_secret = os.getenv("INFISICAL_CLIENT_SECRET")

    if not client_id or not client_secret:
        return None

    try:
        client = InfisicalSDKClient(host=host)
        client.auth.universal_auth.login(client_id, client_secret)
        return client
    except Exception as e:
        console.print(f"[yellow]Infisical auth failed: {e}[/yellow]")
        return None


def load_secrets(
    project_id: str | None = None,
    environment: str = "dev",
    secret_path: str = "/",
    keys: list[str] | None = None,
) -> dict[str, str]:
    """
    Load secrets from Infisical and return as dict.
    
    Args:
        project_id: Infisical project ID (or INFISICAL_PROJECT_ID env var)
        environment: Environment slug (dev, staging, prod)
        secret_path: Path within the environment
        keys: Specific keys to fetch (fetches all if None)
    
    Returns:
        Dict of secret key -> value
    """
    project_id = project_id or os.getenv("INFISICAL_PROJECT_ID")
    if not project_id:
        return {}

    client = _get_infisical_client()
    if not client:
        return {}

    secrets = {}
    try:
        if keys:
            for key in keys:
                try:
                    secret = client.secrets.get_secret_by_name(
                        secret_name=key,
                        project_id=project_id,
                        environment_slug=environment,
                        secret_path=secret_path,
                    )
                    secrets[key] = secret.secretValue
                except Exception:
                    pass  # Key not found, skip
        else:
            # Fetch all secrets in path
            all_secrets = client.secrets.list_secrets(
                project_id=project_id,
                environment_slug=environment,
                secret_path=secret_path,
            )
            for secret in all_secrets.secrets:
                secrets[secret.secretKey] = secret.secretValue
    except Exception as e:
        console.print(f"[yellow]Failed to fetch secrets: {e}[/yellow]")

    return secrets


def inject_secrets(
    project_id: str | None = None,
    environment: str = "dev",
    secret_path: str = "/",
    keys: list[str] | None = None,
    override: bool = False,
) -> int:
    """
    Load secrets from Infisical and inject into environment variables.
    
    Args:
        project_id: Infisical project ID
        environment: Environment slug
        secret_path: Path within the environment
        keys: Specific keys to fetch
        override: If True, override existing env vars
    
    Returns:
        Number of secrets injected
    """
    secrets = load_secrets(
        project_id=project_id,
        environment=environment,
        secret_path=secret_path,
        keys=keys,
    )

    injected = 0
    for key, value in secrets.items():
        if override or key not in os.environ:
            os.environ[key] = value
            injected += 1

    if injected > 0:
        console.print(f"[green]✓ Injected {injected} secrets from Infisical[/green]")

    return injected


def init_secrets(environment: str | None = None):
    """
    Initialize secrets from Infisical at application startup.
    
    Convenience function that uses environment variables for configuration:
    - INFISICAL_CLIENT_ID: Machine identity client ID
    - INFISICAL_CLIENT_SECRET: Machine identity client secret
    - INFISICAL_PROJECT_ID: Project ID
    - INFISICAL_ENV: Environment slug (default: dev)
    
    Injects these secrets if found:
    - WANDB_API_KEY
    - HF_TOKEN
    """
    env = environment or os.getenv("INFISICAL_ENV", "dev")
    
    # Keys we care about for this project
    target_keys = [
        "WANDB_API_KEY",
        "HF_TOKEN",
        "MINIMAX_VLLM_API_KEY",
    ]

    return inject_secrets(
        environment=env,
        keys=target_keys,
        override=False,  # Don't override if already set
    )
