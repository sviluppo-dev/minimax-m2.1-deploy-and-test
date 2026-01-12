"""CLI interface for minimax-deploy."""

import typer
from rich.console import Console

from minimax_deploy.benchmark import BenchmarkRunner
from minimax_deploy.client import InferenceClient
from minimax_deploy.config import get_settings
from minimax_deploy.wandb_logger import WandbLogger

app = typer.Typer(
    name="minimax-deploy",
    help="MiniMax-M2.1 deployment and benchmarking CLI",
)
console = Console()


@app.command()
def health(
    base_url: str = typer.Option(None, "--url", "-u", help="vLLM server URL"),
):
    """Check vLLM server health."""
    settings = get_settings()
    url = base_url or settings.vllm_base_url

    client = InferenceClient(base_url=url)
    if client.health_check():
        console.print(f"[green]✓ vLLM server at {url} is healthy[/green]")
        raise typer.Exit(0)
    else:
        console.print(f"[red]✗ vLLM server at {url} is not responding[/red]")
        raise typer.Exit(1)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt for generation"),
    max_tokens: int = typer.Option(256, "--max-tokens", "-m", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    base_url: str = typer.Option(None, "--url", "-u", help="vLLM server URL"),
):
    """Generate text from a prompt."""
    settings = get_settings()
    url = base_url or settings.vllm_base_url

    with InferenceClient(base_url=url) as client:
        result = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    console.print(f"\n[bold cyan]Response:[/bold cyan]\n{result.text}")
    console.print(f"\n[dim]Tokens: {result.total_tokens} | Latency: {result.latency_ms:.0f}ms | "
                  f"Speed: {result.tokens_per_second:.1f} tok/s[/dim]")


@app.command()
def benchmark(
    num_requests: int = typer.Option(None, "--requests", "-n", help="Number of requests"),
    concurrency: int = typer.Option(None, "--concurrency", "-c", help="Concurrent requests"),
    max_tokens: int = typer.Option(None, "--max-tokens", "-m", help="Max tokens per request"),
    warmup: int = typer.Option(None, "--warmup", "-w", help="Warmup requests"),
    base_url: str = typer.Option(None, "--url", "-u", help="vLLM server URL"),
    log_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option(None, "--wandb-project", help="W&B project name"),
    wandb_run_name: str = typer.Option(None, "--wandb-run", help="W&B run name"),
):
    """Run inference benchmark."""
    settings = get_settings()

    client = InferenceClient(base_url=base_url or settings.vllm_base_url)
    runner = BenchmarkRunner(client=client, settings=settings)

    console.print("[bold]Starting benchmark...[/bold]\n")

    result = runner.run(
        num_requests=num_requests,
        concurrency=concurrency,
        max_tokens=max_tokens,
        warmup_requests=warmup,
    )

    result.print_summary()

    if log_wandb:
        console.print("\n[yellow]Logging to W&B...[/yellow]")
        with WandbLogger(
            project=wandb_project,
            run_name=wandb_run_name,
            settings=settings,
        ) as logger:
            logger.log_benchmark(result)
            logger.log_latency_histogram(result)
            logger.log_tokens_histogram(result)
        console.print("[green]✓ Results logged to W&B[/green]")


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-h", help="API server host"),
    port: int = typer.Option(None, "--port", "-p", help="API server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    console.print(f"[bold]Starting API server at http://{host}:{port}[/bold]")
    uvicorn.run(
        "minimax_deploy.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
