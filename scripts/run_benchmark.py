#!/usr/bin/env python3
"""Run baseline benchmark and log to W&B."""

import argparse
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimax_deploy import BenchmarkRunner, InferenceClient, Settings, WandbLogger
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run MiniMax-M2.1 baseline benchmark")
    parser.add_argument("--url", "-u", help="vLLM server URL")
    parser.add_argument("--requests", "-n", type=int, help="Number of requests")
    parser.add_argument("--concurrency", "-c", type=int, help="Concurrent requests")
    parser.add_argument("--max-tokens", "-m", type=int, help="Max tokens per request")
    parser.add_argument("--warmup", "-w", type=int, help="Warmup requests")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--wandb-entity", help="W&B entity")
    parser.add_argument("--wandb-run", help="W&B run name")
    parser.add_argument("--wandb-tags", nargs="+", help="W&B tags")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Build settings overrides
    settings = Settings()

    # Create client
    client = InferenceClient(
        base_url=args.url or settings.vllm_base_url,
        settings=settings,
    )

    # Check health
    console.print("[bold]Checking vLLM server health...[/bold]")
    if not client.health_check():
        console.print("[red]ERROR: vLLM server is not responding[/red]")
        sys.exit(1)
    console.print("[green]✓ vLLM server is healthy[/green]\n")

    # Run benchmark
    runner = BenchmarkRunner(client=client, settings=settings)

    console.print("[bold]Running baseline benchmark...[/bold]\n")
    result = runner.run(
        num_requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup_requests=args.warmup,
    )

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        result.print_summary()

    # Log to W&B
    if args.wandb:
        console.print("\n[yellow]Logging to W&B...[/yellow]")
        tags = args.wandb_tags or settings.wandb_tags

        with WandbLogger(
            project=args.wandb_project or settings.wandb_project,
            entity=args.wandb_entity or settings.wandb_entity,
            run_name=args.wandb_run,
            tags=tags,
            settings=settings,
        ) as logger:
            logger.log_benchmark(result)
            logger.log_latency_histogram(result)
            logger.log_tokens_histogram(result)

        console.print("[green]✓ Results logged to W&B[/green]")

    console.print("\n[bold green]Benchmark complete![/bold green]")


if __name__ == "__main__":
    main()
