"""Benchmarking module for inference performance measurement."""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from minimax_deploy.client import InferenceClient, InferenceResult
from minimax_deploy.config import Settings, get_settings

console = Console()

# Default benchmark prompts covering various use cases
DEFAULT_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a short poem about artificial intelligence.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis step by step.",
    "Write a function in Python that calculates the Fibonacci sequence.",
    "Explain quantum computing to a 10-year-old.",
    "What are the ethical considerations of AI development?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "How does a neural network learn from data?",
    "Write a haiku about the ocean.",
]


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    num_requests: int
    concurrency: int
    total_time_seconds: float
    successful_requests: int
    failed_requests: int

    # Latency metrics (ms)
    latency_mean_ms: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float

    # Throughput metrics
    requests_per_second: float
    tokens_per_second: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    # Per-request token stats
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_tokens_per_request: float

    # Raw results for detailed analysis
    raw_results: list[InferenceResult] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "num_requests": self.num_requests,
            "concurrency": self.concurrency,
            "total_time_seconds": self.total_time_seconds,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_median_ms": self.latency_median_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_std_ms": self.latency_std_ms,
            "requests_per_second": self.requests_per_second,
            "tokens_per_second": self.tokens_per_second,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_tokens_per_request": self.avg_tokens_per_request,
        }

    def print_summary(self):
        """Print a formatted summary table."""
        table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(self.num_requests))
        table.add_row("Successful", str(self.successful_requests))
        table.add_row("Failed", str(self.failed_requests))
        table.add_row("Concurrency", str(self.concurrency))
        table.add_row("Total Time", f"{self.total_time_seconds:.2f}s")
        table.add_row("", "")
        table.add_row("Latency (mean)", f"{self.latency_mean_ms:.2f}ms")
        table.add_row("Latency (median)", f"{self.latency_median_ms:.2f}ms")
        table.add_row("Latency (p90)", f"{self.latency_p90_ms:.2f}ms")
        table.add_row("Latency (p95)", f"{self.latency_p95_ms:.2f}ms")
        table.add_row("Latency (p99)", f"{self.latency_p99_ms:.2f}ms")
        table.add_row("", "")
        table.add_row("Requests/sec", f"{self.requests_per_second:.2f}")
        table.add_row("Tokens/sec", f"{self.tokens_per_second:.2f}")
        table.add_row("Total Tokens", str(self.total_tokens))

        console.print(table)


class BenchmarkRunner:
    """Runner for inference benchmarks."""

    def __init__(
        self,
        client: InferenceClient | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> InferenceClient:
        """Get or create inference client."""
        if self._client is None:
            self._client = InferenceClient(settings=self._settings)
        return self._client

    async def _run_single_request(
        self,
        prompt: str,
        max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> InferenceResult | None:
        """Run a single inference request with concurrency control."""
        async with semaphore:
            try:
                return await self._get_client().agenerate(prompt, max_tokens=max_tokens)
            except Exception as e:
                console.print(f"[red]Request failed: {e}[/red]")
                return None

    async def run_async(
        self,
        prompts: list[str] | None = None,
        num_requests: int | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        warmup_requests: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        """Run benchmark asynchronously."""
        num_requests = num_requests or self._settings.benchmark_num_requests
        concurrency = concurrency or self._settings.benchmark_concurrency
        max_tokens = max_tokens or self._settings.benchmark_max_tokens
        warmup_requests = warmup_requests or self._settings.benchmark_warmup_requests

        # Use provided prompts or cycle through defaults
        if prompts is None:
            prompts = DEFAULT_PROMPTS

        # Extend prompts to match num_requests
        extended_prompts = []
        for i in range(num_requests):
            extended_prompts.append(prompts[i % len(prompts)])

        # Warmup
        if warmup_requests > 0:
            console.print(f"[yellow]Running {warmup_requests} warmup requests...[/yellow]")
            warmup_semaphore = asyncio.Semaphore(concurrency)
            warmup_tasks = [
                self._run_single_request(extended_prompts[i], max_tokens, warmup_semaphore)
                for i in range(min(warmup_requests, len(extended_prompts)))
            ]
            await asyncio.gather(*warmup_tasks)
            console.print("[green]Warmup complete.[/green]")

        # Run benchmark
        semaphore = asyncio.Semaphore(concurrency)
        results: list[InferenceResult | None] = []

        start_time = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmark...", total=num_requests)

            async def run_with_progress(prompt: str) -> InferenceResult | None:
                result = await self._run_single_request(prompt, max_tokens, semaphore)
                progress.advance(task)
                if progress_callback:
                    progress_callback(progress.tasks[task].completed, num_requests)
                return result

            tasks = [run_with_progress(p) for p in extended_prompts]
            results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Filter successful results
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)

        if not successful_results:
            raise RuntimeError("All benchmark requests failed")

        # Calculate metrics
        latencies = [r.latency_ms for r in successful_results]
        latencies_sorted = sorted(latencies)

        def percentile(data: list[float], p: float) -> float:
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

        total_prompt_tokens = sum(r.prompt_tokens for r in successful_results)
        total_completion_tokens = sum(r.completion_tokens for r in successful_results)
        total_tokens = total_prompt_tokens + total_completion_tokens

        return BenchmarkResult(
            num_requests=num_requests,
            concurrency=concurrency,
            total_time_seconds=total_time,
            successful_requests=len(successful_results),
            failed_requests=failed_count,
            latency_mean_ms=statistics.mean(latencies),
            latency_median_ms=statistics.median(latencies),
            latency_p50_ms=percentile(latencies_sorted, 50),
            latency_p90_ms=percentile(latencies_sorted, 90),
            latency_p95_ms=percentile(latencies_sorted, 95),
            latency_p99_ms=percentile(latencies_sorted, 99),
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            requests_per_second=len(successful_results) / total_time,
            tokens_per_second=total_completion_tokens / total_time,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            avg_prompt_tokens=total_prompt_tokens / len(successful_results),
            avg_completion_tokens=total_completion_tokens / len(successful_results),
            avg_tokens_per_request=total_tokens / len(successful_results),
            raw_results=successful_results,
        )

    def run(
        self,
        prompts: list[str] | None = None,
        num_requests: int | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        warmup_requests: int | None = None,
    ) -> BenchmarkResult:
        """Run benchmark synchronously."""
        return asyncio.run(
            self.run_async(
                prompts=prompts,
                num_requests=num_requests,
                concurrency=concurrency,
                max_tokens=max_tokens,
                warmup_requests=warmup_requests,
            )
        )
