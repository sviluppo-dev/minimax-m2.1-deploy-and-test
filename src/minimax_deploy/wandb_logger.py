"""Weights & Biases logging for benchmark results."""

from datetime import datetime
from typing import Any

import wandb

from minimax_deploy.benchmark import BenchmarkResult
from minimax_deploy.config import Settings, get_settings


class WandbLogger:
    """Logger for sending benchmark results to W&B."""

    def __init__(
        self,
        project: str | None = None,
        entity: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._settings = settings or get_settings()
        self.project = project or self._settings.wandb_project
        self.entity = entity or self._settings.wandb_entity
        self.run_name = run_name or self._settings.wandb_run_name
        self.tags = tags or self._settings.wandb_tags
        self._config = config or {}
        self._run: wandb.sdk.wandb_run.Run | None = None

    def _generate_run_name(self) -> str:
        """Generate a default run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"baseline_{timestamp}"

    def init(self, **kwargs) -> "WandbLogger":
        """Initialize W&B run."""
        run_name = self.run_name or self._generate_run_name()

        merged_config = {
            "model": self._settings.model_name,
            "vllm_base_url": self._settings.vllm_base_url,
            "benchmark_num_requests": self._settings.benchmark_num_requests,
            "benchmark_concurrency": self._settings.benchmark_concurrency,
            "benchmark_max_tokens": self._settings.benchmark_max_tokens,
            **self._config,
        }

        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            tags=self.tags,
            config=merged_config,
            **kwargs,
        )
        return self

    def log_benchmark(
        self,
        result: BenchmarkResult,
        step: int | None = None,
        prefix: str = "benchmark",
    ) -> None:
        """Log benchmark results to W&B."""
        if self._run is None:
            self.init()

        metrics = result.to_dict()
        prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        if step is not None:
            wandb.log(prefixed_metrics, step=step)
        else:
            wandb.log(prefixed_metrics)

        # Log summary metrics
        wandb.run.summary[f"{prefix}/latency_p95_ms"] = result.latency_p95_ms
        wandb.run.summary[f"{prefix}/tokens_per_second"] = result.tokens_per_second
        wandb.run.summary[f"{prefix}/requests_per_second"] = result.requests_per_second

    def log_latency_histogram(
        self,
        result: BenchmarkResult,
        prefix: str = "benchmark",
    ) -> None:
        """Log latency distribution as histogram."""
        if self._run is None:
            self.init()

        latencies = [r.latency_ms for r in result.raw_results]
        wandb.log({f"{prefix}/latency_distribution": wandb.Histogram(latencies)})

    def log_tokens_histogram(
        self,
        result: BenchmarkResult,
        prefix: str = "benchmark",
    ) -> None:
        """Log token count distributions."""
        if self._run is None:
            self.init()

        completion_tokens = [r.completion_tokens for r in result.raw_results]
        prompt_tokens = [r.prompt_tokens for r in result.raw_results]

        wandb.log({
            f"{prefix}/completion_tokens_distribution": wandb.Histogram(completion_tokens),
            f"{prefix}/prompt_tokens_distribution": wandb.Histogram(prompt_tokens),
        })

    def log_custom(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log custom metrics."""
        if self._run is None:
            self.init()

        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def finish(self) -> None:
        """Finish W&B run."""
        if self._run is not None:
            wandb.finish()
            self._run = None

    def __enter__(self) -> "WandbLogger":
        self.init()
        return self

    def __exit__(self, *args) -> None:
        self.finish()


def log_baseline_benchmark(
    result: BenchmarkResult,
    project: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
    tags: list[str] | None = None,
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Convenience function to log a baseline benchmark to W&B."""
    with WandbLogger(
        project=project,
        entity=entity,
        run_name=run_name,
        tags=tags,
        config=extra_config,
    ) as logger:
        logger.log_benchmark(result)
        logger.log_latency_histogram(result)
        logger.log_tokens_histogram(result)
