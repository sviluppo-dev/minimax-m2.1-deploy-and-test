"""FastAPI service for inference and benchmarking."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from minimax_deploy.benchmark import BenchmarkRunner, BenchmarkResult, DEFAULT_PROMPTS
from minimax_deploy.client import InferenceClient
from minimax_deploy.config import get_settings
from minimax_deploy.wandb_logger import WandbLogger

# Global state
_client: InferenceClient | None = None
_benchmark_status: dict[str, Any] = {"running": False, "progress": 0, "total": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _client
    settings = get_settings()
    _client = InferenceClient(settings=settings)
    yield
    if _client:
        await _client.aclose()


app = FastAPI(
    title="MiniMax-M2.1 Inference API",
    description="API for inference and benchmarking MiniMax-M2.1 with vLLM",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    tokens_per_second: float


class BenchmarkRequest(BaseModel):
    num_requests: int = Field(default=100, ge=1, le=10000)
    concurrency: int = Field(default=10, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    warmup_requests: int = Field(default=5, ge=0, le=50)
    prompts: list[str] | None = Field(default=None, description="Custom prompts (uses defaults if not provided)")
    log_to_wandb: bool = Field(default=False)
    wandb_project: str | None = Field(default=None)
    wandb_run_name: str | None = Field(default=None)
    wandb_tags: list[str] | None = Field(default=None)


class BenchmarkResponse(BaseModel):
    num_requests: int
    concurrency: int
    total_time_seconds: float
    successful_requests: int
    failed_requests: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    requests_per_second: float
    tokens_per_second: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_tokens_per_request: float
    wandb_logged: bool = False


class HealthResponse(BaseModel):
    status: str
    vllm_healthy: bool
    model: str


class BenchmarkStatusResponse(BaseModel):
    running: bool
    progress: int
    total: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and vLLM server health."""
    settings = get_settings()
    vllm_healthy = _client.health_check() if _client else False
    return HealthResponse(
        status="ok" if vllm_healthy else "degraded",
        vllm_healthy=vllm_healthy,
        model=settings.model_name,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    if not _client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    try:
        result = await _client.agenerate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return GenerateResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            tokens_per_second=result.tokens_per_second,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """Run inference benchmark."""
    global _benchmark_status

    if _benchmark_status["running"]:
        raise HTTPException(status_code=409, detail="Benchmark already running")

    if not _client:
        raise HTTPException(status_code=503, detail="Client not initialized")

    _benchmark_status = {"running": True, "progress": 0, "total": request.num_requests}

    def progress_callback(current: int, total: int):
        _benchmark_status["progress"] = current
        _benchmark_status["total"] = total

    try:
        runner = BenchmarkRunner(client=_client)
        result = await runner.run_async(
            prompts=request.prompts,
            num_requests=request.num_requests,
            concurrency=request.concurrency,
            max_tokens=request.max_tokens,
            warmup_requests=request.warmup_requests,
            progress_callback=progress_callback,
        )

        wandb_logged = False
        if request.log_to_wandb:
            try:
                with WandbLogger(
                    project=request.wandb_project,
                    run_name=request.wandb_run_name,
                    tags=request.wandb_tags,
                ) as logger:
                    logger.log_benchmark(result)
                    logger.log_latency_histogram(result)
                    logger.log_tokens_histogram(result)
                wandb_logged = True
            except Exception as e:
                # Log but don't fail the benchmark
                print(f"W&B logging failed: {e}")

        return BenchmarkResponse(
            **result.to_dict(),
            wandb_logged=wandb_logged,
        )
    finally:
        _benchmark_status = {"running": False, "progress": 0, "total": 0}


@app.get("/benchmark/status", response_model=BenchmarkStatusResponse)
async def benchmark_status():
    """Get current benchmark status."""
    return BenchmarkStatusResponse(**_benchmark_status)


@app.get("/benchmark/prompts")
async def get_default_prompts():
    """Get default benchmark prompts."""
    return {"prompts": DEFAULT_PROMPTS}


@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)."""
    settings = get_settings()
    return {
        "model_name": settings.model_name,
        "vllm_base_url": settings.vllm_base_url,
        "benchmark_num_requests": settings.benchmark_num_requests,
        "benchmark_concurrency": settings.benchmark_concurrency,
        "benchmark_max_tokens": settings.benchmark_max_tokens,
        "wandb_project": settings.wandb_project,
    }
