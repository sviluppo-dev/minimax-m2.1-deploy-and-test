"""MiniMax-M2.1 deployment and benchmarking library."""

from minimax_deploy.client import InferenceClient
from minimax_deploy.benchmark import BenchmarkRunner, BenchmarkResult
from minimax_deploy.config import Settings
from minimax_deploy.wandb_logger import WandbLogger
from minimax_deploy.secrets import init_secrets, load_secrets, inject_secrets

__version__ = "0.1.0"
__all__ = [
    "InferenceClient",
    "BenchmarkRunner",
    "BenchmarkResult",
    "Settings",
    "WandbLogger",
    "init_secrets",
    "load_secrets",
    "inject_secrets",
]
