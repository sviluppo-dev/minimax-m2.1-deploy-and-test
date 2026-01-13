#!/usr/bin/env python3
"""
Initialize W&B project with baseline metrics structure.
Run this to create the project and log initial baseline data.

Usage:
    # With Infisical (recommended)
    export INFISICAL_CLIENT_ID=xxx
    export INFISICAL_CLIENT_SECRET=xxx
    export INFISICAL_PROJECT_ID=xxx
    python scripts/init_wandb_baseline.py

    # Or with direct API key
    export WANDB_API_KEY=xxx
    python scripts/init_wandb_baseline.py
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minimax_deploy.secrets import init_secrets

# Load secrets from Infisical if configured
init_secrets()

import wandb
from rich.console import Console

console = Console()

# Project configuration
PROJECT_NAME = "Minimax-m2.1 Router Study"
ENTITY = "sviluppo-ac"

# Baseline metrics structure matching the proposal
BASELINE_CONFIG = {
    "model": "MiniMaxAI/MiniMax-M1-80k",
    "model_params": "230B (sparse MoE)",
    "experiment": "TB2 Alignment Baseline",
    "method": "DPO + LoRA",
    "lora_rank": 64,
    "lora_alpha": 128,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "trainable_params": "~200M (0.09%)",
    "hardware": "8xA100 (Lambda Stack)",
    "framework": "vLLM + OpenHands",
    "data_source": "Pinecone trace index (~15k traces)",
    "preference_threshold": 0.15,
}

# Placeholder baseline metrics (will be replaced with real data)
BASELINE_METRICS = {
    # Inference performance (pre-training baseline)
    "baseline/latency_mean_ms": 0.0,
    "baseline/latency_p95_ms": 0.0,
    "baseline/tokens_per_second": 0.0,
    "baseline/requests_per_second": 0.0,
    
    # TB2 task performance (pre-training)
    "baseline/tb2_pass_rate": 0.0,
    "baseline/tb2_avg_score": 0.0,
    "baseline/tb2_tasks_evaluated": 0,
    
    # Training metrics (will be populated during training)
    "training/dpo_loss": 0.0,
    "training/chosen_rewards": 0.0,
    "training/rejected_rewards": 0.0,
    "training/reward_margin": 0.0,
    "training/epoch": 0,
    "training/step": 0,
}


def init_baseline_project():
    """Initialize W&B project with baseline structure."""
    console.print("[bold cyan]Initializing W&B project for MiniMax M2.1 TB2 Alignment[/bold cyan]\n")
    
    # Initialize run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        name=f"baseline-init-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["baseline", "initialization", "minimax-m2.1", "tb2"],
        config=BASELINE_CONFIG,
        notes="Initial baseline setup for MiniMax M2.1 TB2 alignment experiment. "
              "Real metrics will be logged when benchmark runs on Lambda 8xA100.",
    )
    
    console.print(f"[green]✓ Created W&B run: {run.name}[/green]")
    console.print(f"[green]✓ Project: {PROJECT_NAME}[/green]")
    console.print(f"[green]✓ Run URL: {run.url}[/green]\n")
    
    # Log baseline metrics
    wandb.log(BASELINE_METRICS)
    console.print("[green]✓ Logged baseline metrics structure[/green]")
    
    # Create summary
    wandb.run.summary["status"] = "awaiting_baseline_benchmark"
    wandb.run.summary["hardware"] = "8xA100 (Lambda Stack)"
    wandb.run.summary["model"] = "MiniMax-M2.1"
    wandb.run.summary["method"] = "DPO + LoRA"
    
    # Log experiment timeline from proposal
    timeline_table = wandb.Table(columns=["Day", "Milestone", "Status"])
    timeline_data = [
        ("0", "Data pipeline deployment, preference pair generation", "pending"),
        ("1", "LoRA training begins (DPO, β=0.1)", "pending"),
        ("2", "First checkpoint evaluation (TB2 subset)", "pending"),
        ("3", "Hyperparameter adjustment based on early signal", "pending"),
        ("4-5", "Full training run, checkpoint every 500 steps", "pending"),
        ("6", "Best checkpoint selection, full TB2 evaluation", "pending"),
        ("7", "Results documentation, failure analysis", "pending"),
    ]
    for day, milestone, status in timeline_data:
        timeline_table.add_data(day, milestone, status)
    
    wandb.log({"experiment/timeline": timeline_table})
    console.print("[green]✓ Logged experiment timeline[/green]")
    
    # Log LoRA config as artifact
    lora_config_artifact = wandb.Artifact("lora-config", type="config")
    lora_config_artifact.add(wandb.Table(
        columns=["Parameter", "Value"],
        data=[
            ["rank (r)", "64"],
            ["lora_alpha", "128"],
            ["target_modules", "q_proj, k_proj, v_proj, o_proj, gate_proj"],
            ["lora_dropout", "0.05"],
            ["bias", "none"],
            ["task_type", "CAUSAL_LM"],
        ]
    ), "lora_config")
    wandb.log_artifact(lora_config_artifact)
    console.print("[green]✓ Logged LoRA configuration artifact[/green]")
    
    # Finish run
    wandb.finish()
    
    console.print("\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]W&B Project Initialized Successfully![/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print(f"\n[cyan]Project URL:[/cyan] https://wandb.ai/{ENTITY or 'your-entity'}/{PROJECT_NAME}")
    console.print(f"[cyan]Run URL:[/cyan] {run.url}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Run baseline benchmark on Lambda: uv run python scripts/run_benchmark.py --wandb")
    console.print("2. Start DPO training with preference pairs from Pinecone")
    console.print("3. Monitor training metrics in W&B dashboard")
    
    return run.url


if __name__ == "__main__":
    url = init_baseline_project()
    print(f"\n\nSHAREABLE LINK: {url}")
