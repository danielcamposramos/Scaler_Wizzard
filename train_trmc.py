#!/usr/bin/env python3
"""
Scaler Wizard: TRMC Main Training Script
Replicates the Jupyter Notebook logic for standalone execution on Debian.
"""

import os
import sys
import logging
from pathlib import Path
import platform
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

# Ensure the project root is in the python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from tools.verify_debian import verify_debian_readiness
from components.scaling_engine.unsloth_engine import UnslothEngine
from components.scaling.profile_recommender import ProfileRecommender
from components.scaling.dataset_curator import DatasetCurator

console = Console()

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("🧙‍♂️ Starting Scaler Wizard TRMC Training Pipeline")

    # 1. Environment Readiness Check
    logger.info("Verifying Debian system readiness...")
    verify_debian_readiness()

    # 2. Initialize Components
    recommender = ProfileRecommender()
    # Dataset cache directory
    cache_dir = project_root / "checkpoints" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    curator = DatasetCurator(cache_dir=str(cache_dir))

    # 3. Get hardware-aware recommendations for the 'trmc_special' Vibe
    # This vibe includes Wikipedia, NuminaMath, Chess, and ORPO data
    profile = recommender.recommend_profile(use_case="trmc_special", model_size="small")

    # Dashboard View
    console.print(Panel.fit(
        f"[bold cyan]TRMC Scaling Session[/bold cyan]\n"
        f"Device: [green]{profile.get('device', 'cpu')}[/green]\n"
        f"Unsloth Turbo: [yellow]{profile.get('use_unsloth', False)}[/yellow]\n"
        f"Epochs: [white]{profile.get('epochs', 5)}[/white]\n"
        f"OS: [magenta]{platform.system()}[/magenta]",
        title="Scaler Wizard Cockpit"
    ))

    # 4. Initialize the Unsloth Speed Engine
    # Base model selected for context and license compatibility
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info(f"Loading base model: {base_model}")
    
    try:
        with console.status("[bold green]Waking the beast (Loading Model)..."):
            engine = UnslothEngine(model_name=base_model, max_seq_length=4096)
            engine.apply_fast_lora(r=profile.get("lora_rank", 16))
            logger.info("✅ Model loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Unsloth Engine: {e}")
        sys.exit(1)

    # 5. Resume Logic: Detect last checkpoint
    output_dir = project_root / "outputs" / "trmc_final_run"
    last_checkpoint = None
    if output_dir.exists():
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            last_checkpoint = str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1])))
            console.print(f"[bold yellow]Found existing progress. Resuming from: {last_checkpoint}[/bold yellow]")

    # 5. Execute Linear Training Run (5 Epochs Mixture)
    training_args = {
        "epochs": profile.get("epochs", 5),
        "batch_size": 2,
        "learning_rate": 5e-6,
        "output_dir": str(output_dir),
        "resume_from_checkpoint": last_checkpoint
    }

    with console.status("[bold red]Training Active - Monitoring Quality & Logic..."):
        engine.run_pretraining(
            dataset_configs=profile.get("recommended_datasets", []),
            args_dict=training_args
        )
    console.print("[bold green]🎉 Training session completed successfully![/bold green]")

if __name__ == "__main__":
    main()