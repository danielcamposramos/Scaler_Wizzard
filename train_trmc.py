#!/usr/bin/env python3
"""
Scaler Wizard: TRMC Main Training Script
Replicates the Jupyter Notebook logic for standalone execution on Debian.
"""

import os
import sys
import shutil
import logging
import time
from pathlib import Path
import torch
import platform
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live

# --- VRAM SAFETY PROTOCOLS ---
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the project root is in the python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from tools.verify_debian import verify_debian_readiness

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

    # 0. Safety Guard: Ensure we aren't running a stale copy in research/
    if project_root.name == "research":
        console.print("[bold red]❌ CRITICAL ERROR: You are running train_trmc.py from the 'research' folder.[/bold red]")
        console.print("[yellow]Please delete 'research/train_trmc.py' and run the root script instead:[/yellow]")
        console.print("[white]cd .. && python3 train_trmc.py[/white]")
        sys.exit(1)

    # 1. Environment Readiness Check
    logger.info("Verifying Debian system readiness...")
    if not verify_debian_readiness():
        logger.error("Environment verification failed. Please resolve the missing dependencies above.")
        sys.exit(1)

    # 2. Initialize Components (Imported after readiness check)
    from components.scaling.profile_recommender import ProfileRecommender
    from components.scaling.dataset_curator import DatasetCurator

    # 2. Initialize Components
    recommender = ProfileRecommender()
    # Dataset cache directory
    cache_dir = Path("/home/daniel/TRMC")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    curator = DatasetCurator(cache_dir=str(cache_dir))

    # 3. Get hardware-aware recommendations for the 'trmc_special' Vibe
    # This vibe includes Wikipedia, NuminaMath, Chess, and ORPO data
    profile = recommender.recommend_profile(use_case="trmc_special", model_size="small")

    # 4. Defer import until after readiness check
    from components.scaling_engine.unsloth_engine import UnslothEngine
    from unsloth import is_bfloat16_supported # type: ignore

    # Dashboard View
    is_cuda = torch.cuda.is_available()
    console.print(Panel.fit(
        f"[bold cyan]TRMC-7B NATIVE CORE IGNITION: Recursive MoE Training[/bold cyan]\n"
        f"Vibe: [white]Two-Way Contrastive Logic (True vs False)[/white]\n"
        f"Features: [white]Recursive Core, MoE, Matryoshka, Contrastive Saw[/white]\n"
        f"Device: [green]{'cuda (RTX 3060)' if is_cuda else 'cpu'}[/green]\n"
        f"Unsloth Turbo: [yellow]{'True (Fast Kernels Engaged)' if is_cuda else 'False'}[/yellow]\n"
        f"Target Epochs: [white]20 (High-Saturation reasoning)[/white]\n"
        f"Metrics: [bold green]Monitoring Logic Gap (Truth vs False)[/bold green]",
        title="Scaler Wizard Cockpit"
    ))

    # 4. Initialize the Unsloth Speed Engine
    trmc_spec = "TRMC-7B-Recursive-MoE"
    logger.info(f"🔨 BUILDING NATIVE TRMC CORE: {trmc_spec}...")
    
    try:
        # Building for Daniel's 32k context target
        target_context = 32768 
        with console.status(f"[bold green]Synthesizing {trmc_spec} with {target_context} context..."):
            engine = UnslothEngine(model_name=trmc_spec, max_seq_length=target_context)
            engine.apply_fast_lora(r=profile.get("lora_rank", 16))
            logger.info("✅ Model loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Unsloth Engine: {e}")
        sys.exit(1)

    # 5. Resume Logic: Detect last checkpoint
    output_dir = project_root / "outputs" / "trmc_final_run"
    output_dir.parent.mkdir(exist_ok=True)
    last_checkpoint = None
    
    if output_dir.exists():
        existing_content = [p.name for p in output_dir.iterdir()]
        if existing_content:
            console.print(Panel(
                "\n".join([f"• {item}" for item in existing_content[:10]]),
                title="[bold yellow]Existing Attempt Found[/bold yellow]",
                subtitle=f"outputs/{output_dir.name}"
            ))
            ans = console.input("[bold cyan]What would you like to do? (d)elete, (o)verwrite, (a)longside, (r)esume? [/bold cyan]").lower()
            
            if ans == 'd':
                shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
            elif ans == 'a':
                output_dir = project_root / "outputs" / f"trmc_run_{int(time.time())}"
                output_dir.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]New path created: {output_dir.name}[/green]")
            elif ans == 'r':
                checkpoints = list(output_dir.glob("checkpoint-*"))
                if checkpoints:
                    last_checkpoint = str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1])))
                    console.print(f"[bold yellow]Resuming from: {last_checkpoint}[/bold yellow]")
            # 'o' (overwrite) choice clears last_checkpoint and proceeds fresh in the same folder

    # 5. Execute Linear Training Run (50 Epochs Mixture)
    training_args = {
        "epochs": int(profile.get("epochs", 20)),
        "batch_size": 1, # Reduced to 1 for 7B @ 32k context on 12GB VRAM
        "gradient_accumulation_steps": 16, # Increased to maintain stable gradients
        "learning_rate": 5e-6,
        "max_length": 32768,
        "output_dir": str(output_dir),
        "resume_from_checkpoint": last_checkpoint
    }

    with console.status("[bold red]Training Active - Ground-Truth Data Stream Engaged..."):
        engine.run_pretraining(
            dataset_configs=profile.get("recommended_datasets", []),
            args_dict=training_args
        )
    console.print("[bold green]🎉 Training session completed successfully![/bold green]")

if __name__ == "__main__":
    main()