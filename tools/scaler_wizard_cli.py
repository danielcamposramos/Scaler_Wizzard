import click
import json
from components.pipeline.scaling_pipeline import ScalingPipeline

@click.command()
@click.option('--model', default='TRMC-7B-Recursive-MoE', help='Base model name')
@click.option('--use-case', default='trmc_special', help='Scaling profile')
def run_scaler(model, use_case):
    """Scaler Wizard CLI for one-shot model scaling."""
    click.echo(f"🧙‍♂️ Initializing Scaler Wizard for {model}...")
    
    config = {
        'model_name': model,
        'max_length': 32768,
        'epochs': 20
    }
    
    pipeline = ScalingPipeline(config)
    click.echo("🔥 Engine Engaged. Monitoring Logic Gap...")

if __name__ == "__main__":
    run_scaler()