#!/bin/bash
# Scaler Wizard: Clean Environment Setup
echo "🧹 Creating isolated TRMC environment..."

conda create -n trmc_core python=3.11 -y
source activate trmc_core

echo "🚀 Installing 2026-grade AI stack..."
pip install torch>=2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install ninja
pip install unsloth trl>=0.13.0 transformers>=4.58.0 datasets peft accelerate bitsandbytes flash-attn --no-build-isolation
pip install rich GPUtil psutil pyyaml jsonschema Pillow numpy pynput click

echo "✅ Environment 'trmc_core' is ready."
echo "👉 Run: conda activate trmc_core && python3 train_trmc.py"