import shutil
import os
import sys
import psutil

def verify_debian_readiness():
    print("🚀 Verifying Scaler Wizard readiness for Debian...")
    
    # 1. Check Python and OS
    if not sys.platform.startswith('linux'):
        print("❌ Warning: Not running on Linux. This project is optimized for Debian/SparkyLinux.")
    
    # 2. Check for CUDA/NVIDIA
    has_nvidia = shutil.which('nvidia-smi') is not None
    if has_nvidia:
        print("✅ NVIDIA Drivers detected.")
    else:
        print("⚠️ No NVIDIA Drivers found. Training will be extremely slow on CPU.")

    # 3. Check Disk Space (Crucial for Wikipedia/Math datasets)
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < 100:
        print(f"⚠️ Disk space low: {free_gb:.1f}GB. Training 5 epochs on real-world data requires ~150GB for checkpoints.")
    else:
        print(f"✅ Disk space looks good: {free_gb:.1f}GB free.")

    # 4. Check RAM
    total_ram = psutil.virtual_memory().total / (1024**3)
    if total_ram < 16:
        print(f"⚠️ RAM is low: {total_ram:.1f}GB. MoE models benefit from 32GB+.")
    else:
        print(f"✅ RAM detected: {total_ram:.1f}GB.")

    # 5. Check dependencies
    try:
        import datasets, transformers, peft, triton, torch
        print("✅ Core AI Libraries (datasets, transformers, peft, triton) are installed.")
        
        try:
            import unsloth
            print("✅ Unsloth Speed-Training library detected.")
        except ImportError:
            print("⚠️ Unsloth not found. Speed training disabled. Install via: pip install unsloth")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}. Please run 'pip install datasets transformers peft triton'")

if __name__ == "__main__":
    verify_debian_readiness()