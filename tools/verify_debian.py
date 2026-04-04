import shutil
import os
import sys
import psutil
import importlib.util

def verify_debian_readiness():
    print("🚀 Verifying Scaler Wizard readiness for Debian...")
    is_ready = True
    
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
    if free_gb < 150:
        print(f"⚠️ Disk space low: {free_gb:.1f}GB. Training 50 epochs on ground-truth data requires ~200GB+ for reliable checkpoint history.")
    else:
        print(f"✅ Disk space looks good: {free_gb:.1f}GB free.")

    # 4. Check RAM
    total_ram = psutil.virtual_memory().total / (1024**3)
    if total_ram < 16:
        print(f"⚠️ RAM is low: {total_ram:.1f}GB. MoE models benefit from 32GB+.")
    else:
        print(f"✅ RAM detected: {total_ram:.1f}GB.")

    # 5. Check dependencies without initializing them (prevents import order warnings)
    deps = ['unsloth', 'torch', 'datasets', 'transformers', 'peft', 'triton', 'trl', 'flash_attn', 'ninja']
    missing = []
    for dep in deps:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"👉 Please run: pip install {' '.join(missing)}")
        if 'flash_attn' in missing:
            print("⚠️ Note: flash-attn installation can take 15-30 minutes as it compiles CUDA kernels from source.")
        is_ready = False
    else:
        print(f"✅ All {len(deps)} core AI libraries are detected.")

    return is_ready

if __name__ == "__main__":
    verify_debian_readiness()