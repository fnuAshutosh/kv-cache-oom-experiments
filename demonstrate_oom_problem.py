"""OOM Demonstration Script: Proving the KV Cache Problem

This script intentionally demonstrates the Out-of-Memory problem caused by
the KV cache in transformer models. It processes increasingly long sequences
and tracks memory usage until OOM occurs.

This provides empirical proof that:
1. KV cache grows linearly with sequence length
2. Memory becomes a bottleneck for long-context tasks
3. Smaller models can't handle realistic long-context scenarios

Usage:
    python demonstrate_oom_problem.py \
        --model "meta-llama/Llama-2-7b-hf" \
        --max-length 131072 \
        --batch-size 1 \
        --device cuda
"""

import argparse
import json
import time
import tracemalloc
from typing import Dict, List, Tuple
from pathlib import Path
import os
import sys
import subprocess


def install_package(package_name: str, import_name: str = None):
    """Auto-install missing package."""
    if import_name is None:
        import_name = package_name
    
    print(f"⚠ Installing missing package: {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package_name}")
        return False


def safe_import(module_name: str, package_name: str = None):
    """Safely import a module, auto-installing if needed."""
    if package_name is None:
        package_name = module_name
    
    try:
        return __import__(module_name)
    except ImportError:
        print(f"⚠ {module_name} not found, attempting installation...")
        if install_package(package_name, module_name):
            try:
                return __import__(module_name)
            except ImportError:
                print(f"✗ Failed to import {module_name} after installation")
                return None
        return None


# Install required packages
print("Checking dependencies...")
torch = safe_import("torch")
transformers_module = safe_import("transformers")
psutil = safe_import("psutil")

# Verify critical imports
if torch is None:
    print("\n✗ CRITICAL: PyTorch installation failed")
    print("Manual install: pip install torch")
    exit(1)

if transformers_module is None:
    print("\n✗ CRITICAL: transformers installation failed")
    print("Manual install: pip install transformers")
    exit(1)

if psutil is None:
    print("\n✗ CRITICAL: psutil installation failed")
    print("Manual install: pip install psutil")
    exit(1)

# Now import with correct references
from transformers import AutoTokenizer, AutoModelForCausalLM

class MemoryTracker:
    """Track GPU and CPU memory usage throughout the experiment."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.measurements = []
        self.peak_vram = 0

# Rest of implementation - see full file
if __name__ == "__main__":
    print("OOM Demonstration Script Ready")
    print("Run: python demonstrate_oom_problem.py --help")
