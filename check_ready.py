#!/usr/bin/env python3
"""
Pre-Flight Checklist for Real-World Testing
Verify everything is ready before running production tests
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Verify Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor} (good)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.7+)")
        return False


def check_virtual_env():
    """Check if in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✓ Virtual environment active")
        return True
    else:
        print(f"⚠ Not in virtual environment")
        print(f"  → Run: source .venv/bin/activate")
        return False


def check_packages():
    """Check required packages"""
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "json": "JSON (built-in)",
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - missing")
            missing.append(package)
    
    return len(missing) == 0


def check_files():
    """Check if all test files exist"""
    files = [
        "test_real_world.py",
        "test_compression_strategies.py",
        "compare_real_world_results.py",
        "run_real_world_tests.py",
        "entropy_merged_kv_cache.py",
        "baselines.py",
    ]
    
    missing = []
    for file in files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - missing")
            missing.append(file)
    
    return len(missing) == 0


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {device_name} ({memory:.1f} GB VRAM)")
            return True
        else:
            print(f"⚠ GPU not available")
            print(f"  → Can use CPU instead: --device cpu")
            return False
    except Exception as e:
        print(f"⚠ Cannot check GPU: {e}")
        return False


def check_storage():
    """Check available storage"""
    try:
        import shutil
        stat = shutil.disk_usage("/")
        free_gb = stat.free / 1e9
        
        if free_gb > 30:
            print(f"✓ Storage: {free_gb:.1f} GB free (good)")
            return True
        elif free_gb > 20:
            print(f"⚠ Storage: {free_gb:.1f} GB free (tight, but should work)")
            print(f"  → Model weights: ~13GB")
            print(f"  → Results: ~1GB")
            return True
        else:
            print(f"✗ Storage: {free_gb:.1f} GB free (not enough)")
            return False
    except Exception as e:
        print(f"⚠ Cannot check storage: {e}")
        return True


def check_huggingface_access():
    """Check Hugging Face access"""
    try:
        from transformers import AutoTokenizer
        print("✓ Testing HuggingFace access...", end=" ")
        
        # Try a fast model first (GPT-2)
        try:
            AutoTokenizer.from_pretrained("gpt2")
            print("(GPT-2 ✓)")
            return True
        except Exception as e:
            print(f"(Error: {e})")
            print(f"\n  → You may need to authenticate:")
            print(f"     huggingface-cli login")
            return False
    except Exception as e:
        print(f"✗ Cannot check HuggingFace: {e}")
        return False


def check_directory():
    """Check working directory"""
    expected_files = ["entropy_merged_kv_cache.py", "baselines.py"]
    
    cwd = Path.cwd()
    print(f"Current directory: {cwd}")
    
    found = all(Path(f).exists() for f in expected_files)
    if found:
        print("✓ Correct directory (has core files)")
        return True
    else:
        print("✗ Wrong directory (missing core files)")
        print(f"  → Expected to be in: /Users/ashu/Projects/KV_caching_experiments")
        return False


def print_header(title):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def print_summary(checks):
    """Print summary of all checks"""
    passed = sum(1 for c in checks.values() if c)
    total = len(checks)
    
    print_header(f"Summary: {passed}/{total} checks passed")
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    if passed == total:
        print("\n✓ All checks passed! You're ready to run tests.")
        return True
    else:
        print(f"\n⚠ {total - passed} check(s) failed or skipped.")
        print("Review issues above and fix if needed.")
        return False


def main():
    print_header("Pre-Flight Checklist for Real-World Testing")
    
    print("This checklist verifies your system is ready to run")
    print("the entropy compression tests on production models.")
    print("")
    
    # Run all checks
    print("1. Python Environment")
    check_1 = check_python_version()
    check_2 = check_virtual_env()
    
    print("\n2. Required Packages")
    check_3 = check_packages()
    
    print("\n3. Project Files")
    check_4 = check_files()
    
    print("\n4. Hardware")
    check_5 = check_gpu()
    
    print("\n5. Storage")
    check_6 = check_storage()
    
    print("\n6. Hugging Face Access")
    check_7 = check_huggingface_access()
    
    print("\n7. Working Directory")
    check_8 = check_directory()
    
    # Summary
    checks = {
        "Python version": check_1,
        "Virtual environment": check_2,
        "Required packages": check_3,
        "Project files": check_4,
        "GPU available": check_5,
        "Storage space": check_6,
        "HuggingFace access": check_7,
        "Working directory": check_8,
    }
    
    ready = print_summary(checks)
    
    # Next steps
    if ready:
        print_header("Next Steps")
        print("Run Phase 1 Testing:")
        print("")
        print("python3 test_real_world.py --model meta-llama/Llama-2-7b-hf --device cuda")
        print("")
        print("Expected duration: 45 minutes")
        print("Output: results_real_world.json")
        return True
    else:
        print_header("Fix Issues")
        print("Common solutions:")
        print("")
        print("1. Not in virtual environment?")
        print("   → source .venv/bin/activate")
        print("")
        print("2. Missing packages?")
        print("   → pip install -r requirements.txt")
        print("")
        print("3. Wrong directory?")
        print("   → cd /Users/ashu/Projects/KV_caching_experiments")
        print("")
        print("4. HuggingFace access?")
        print("   → huggingface-cli login")
        print("")
        print("5. Need more storage?")
        print("   → Free up space and try again")
        print("")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)