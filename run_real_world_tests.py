#!/usr/bin/env python3
"""
Quick-Start Real-World Testing on Llama-2-7B
One command to validate entropy compression on production models
"""

import sys
import subprocess
import json
from pathlib import Path


def check_dependencies():
    """Check required packages"""
    print("Checking dependencies...")
    required = ["torch", "transformers", "numpy", "pandas"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  ✗ {pkg} - missing")
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_model_access():
    """Verify Hugging Face credentials"""
    print("\nChecking Hugging Face access...")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        print("  ✓ Can access Llama-2-7B")
        return True
    except Exception as e:
        print(f"  ✗ Cannot access Llama-2-7B: {e}")
        print("\nYou may need to:")
        print("  1. Accept license at: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("  2. Run: huggingface-cli login")
        print("  3. Paste your HF token")
        return False


def run_phase_1():
    """Phase 1: Foundation validation"""
    print("\n" + "="*80)
    print("PHASE 1: Foundation Validation on Llama-2-7B")
    print("="*80)
    print("Testing: Question Answering, Long-Context, Code Gen, Memory")
    print("Duration: ~30-45 minutes")
    
    result = subprocess.run(
        ["python3", "test_real_world.py", 
         "--model", "meta-llama/Llama-2-7b-hf",
         "--device", "cuda"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Phase 1 PASSED")
        return True
    else:
        print("\n✗ Phase 1 FAILED")
        return False


def run_phase_2():
    """Phase 2: Compression strategy comparison"""
    print("\n" + "="*80)
    print("PHASE 2: Compression Strategy Comparison")
    print("="*80)
    print("Testing: Different compression approaches on real tasks")
    print("Duration: ~15-20 minutes")
    
    result = subprocess.run(
        ["python3", "test_compression_strategies.py",
         "--model", "meta-llama/Llama-2-7b-hf",
         "--device", "cuda"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Phase 2 PASSED")
        return True
    else:
        print("\n✗ Phase 2 FAILED (note: benchmark might not be updated yet)")
        return False


def run_phase_3():
    """Phase 3: Enhanced benchmarking"""
    print("\n" + "="*80)
    print("PHASE 3: Extended Benchmarking")
    print("="*80)
    print("Testing: Full benchmark on Llama-2-7B (20 samples)")
    print("Duration: ~20-30 minutes")
    
    result = subprocess.run(
        ["python3", "run_benchmark.py",
         "--model", "meta-llama/Llama-2-7b-hf",
         "--device", "cuda",
         "--samples", "20"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Phase 3 PASSED")
        return True
    else:
        print("\n✗ Phase 3 FAILED (note: benchmark might not be updated yet)")
        return False


def run_phase_4():
    """Phase 4: Visualization and analysis"""
    print("\n" + "="*80)
    print("PHASE 4: Visualization and Analysis")
    print("="*80)
    print("Generating: Comparison plots and statistical summary")
    print("Duration: ~5 minutes")
    
    result = subprocess.run(
        ["python3", "visualize_benchmark.py"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Phase 4 PASSED")
        return True
    else:
        print("\n✗ Phase 4 FAILED (note: requires Phase 3 data)")
        return False


def summarize_results():
    """Summarize all results"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    results = {}
    files_to_check = [
        ("results_real_world.json", "Real World Tests"),
        ("results_strategy_comparison.json", "Strategy Comparison"),
        ("results_gpt2/benchmark_results.json", "Extended Benchmark"),
        ("results_gpt2/benchmark_comparison.png", "Visualization")
    ]
    
    for filepath, description in files_to_check:
        path = Path(filepath)
        if path.exists():
            print(f"  ✓ {description}: {filepath}")
            results[description] = str(path)
        else:
            print(f"  - {description}: {filepath} (not found)")
    
    print("\nNext Steps:")
    print("  1. Review results_real_world.json")
    print("  2. Check entropy compression performance")
    print("  3. Compare with GPT-2 baseline (1.4678 PPL)")
    print("  4. Create REAL_WORLD_VALIDATION_REPORT.md")
    
    return results


def main():
    print("\n" + "="*80)
    print("REAL-WORLD TESTING SUITE FOR ENTROPY KV CACHE COMPRESSION")
    print("="*80)
    print("Testing on: Llama-2-7B (production model)")
    print("Tasks: QA, Long-Context, Code Gen, Compression Comparison, Benchmarking")
    print("="*80)
    
    # Check setup
    if not check_dependencies():
        print("\n✗ Please install missing dependencies")
        return False
    
    if not check_model_access():
        print("\n✗ Please set up Hugging Face access")
        return False
    
    # Run phases
    phases = [
        ("Phase 1 (Foundation)", run_phase_1),
        ("Phase 2 (Strategy)", run_phase_2),
        ("Phase 3 (Benchmark)", run_phase_3),
        ("Phase 4 (Visualize)", run_phase_4),
    ]
    
    results = {}
    for phase_name, phase_func in phases:
        try:
            success = phase_func()
            results[phase_name] = "PASSED" if success else "FAILED"
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted at {phase_name}")
            results[phase_name] = "INTERRUPTED"
            break
        except Exception as e:
            print(f"\n✗ Error in {phase_name}: {e}")
            results[phase_name] = "ERROR"
    
    # Summary
    summarize_results()
    
    print("\n" + "="*80)
    print("PHASE RESULTS")
    print("="*80)
    for phase, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "⚠"
        print(f"  {symbol} {phase}: {status}")
    
    all_passed = all(s == "PASSED" for s in results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nYour entropy compression approach works on modern production models!")
        print("Next: Compare results with GPT-2 baseline")
        return True
    else:
        print("\n⚠ Some tests failed or were interrupted")
        print("Review errors above and retry")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)