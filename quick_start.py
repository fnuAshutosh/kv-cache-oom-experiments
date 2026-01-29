```python
"""
Quick Start Guide: Running Your First Experiment

This script demonstrates how to use the EntropyMergedKVCache framework
in a simple, step-by-step manner.

Run: python quick_start.py
"""

import torch
import json
from pathlib import Path

# Import our modules
from entropy_merged_kv_cache import EntropyMergedKVCache


def quick_demo():
    """
    Minimal demonstration of the entropy-guided KV cache merging.
    """
    print("=" * 80)
    print("ENTROPY-GUIDED KV CACHE MERGING: QUICK START")
    print("=" * 80)
    print()
    
    # ========== STEP 1: Create Dummy Data ==========
    print("[STEP 1] Creating dummy KV cache tensors...")
    
    batch_size = 1
    seq_len = 128
    hidden_dim = 768
    num_layers = 32
    
    # Create dummy attention scores
    dummy_attention = torch.randn(batch_size, 32, 1, seq_len)
    dummy_attention = torch.softmax(dummy_attention, dim=-1)
    
    # Create dummy KV cache
    past_key_values = []
    for _ in range(num_layers):
        key = torch.randn(batch_size, seq_len, hidden_dim)
        value = torch.randn(batch_size, seq_len, hidden_dim)
        past_key_values.append((key, value))
    
    past_key_values = tuple(past_key_values)
    
    print(f"  ✓ Created KV cache with {len(past_key_values)} layers")
    print(f"  ✓ Each layer: Key {past_key_values[0][0].shape}, Value {past_key_values[0][1].shape}")
    print()
    
    # ========== STEP 2: Test Entropy Calculation ==========
    print("[STEP 2] Testing entropy calculation...")
    
    cache_mgr = EntropyMergedKVCache(entropy_threshold=0.5)
    entropy = cache_mgr.calculate_entropy(dummy_attention)
    
    entropy_val = entropy.item() if entropy.numel() == 1 else entropy.mean().item()
    print(f"  ✓ Computed entropy: {entropy_val:.4f} bits")
    print()
    
    # ========== STEP 3: Apply Compression ==========
    print("[STEP 3] Applying entropy merging compression...")
    
    result_entropy = cache_mgr.compress_kv_cache(
        past_key_values,
        attention_scores=[dummy_attention] * num_layers
    )
    
    print("  ✓ Compression applied")
    print()
    
    # ========== STEP 4: Analyze Results ==========
    print("[STEP 4] Compression results...")
    
    original_size = sum(k.numel() + v.numel() for k, v in past_key_values)
    compressed_size = sum(k.numel() + v.numel() for k, v in result_entropy)
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"  Original size: {original_size:,} elements")
    print(f"  Compressed size: {compressed_size:,} elements")
    print(f"  Compression ratio: {compression_ratio:.1f}%")
    print()
    
    # ========== STEP 5: Statistics ==========
    print("[STEP 5] Entropy merging statistics...")
    
    stats = cache_mgr.get_compression_stats()
    if "message" not in stats:
        print(f"  Total merges: {stats['total_merges']}")
        print(f"  Avg compression ratio: {stats['avg_compression_ratio']*100:.1f}%")
        print(f"  Total tokens merged: {stats['total_tokens_merged']}")
    print()
    
    print("=" * 80)
    print("QUICK START COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review RESEARCH_REPORT.md for full methodology")
    print("  2. Read entropy_merged_kv_cache.py for algorithm details")
    print("  3. Run main_experiment.py for full benchmark on real models")
    print("  4. Check README.md for configuration options")
    print()


if __name__ == "__main__":
    print("\n")
    try:
        quick_demo()
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("\nInstall dependencies with:")
        print("  pip install torch transformers matplotlib numpy")
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nFor troubleshooting, see README.md")
```