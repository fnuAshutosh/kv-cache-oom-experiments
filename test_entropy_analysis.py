```python
"""
Entropy Analysis Test: Validate entropy calculation on real model attention patterns
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from entropy_merged_kv_cache import EntropyMergedKVCache


def test_entropy_with_real_attention():
    """Test entropy calculation with real model attention patterns"""
    
    print("="*80)
    print("ENTROPY ANALYSIS: TESTING WITH REAL GPT-2 ATTENTION PATTERNS")
    print("="*80)
    print()
    
    # Load model
    print("[1] Loading GPT-2 model...")
    device = "cpu"
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded: {model_name}")
    print(f"  - Layers: {model.config.n_layer}")
    print(f"  - Hidden size: {model.config.hidden_size}")
    print()
    
    # Run inference
    print("[2] Running inference on sample text...")
    sample_text = "The quick brown fox jumps over the lazy dog. The cat watches from the fence."
    inputs = tokenizer(sample_text, return_tensors="pt", max_length=128, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    seq_len = input_ids.shape[1]
    
    print(f"  Sequence length: {seq_len} tokens")
    
    # Get attention
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions
    
    print(f"  Extracted attention from {len(attentions)} layers")
    print()
    
    # Analyze entropy
    print("[3] Entropy Analysis Results")
    print("-" * 80)
    
    cache_mgr = EntropyMergedKVCache()
    entropies = []
    
    for layer_idx, attention in enumerate(attentions):
        # attention shape: (batch, heads, seq_len, seq_len)
        avg_attention = attention[0].mean(dim=0)  # Average across heads
        
        for q_idx in range(seq_len):
            attn_weights = avg_attention[q_idx, :seq_len]
            attn_weights = attn_weights / attn_weights.sum()
            attn_weights_clipped = torch.clamp(attn_weights, min=1e-7)
            entropy = -(attn_weights * torch.log(attn_weights_clipped)).sum().item()
            entropies.append(entropy)
    
    print(f"\nEntropy Statistics:")
    print(f"  Mean entropy: {np.mean(entropies):.4f} bits")
    print(f"  Min entropy: {np.min(entropies):.4f} bits")
    print(f"  Max entropy: {np.max(entropies):.4f} bits")
    print()
    
    print("="*80)
    print("✓ ENTROPY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_entropy_with_real_attention()
```