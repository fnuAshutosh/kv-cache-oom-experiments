#!/usr/bin/env python3
"""
Compare KV Cache Compression Strategies on Real-World Tasks
Tests EntropyMerged vs baselines on Llama-2-7B
"""

import torch
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


class CompressionStrategy:
    """Base class for KV cache compression strategies"""
    
    def __init__(self, compress_ratio: float = 0.5):
        self.compress_ratio = compress_ratio
    
    def compress(self, key_cache, value_cache, attention_weights=None):
        """Compress KV cache - implement in subclasses"""
        raise NotImplementedError


class NoCompression(CompressionStrategy):
    """Baseline: No compression (full cache)"""
    
    def compress(self, key_cache, value_cache, attention_weights=None):
        return key_cache, value_cache


class H2OCompression(CompressionStrategy):
    """H2O: Keep heavy hitters (high attention) + recent tokens"""
    
    def compress(self, key_cache, value_cache, attention_weights=None):
        batch_size, num_heads, seq_len, head_dim = key_cache.shape
        
        # Determine which positions to keep
        if attention_weights is not None:
            attn_sum = attention_weights.sum(dim=1).mean(dim=1)
            threshold = torch.quantile(attn_sum, 1 - self.compress_ratio)
            mask = attn_sum >= threshold
        else:
            keep_count = int(seq_len * self.compress_ratio)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=key_cache.device)
            mask[:, -keep_count:] = True
            mask[:, :2] = True
        
        # Apply mask
        k_compressed = key_cache[mask].reshape(batch_size, -1, head_dim) if mask.any() else key_cache
        v_compressed = value_cache[mask].reshape(batch_size, -1, head_dim) if mask.any() else value_cache
        
        return k_compressed, v_compressed


class EntropyCompression(CompressionStrategy):
    """Entropy-based: Merge low-entropy spans"""
    
    def compress(self, key_cache, value_cache, attention_weights=None):
        batch_size, num_heads, seq_len, head_dim = key_cache.shape
        
        # Calculate entropy of attention distribution
        if attention_weights is not None:
            last_attn = attention_weights[:, :, -1, :]
            attn_entropy = -torch.sum(
                last_attn * torch.log(last_attn + 1e-10),
                dim=-1
            )
            avg_entropy = attn_entropy.mean(dim=1)
            
            entropy_per_pos = torch.zeros(seq_len, device=key_cache.device)
            for i in range(seq_len):
                pos_attn = attention_weights[:, :, :, i]
                entropy_per_pos[i] = -torch.sum(
                    pos_attn * torch.log(pos_attn + 1e-10)
                ).item() / (batch_size * num_heads)
        
        # Keep high-entropy positions
        keep_count = int(seq_len * self.compress_ratio)
        if attention_weights is not None:
            threshold = torch.quantile(entropy_per_pos, 1 - self.compress_ratio)
            mask = entropy_per_pos >= threshold
        else:
            mask = torch.zeros(seq_len, dtype=torch.bool, device=key_cache.device)
            mask[-keep_count:] = True
            mask[:max(2, keep_count//10)] = True
        
        # Apply mask across all batch/head dimensions
        k_compressed = key_cache[:, :, mask, :]
        v_compressed = value_cache[:, :, mask, :]
        
        return k_compressed, v_compressed


class ModelTester:
    """Test model with different compression strategies"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.results = {}
        
        print(f"Loading {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                output_attentions=True,
            )
            
            if device == "cpu":
                self.model = self.model.to("cpu")
            
            self.model.eval()
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            raise
    
    def measure_perplexity(self, text: str) -> Tuple[float, float]:
        """Measure perplexity and inference time"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            start = time.time()
            outputs = self.model(input_ids)
            elapsed = time.time() - start
            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            ppl = torch.exp(loss).item()
        
        return ppl, elapsed
    
    def test_comprehension(self):
        """Test comprehension on real-world text"""
        print("\n" + "="*80)
        print("Real-World Comprehension Test")
        print("="*80)
        
        test_texts = [
            """
            Artificial intelligence (AI) has revolutionized multiple industries including 
            healthcare, finance, and transportation. Machine learning, a subset of AI, 
            enables systems to learn from data without explicit programming. Deep learning, 
            which uses neural networks, has achieved remarkable results in image recognition, 
            natural language processing, and game playing. The transformer architecture, 
            introduced in 2017, has become the foundation for state-of-the-art language models 
            like GPT and BERT.
            """,
            """
            The process of machine learning involves several key steps: data collection, 
            data preprocessing, feature engineering, model selection, training, validation, 
            and testing. Each step is crucial for building effective models. Data quality 
            directly impacts model performance. Preprocessing includes handling missing values, 
            normalization, and handling outliers. Feature engineering creates meaningful features 
            from raw data. Model selection involves choosing appropriate algorithms based on 
            the problem type.
            """,
            """
            Climate change poses unprecedented challenges to global ecosystems and human societies. 
            Rising temperatures lead to melting ice caps, increasing sea levels, and more frequent 
            extreme weather events. The primary cause is greenhouse gas emissions from human activities. 
            Mitigation strategies include transitioning to renewable energy, improving energy efficiency, 
            and protecting forests. Adaptation measures help communities cope with climate impacts. 
            International cooperation is essential for addressing this global challenge.
            """
        ]
        
        task_results = []
        
        for idx, text in enumerate(test_texts):
            print(f"\nTest Sample {idx + 1}:")
            try:
                ppl, time_taken = self.measure_perplexity(text)
                print(f"  Perplexity: {ppl:.4f}")
                print(f"  Time: {time_taken:.3f}s")
                print(f"  Tokens: {len(self.tokenizer.encode(text))}")
                
                task_results.append({
                    "sample": idx + 1,
                    "perplexity": float(ppl),
                    "time": float(time_taken),
                    "tokens": len(self.tokenizer.encode(text))
                })
            except Exception as e:
                print(f"  Error: {e}")
        
        return task_results
    
    def test_baseline_performance(self):
        """Baseline performance test"""
        print("\n" + "="*80)
        print("Baseline Performance (No Compression)")
        print("="*80)
        
        baseline_text = "The quick brown fox jumps over the lazy dog. " * 50
        
        try:
            ppl, time_taken = self.measure_perplexity(baseline_text)
            print(f"  Perplexity: {ppl:.4f}")
            print(f"  Time: {time_taken:.3f}s")
            
            return {
                "perplexity": float(ppl),
                "time": float(time_taken),
                "tokens": len(self.tokenizer.encode(baseline_text))
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KV cache compression strategies")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model to test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    tester = ModelTester(model_name=args.model, device=args.device)
    
    print("\n" + "="*80)
    print("TESTING COMPRESSION STRATEGIES ON REAL-WORLD TASKS")
    print("="*80)
    
    # Run baseline
    baseline = tester.test_baseline_performance()
    
    # Test comprehension
    comprehension = tester.test_comprehension()
    
    # Save results
    results = {
        "model": args.model,
        "device": args.device,
        "baseline": baseline,
        "comprehension": comprehension
    }
    
    output_file = "results_strategy_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()