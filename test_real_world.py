#!/usr/bin/env python3
"""
Real-World Test Suite: Llama-2-7B on Production Tasks
Test entropy-guided compression on current production models
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class RealWorldBenchmark:
    """Benchmark on production tasks with real-world scenarios"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.results = {"model": model_name, "device": device, "tasks": {}}
        
        print(f"Loading {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
            print(f"✓ Loaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print(f"  → Install with: pip install transformers torch accelerate")
            raise
    
    def test_question_answering(self):
        """Test on Question Answering task"""
        print("\n" + "="*80)
        print("TEST 1: Question Answering (SQuAD-style)")
        print("="*80)
        
        qa_examples = [
            {
                "context": "Super Bowl 50 was an American football game between the Denver Broncos and Carolina Panthers to determine the National Football League (NFL) champion for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), branding it simply as \"Super Bowl 50\".",
                "question": "Which NFL team won Super Bowl 50?",
                "expected": "Denver Broncos"
            },
            {
                "context": "The Amazon rainforest is a moist broadleaf tropical forest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,050,000 km2 (2,722,000 sq mi), of which 5,500,000 km2 (2,124,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with the remaining 17% split among Bolivia, Ecuador, French Guiana, Guyana, Suriname, and Venezuela.",
                "question": "What percentage of the Amazon rainforest is in Brazil?",
                "expected": "60%"
            }
        ]
        
        task_results = {"examples": []}
        
        for idx, example in enumerate(qa_examples):
            prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(self.device)
            
            print(f"\nExample {idx + 1}:")
            print(f"  Context: {example['context'][:100]}...")
            print(f"  Question: {example['question']}")
            
            with torch.no_grad():
                start = time.time()
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=False
                )
                elapsed = time.time() - start
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"  Answer: {answer[-50:]}")
            print(f"  Expected: {example['expected']}")
            print(f"  Time: {elapsed:.2f}s")
            
            task_results["examples"].append({
                "question": example["question"],
                "expected": example["expected"],
                "time": elapsed,
                "context_length": input_ids.shape[1]
            })
        
        self.results["tasks"]["qa"] = task_results
        return task_results
    
    def test_long_context(self, length: int = 4000):
        """Test on long context (where KV cache compression matters most)"""
        print("\n" + "="*80)
        print(f"TEST 2: Long-Context Processing ({length} tokens)")
        print("="*80)
        
        long_text = """
        Machine learning is a subset of artificial intelligence (AI) that enables systems to learn 
        and improve from experience without being explicitly programmed. It focuses on developing algorithms 
        and statistical models that allow computers to recognize patterns, make decisions, and predict outcomes 
        based on data.
        """ * (length // 60)
        
        inputs = self.tokenizer(long_text, return_tensors="pt", truncation=True, max_length=length)
        input_ids = inputs["input_ids"].to(self.device)
        
        print(f"  Context length: {input_ids.shape[1]} tokens")
        
        # Measure perplexity
        with torch.no_grad():
            start = time.time()
            outputs = self.model(input_ids, return_dict=True)
            elapsed = time.time() - start
            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            ppl = torch.exp(loss).item()
        
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens/sec: {input_ids.shape[1] / elapsed:.0f}")
        
        result = {
            "length": input_ids.shape[1],
            "perplexity": float(ppl),
            "time": elapsed,
            "throughput": input_ids.shape[1] / elapsed
        }
        
        self.results["tasks"]["long_context"] = result
        return result
    
    def test_code_generation(self):
        """Test on code generation (HumanEval-style)"""
        print("\n" + "="*80)
        print("TEST 3: Code Generation")
        print("="*80)
        
        code_prompts = [
            {
                "prompt": "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
                "task": "Fibonacci"
            },
            {
                "prompt": "def is_prime(n: int) -> bool:\n    \"\"\"Check if a number is prime.\"\"\"\n",
                "task": "Prime check"
            }
        ]
        
        task_results = {"examples": []}
        
        for example in code_prompts:
            print(f"\n  Task: {example['task']}")
            print(f"  Prompt: {example['prompt']}")
            
            inputs = self.tokenizer(example["prompt"], return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            with torch.no_grad():
                start = time.time()
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=False
                )
                elapsed = time.time() - start
            
            code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = code[len(example["prompt"]):]
            print(f"  Generated: {generated[:100]}...")
            print(f"  Time: {elapsed:.2f}s")
            
            task_results["examples"].append({
                "task": example["task"],
                "time": elapsed,
                "length": input_ids.shape[1]
            })
        
        self.results["tasks"]["code_gen"] = task_results
        return task_results
    
    def test_memory_usage(self):
        """Measure actual memory usage"""
        print("\n" + "="*80)
        print("TEST 4: Memory Usage Analysis")
        print("="*80)
        
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            long_text = "Machine learning " * 500
            inputs = self.tokenizer(long_text, return_tensors="pt", truncation=True, max_length=2000)
            input_ids = inputs["input_ids"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Peak VRAM: {peak_memory:.2f} GB")
            
            return {"peak_memory_gb": peak_memory}
        else:
            print("  (Memory profiling not available on CPU)")
            return None
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print(f"REAL-WORLD TESTING SUITE: {self.model_name}")
        print(f"Device: {self.device}")
        print("="*80)
        
        try:
            self.test_question_answering()
        except Exception as e:
            print(f"  ✗ QA test failed: {e}")
        
        try:
            self.test_long_context()
        except Exception as e:
            print(f"  ✗ Long context test failed: {e}")
        
        try:
            self.test_code_generation()
        except Exception as e:
            print(f"  ✗ Code generation test failed: {e}")
        
        try:
            self.test_memory_usage()
        except Exception as e:
            print(f"  ✗ Memory test failed: {e}")
        
        # Save results
        output_file = "results_real_world.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*80)
        print(f"Results saved to {output_file}")
        print("="*80)
        
        return self.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-world testing on production models")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name (Llama-2-7b-hf, meta-llama/Llama-2-7b-hf, etc.)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    benchmark = RealWorldBenchmark(model_name=args.model, device=args.device)
    benchmark.run_all_tests()