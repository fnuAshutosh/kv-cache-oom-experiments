```python
"""
Main Experiment Script: Entropy-Guided KV Cache Merging

This script runs the complete experimental pipeline for evaluating the
EntropyMergedKVCache strategy against established baselines.

Workflow:
1. Initialize models and data
2. Run baseline experiments (Full, H2O, StreamingLLM)
3. Run EntropyMergedKVCache experiments
4. Generate comparative analysis and visualizations
5. Produce final research report

Usage:
    python main_experiment.py --model llama-2-7b --dataset pg19 --output results/

Requirements:
    - torch >= 2.0
    - transformers >= 4.30
    - datasets (for PG-19)
    - matplotlib, numpy
"""

import argparse
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Import custom modules
from entropy_merged_kv_cache import EntropyMergedKVCache
from baselines import get_baseline_strategy, BASELINE_STRATEGIES
from evaluation import ExperimentalHarness, EvaluationMetrics
from visualization import (
    plot_entropy_heatmap,
    plot_pareto_frontier,
    plot_compression_analysis,
    plot_token_importance,
    generate_comparison_table,
)


class ExperimentRunner:
    """
    Main experiment orchestration class.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        dataset_name: str = "pg19",
        output_dir: str = "./results",
        device: str = "cuda",
    ):
        """
        Initialize the experiment runner.
        
        Args:
            model_name: HF model ID (e.g., "meta-llama/Llama-2-7b-hf" or "mistralai/Mistral-7B-v0.1").
            dataset_name: Dataset to use for evaluation (currently supports "pg19").
            output_dir: Directory to save results and visualizations.
            device: Device to run inference on ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.eval_data = None
        self.results: List[EvaluationMetrics] = []
    
    def load_model_and_tokenizer(self):
        """
        Load the specified model and tokenizer from Hugging Face.
        
        This method will download the model if not already cached.
        """
        print(f"Loading model: {self.model_name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("ERROR: transformers library not installed.")
            print("Install with: pip install transformers torch")
            return False
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Determine device_map based on device type
            if self.device == "cpu":
                device_map = None
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "trust_remote_code": True,
                }
            else:
                device_map = "auto"
                model_kwargs = {
                    "device_map": device_map,
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
            return False
    
    def load_dataset(self):
        """
        Load the evaluation dataset (PG-19).
        
        PG-19 provides long-context texts from Project Gutenberg,
        ideal for evaluating KV cache compression.
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: datasets library not installed.")
            print("Install with: pip install datasets")
            return False
        
        try:
            if self.dataset_name == "pg19":
                dataset = load_dataset("pg19", split="validation")
                # Sample a subset for efficiency
                self.eval_data = [text[:4000] for text in dataset["text"][:20]]
                print(f"✓ Loaded {len(self.eval_data)} text samples from PG-19")
            else:
                print(f"ERROR: Dataset '{self.dataset_name}' not supported.")
                return False
            return True
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {e}")
            return False
    
    def setup_strategies(self) -> Dict[str, object]:
        """
        Instantiate all compression strategies for evaluation.
        
        Returns:
            Dictionary mapping strategy names to instantiated strategy objects.
        """
        strategies = {}
        
        # Baseline 1: Full Cache (Oracle)
        strategies["full_cache"] = get_baseline_strategy("full")
        
        # Baseline 2: H2O (Heavy Hitter Oracle)
        strategies["h2o"] = get_baseline_strategy("h2o", cache_budget_ratio=0.2)
        
        # Baseline 3: StreamingLLM (Sinks + Window)
        strategies["streaming_llm"] = get_baseline_strategy("streaming", num_sinks=4, window_size=1024)
        
        # Novel Strategy: EntropyMergedKVCache
        strategies["entropy_merged"] = EntropyMergedKVCache(
            entropy_threshold=0.5,
            num_sinks=4,
            enable_merging=True,
        )
        
        print(f"✓ Initialized {len(strategies)} compression strategies")
        return strategies
    
    def run_experiments(self, strategies: Dict[str, object]):
        """
        Run the full experimental evaluation.
        
        Args:
            strategies: Dictionary of strategy instances.
        """
        if not self.model or not self.eval_data:
            print("ERROR: Model and dataset not loaded. Call load_model_and_tokenizer() and load_dataset().")
            return
        
        print("\n" + "=" * 120)
        print("STARTING EXPERIMENTAL EVALUATION")
        print("=" * 120 + "\n")
        
        # Create evaluation harness
        harness = ExperimentalHarness(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        # Run benchmark
        harness.run_benchmark(
            strategies=strategies,
            eval_texts=self.eval_data,
            context_length=2048,
        )
        
        self.results = harness.results
        
        # Save results to JSON
        results_json = self.output_dir / "results.json"
        with open(results_json, 'w') as f:
            json.dump(
                [
                    {
                        "strategy": r.strategy_name,
                        "perplexity": r.perplexity,
                        "cache_memory_gb": r.cache_memory_gb,
                        "throughput": r.throughput_tokens_per_sec,
                        "compression_ratio": r.compression_ratio,
                    }
                    for r in self.results
                ],
                f,
                indent=2,
            )
        print(f"\n✓ Results saved to {results_json}")
    
    def generate_visualizations(self):
        """
        Generate all visualization plots.
        """
        print("\nGenerating visualizations...")
        
        if not self.results:
            print("No results to visualize.")
            return
        
        # 1. Pareto frontier
        plot_pareto_frontier(
            self.results,
            output_path=str(self.output_dir / "pareto_frontier.png"),
        )
        
        # 2. Comprehensive compression analysis
        plot_compression_analysis(
            self.results,
            output_path=str(self.output_dir / "compression_analysis.png"),
        )
        
        # 3. Comparison table
        generate_comparison_table(
            self.results,
            output_path=str(self.output_dir / "comparison_table.txt"),
        )
        
        print("\n✓ All visualizations saved to results directory")
    
    def run(self):
        """
        Execute the complete experimental pipeline.
        """
        print("\n" + "=" * 120)
        print("ENTROPY-GUIDED KV CACHE MERGING: COMPLETE EXPERIMENT")
        print("=" * 120 + "\n")
        
        # Step 1: Load model and data
        if not self.load_model_and_tokenizer():
            print("Failed to load model. Exiting.")
            return
        
        if not self.load_dataset():
            print("Failed to load dataset. Exiting.")
            return
        
        # Step 2: Setup strategies
        strategies = self.setup_strategies()
        
        # Step 3: Run experiments
        self.run_experiments(strategies)
        
        # Step 4: Generate visualizations
        self.generate_visualizations()
        
        print("\n" + "=" * 120)
        print("EXPERIMENT COMPLETE")
        print("=" * 120)
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """
    Entry point for the experiment script.
    """
    parser = argparse.ArgumentParser(
        description="Run entropy-guided KV cache merging experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model ID (default: meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pg19",
        help="Dataset to use (default: pg19)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--device",
        type=str,
```