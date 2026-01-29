```python
"""
Visualization and Analysis Tools

This module provides utilities for visualizing entropy patterns and
generating comparative analysis plots for the experimental results.

Includes:
- Attention entropy heatmaps
- Perplexity vs. Memory Pareto frontier plots
- Compression ratio analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
import torch


def plot_entropy_heatmap(
    entropy_history: List[torch.Tensor],
    layer_indices: List[int] = None,
    output_path: str = "entropy_heatmap.png",
):
    """
    Plot a heatmap of token position vs. layer entropy.
    
    Visualization goal: Confirm that sink tokens (0-4) and specific entities
    trigger low entropy (sharp attention), while function words trigger
    high entropy (diffuse attention).
    
    Args:
        entropy_history: List of entropy tensors from compression steps.
                        Each tensor shape: (num_layers, seq_len) or (seq_len,).
        layer_indices: Optional list of layer indices to plot (default: all).
        output_path: Path to save the figure.
    """
    if not entropy_history:
        print("No entropy history to plot.")
        return
    
    # Stack entropy across time steps
    if isinstance(entropy_history[0], torch.Tensor):
        # Convert to numpy
        entropy_array = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e
                        for e in entropy_history]
    else:
        entropy_array = entropy_history
    
    # Handle different shapes
    if len(entropy_array[0].shape) == 1:
        # 1D: (seq_len,) for each step -> (num_steps, seq_len)
        entropy_matrix = np.stack(entropy_array, axis=0)
    else:
        # 2D: (num_layers, seq_len) for each step
        entropy_matrix = np.stack(entropy_array, axis=0)
    
    # Create figure
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot heatmap (average over steps if multiple)
    if entropy_matrix.ndim > 2:
        entropy_matrix = entropy_matrix.mean(axis=0)  # Average over steps
    
    im = axes.imshow(entropy_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    axes.set_xlabel('Token Position', fontsize=12)
    axes.set_ylabel('Layer Index', fontsize=12)
    axes.set_title('Attention Entropy Heatmap: Token Position vs. Layer', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes)
    cbar.set_label('Shannon Entropy', fontsize=11)
    
    # Highlight sink region
    if entropy_matrix.shape[1] > 4:
        axes.axvline(x=3.5, color='green', linestyle='--', linewidth=2, label='Sink boundary (tokens 0-3)')
        axes.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy heatmap saved to {output_path}")
    plt.close()


def plot_pareto_frontier(
    results: List,
    output_path: str = "pareto_frontier.png",
):
    """
    Plot Pareto frontier in Memory vs. Perplexity space.
    
    The Pareto frontier represents the set of non-dominated solutions,
    showing the trade-off between accuracy (perplexity) and memory efficiency.
    
    Args:
        results: List of EvaluationMetrics objects.
        output_path: Path to save the figure.
    """
    if not results:
        print("No results to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    strategies = [r.strategy_name for r in results]
    memory = [r.cache_memory_gb for r in results]
    ppl = [r.perplexity for r in results]
    compression = [r.compression_ratio for r in results]
    
    # Color mapping for compression ratio
    colors = plt.cm.viridis(np.array(compression) / max(compression))
    
    # Scatter plot
    scatter = ax.scatter(
        memory, ppl,
        s=300,
        c=compression,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
    )
    
    # Annotate points
    for i, strategy in enumerate(strategies):
        ax.annotate(
            strategy,
            (memory[i], ppl[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
        )
    
    # Compute and plot Pareto frontier
    pareto_indices = []
    for i, (m, p) in enumerate(zip(memory, ppl)):
        is_dominated = False
        for j, (m_other, p_other) in enumerate(zip(memory, ppl)):
            if i != j and m_other <= m and p_other <= p:
                if m_other < m or p_other < p:
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)
    
    if pareto_indices:
        pareto_memory = [memory[i] for i in sorted(pareto_indices)]
        pareto_ppl = [ppl[i] for i in sorted(pareto_indices)]
        
        # Sort by memory for line plot
        sorted_pairs = sorted(zip(pareto_memory, pareto_ppl), key=lambda x: x[0])
        pareto_memory_sorted = [x[0] for x in sorted_pairs]
        pareto_ppl_sorted = [x[1] for x in sorted_pairs]
        
        ax.plot(
            pareto_memory_sorted, pareto_ppl_sorted,
            'r--', linewidth=2, label='Pareto Frontier', alpha=0.7
        )
    
    # Labels and legend
    ax.set_xlabel('Cache Memory (GB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (PPL)', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Memory vs. Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colorbar for compression ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Compression Ratio', fontsize=11)
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pareto frontier plot saved to {output_path}")
    plt.close()


def plot_compression_analysis(
    results: List,
    output_path: str = "compression_analysis.png",
):
    """
    Create a comprehensive compression analysis visualization.
    
    Shows:
    - Compression ratio comparison across strategies
    - Perplexity impact of compression
    - Memory savings
    
    Args:
        results: List of EvaluationMetrics objects.
        output_path: Path to save the figure.
    """
    if not results:
        print("No results to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = [r.strategy_name for r in results]
    compression = [r.compression_ratio * 100 for r in results]
    ppl = [r.perplexity for r in results]
    memory = [r.cache_memory_gb for r in results]
    throughput = [r.throughput_tokens_per_sec for r in results]
    
    # Plot 1: Compression Ratio
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    axes[0, 0].barh(strategies, compression, color=colors, edgecolor='black')
    axes[0, 0].set_xlabel('Compression Ratio (%)', fontsize=11)
    axes[0, 0].set_title('Compression Ratio by Strategy', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Perplexity
    axes[0, 1].bar(strategies, ppl, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Perplexity (PPL)', fontsize=11)
    axes[0, 1].set_title('Perplexity by Strategy', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Memory Usage
    axes[1, 0].bar(strategies, memory, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_ylabel('Memory (GB)', fontsize=11)
    axes[1, 0].set_title('Cache Memory Usage', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Throughput
    axes[1, 1].bar(strategies, throughput, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 1].set_ylabel('Throughput (Tokens/sec)', fontsize=11)
    axes[1, 1].set_title('Decoding Throughput', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Adjust layout
    for ax in axes.flat:
        ax.tick_params(axis='x', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Compression analysis plot saved to {output_path}")
    plt.close()


def generate_comparison_table(
    results: List,
    output_path: str = "comparison_table.txt",
) -> str:
    """
    Generate a formatted text table comparing all strategies.
    
    Args:
        results: List of EvaluationMetrics objects.
        output_path: Path to save the table.
    
    Returns:
        Formatted table as string.
    """
    if not results:
        return "No results to display."
    
    header = (
        f"{'Strategy':<20} | {'Perplexity':>10} | {'Memory (GB)':>12} | "
        f"{'Compression':>12} | {'Throughput':>15}"
    )
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for metric in sorted(results, key=lambda x: x.perplexity):
        line = (
            f"{metric.strategy_name:<20} | "
            f"{metric.perplexity:>10.2f} | "
            f"{metric.cache_memory_gb:>12.2f} | "
            f"{metric.compression_ratio*100:>11.1f}% | "
            f"{metric.throughput_tokens_per_sec:>14.1f}"
        )
        lines.append(line)
    
    table = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"\nComparison table saved to {output_path}")
    print(table)
    
    return table
```