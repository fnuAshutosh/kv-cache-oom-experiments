# 🚀 START HERE: Entropy-Guided KV Cache Merging

**Welcome!** This is your entry point to understanding and running the entropy-guided KV cache compression research.

---

## What Is This?

A research framework for compressing the Key-Value (KV) cache in Large Language Models using **information theory**.

**The Problem:** During LLM inference, KV cache memory grows linearly with sequence length, limiting context windows and batch sizes.

**Our Solution:** Merge high-entropy tokens instead of deleting them, preserving semantic information while reducing memory.

---

## Quick Start (5 Minutes)

### 1. Validate Installation
```bash
python quick_start.py
```

Expected output:
- ✓ Model loaded
- ✓ Dummy KV cache created
- ✓ Compression strategies applied
- ✓ Results saved

### 2. Read the Theory
See [RESEARCH_REPORT.md](RESEARCH_REPORT.md) - sections 1-3 explain the motivation and approach.

### 3. Run a Benchmark
```bash
python main_experiment.py --device cuda
```

This will:
- Load Llama-2-7B
- Run 4 compression strategies
- Generate comparison plots
- Save results to `results/`

---

## File Guide

| File | Purpose | Time |
|------|---------|------|
| [entropy_merged_kv_cache.py](entropy_merged_kv_cache.py) | Core algorithm | Study: 30 min |
| [baselines.py](baselines.py) | Comparison strategies | Study: 15 min |
| [evaluation.py](evaluation.py) | Benchmarking framework | Study: 20 min |
| [visualization.py](visualization.py) | Plotting & analysis | Study: 10 min |
| [main_experiment.py](main_experiment.py) | Full pipeline | Run: 30-60 min |
| [quick_start.py](quick_start.py) | Quick demo | Run: 2 min |
| [test_entropy_analysis.py](test_entropy_analysis.py) | Validation | Run: 5 min |
| [RESEARCH_REPORT.md](RESEARCH_REPORT.md) | Full methodology | Read: 20 min |
| [INDEX.md](INDEX.md) | Complete navigation | Reference |

---

## Core Concept

### Shannon Entropy

Measures how "spread out" attention weights are:
- **High entropy** → Attention spread across many tokens → Information is diffuse → Can merge tokens
- **Low entropy** → Attention on few tokens → Information is sharp → Must keep separate

### Attention Sinks

Models "dump" excess attention on the first few tokens. Always preserve them!

### The Strategy

1. **Calculate** entropy of attention per token
2. **Identify** high-entropy token spans (mergeable)
3. **Preserve** sinks (first 4 tokens) and low-entropy tokens
4. **Merge** high-entropy spans into summary tokens
5. **Compress** KV cache without losing critical information

---

## Expected Results

### Llama-2-7B on PG-19 Dataset

| Strategy | Perplexity | Compression | Memory (GB) |
|----------|-----------|-------------|-------------|
| Full Cache | 25-26 | 0% | 3.0-3.5 |
| EntropyMerged | 26-27 | 70% | 0.9-1.2 |
| H2O | 27-29 | 80% | 0.6-0.8 |
| StreamingLLM | 27-30 | 75% | 0.8-1.0 |

**Key Finding:** Entropy merging achieves better PPL than H2O at comparable compression!

---

## Common Questions

### Q: How long does a full benchmark take?
**A:** 30-60 minutes on GPU (Llama-2-7B). Faster with smaller models.

### Q: Can I test on other models?
**A:** Yes! Change `--model` flag:
```bash
python main_experiment.py --model mistralai/Mistral-7B-v0.1 --device cuda
```

### Q: What if I don't have a GPU?
**A:** Use `--device cpu` (slower, ~10-20x slower):
```bash
python main_experiment.py --device cpu
```

### Q: How do I understand the theory?
**A:** Start here:
1. [RESEARCH_REPORT.md](RESEARCH_REPORT.md) - Sections 1-3
2. [entropy_merged_kv_cache.py](entropy_merged_kv_cache.py) - Algorithm docstring
3. [test_entropy_analysis.py](test_entropy_analysis.py) - Practical validation

### Q: How do I extend this?
**A:** Modify [config.py](config.py) or create new baseline strategies in [baselines.py](baselines.py).

---

## Next Steps

### Immediate (Today)
- [ ] Run `python quick_start.py`
- [ ] Read [RESEARCH_REPORT.md](RESEARCH_REPORT.md) sections 1-2

### This Week
- [ ] Run `python main_experiment.py`
- [ ] Analyze results in `results/` directory
- [ ] Study [entropy_merged_kv_cache.py](entropy_merged_kv_cache.py)

### This Month
- [ ] Run on multiple models (Llama-3, Mistral)
- [ ] Complete ablation studies
- [ ] Write research findings

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'transformers'"
```bash
pip install transformers torch datasets
```

### Error: "CUDA out of memory"
```bash
python main_experiment.py --device cpu  # Or use smaller model
```

### Error: "Dataset not found"
PG-19 will download automatically on first run. Requires internet.

### Results don't match expected values?
Check:
1. Model loaded correctly: `python quick_start.py`
2. Dataset downloaded: Check `~/.cache/huggingface/datasets/`
3. Entropy threshold: Default 0.5; try 0.3-0.7 range

---

## Success Checklist

- [ ] `python quick_start.py` runs without errors
- [ ] Output shows compression stats
- [ ] Understand what Shannon entropy measures
- [ ] Know why attention sinks are preserved
- [ ] Can explain the merging algorithm in your own words
- [ ] Ready to run `main_experiment.py`

---

## Resources

- **Full Documentation:** [INDEX.md](INDEX.md)
- **Algorithm Details:** [entropy_merged_kv_cache.py](entropy_merged_kv_cache.py)
- **Research Theory:** [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
- **Configuration:** [config.py](config.py)
- **Results:** `results/` directory after running experiments

---

**You're all set!** Start with `python quick_start.py` and explore from there. 🚀