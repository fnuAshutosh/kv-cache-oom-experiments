# GitHub Setup Guide

## Repository Information

**Repository:** https://github.com/fnuAshutosh/kv-cache-oom-experiments  
**Status:** ✓ Public, Ready for use  
**Files:** 100+ tracked files  

## What's Included

### Core Experiments
- ✓ `demonstrate_oom_problem.py` - Standalone OOM test
- ✓ `KV_Cache_OOM_Demonstration_Colab.ipynb` - Multi-model notebook
- ✓ `setup_and_run_colab_cli.sh` - Automation script

### Documentation  
- ✓ `README.md` - Project overview
- ✓ `COLAB_MANUAL_SETUP_GUIDE.md` - Setup instructions
- ✓ `CUDA_ERROR_FIX_GUIDE.md` - Troubleshooting
- ✓ `requirements.txt` - Dependencies

### Configuration
- ✓ `.gitignore` - Excludes large files

## Clone & Setup

```bash
git clone https://github.com/fnuAshutosh/kv-cache-oom-experiments.git
cd kv-cache-oom-experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

**Option 1: Local**
```bash
python demonstrate_oom_problem.py --model gpt2
```

**Option 2: Colab (Recommended)**
1. Go to https://colab.research.google.com
2. File → Open notebook → GitHub
3. Paste: https://github.com/fnuAshutosh/kv-cache-oom-experiments
4. Select: KV_Cache_OOM_Demonstration_Colab.ipynb
5. Runtime → Change runtime type → GPU
6. Runtime → Run all

---

**Ready to use!** 🚀
