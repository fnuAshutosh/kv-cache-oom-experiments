# CUDA Device Assertion Error - Diagnostics & Fix

## What Happened

```
AcceleratorError: CUDA error: device-side assert triggered
```

This error occurred during the **baseline memory measurement** when calling `tracker.measure("baseline", 0)`.

### Root Cause

The CUDA device entered an **error state** after loading the model. When you call `torch.cuda.empty_cache()` in the `measure()` function, CUDA detects a problem with the device state and triggers an assertion error.

This can happen due to:
1. **GPU Memory Corruption**: Invalid memory access during model loading
2. **Bad Device State**: Model loading left GPU in inconsistent state
3. **Kernel Launch Failures**: Silent CUDA kernel failures that surface later
4. **Version Incompatibilities**: PyTorch/CUDA version mismatches causing device conflicts