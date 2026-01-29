# StreamingLLM: Why It's Faster & Modern LLM Cache Techniques

## Part 1: Why StreamingLLM is Faster and Ideal

### The StreamingLLM Approach

StreamingLLM keeps two types of tokens in the cache:
1. **Attention Sinks** (First ~4 tokens) - Empirically shown to receive high attention
2. **Recent Window** (Last N tokens) - Contains current context

```
Original Cache:  [token1][token2][token3]...[token976]
                 Full 976 tokens, exponential memory

StreamingLLM:    [Sinks(4)][Recent(488)]
                 ~500 tokens, 50% cache saved
                 Both critical parts retained
```