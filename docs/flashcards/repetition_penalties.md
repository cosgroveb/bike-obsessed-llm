# Repetition Penalty Fundamentals

# What mathematical operation does repetition penalty apply to previously generated tokens?
---
It divides the logit/probability of previously generated tokens by the penalty factor. For repetition_penalty=1.1, each repeated token gets its probability reduced to P/1.1 ≈ 0.91P, or in log-space: log(P) - log(1.1) ≈ log(P) - 0.095.

# What is the difference between repetition penalty and no_repeat_ngram_size?
---
Repetition penalty applies a soft multiplicative reduction to all previously seen tokens, while no_repeat_ngram_size creates hard constraints by setting logits to -∞ for tokens that would complete already-seen n-grams. One is gradual discouragement, the other is absolute blocking.

# What does a repetition penalty of 1.1 represent in terms of sampling bias?
---
It creates a small but consistent logarithmic bias against repetition that accumulates over generation. The 1.1 factor represents a conservative approach that provides measurable diversity improvement without dramatic probability distribution shifts.

# Research and Applications

# Which paper introduced the concept of neural text degeneration that repetition penalties address?
---
"The Curious Case of Neural Text Degeneration" by Holtzman et al. (2019) states that "using likelihood as a decoding objective leads to text that is bland and strangely repetitive." They propose Nucleus Sampling to address repetition issues. Available at: https://arxiv.org/abs/1904.09751

# What foundational paper established the autoregressive generation paradigm where repetition becomes problematic?
---
"Attention Is All You Need" by Vaswani et al. (2017) introduced the Transformer architecture that uses autoregressive generation, where each token depends on previous tokens - the paradigm that makes repetition penalties necessary. Available at: https://arxiv.org/abs/1706.03762

# Which paper analyzes how different decoding strategies affect generation quality?
---
"Typical Sampling" by Meister et al. (2022) analyzes various decoding strategies including repetition penalties and their effects on generation quality. Available at: https://arxiv.org/abs/2202.00666

# Practical Implementation

# What are typical values for repetition penalty in practice?
---
Common values range from 1.0 (no penalty) to 1.2 (strong penalty). 1.1 is widely used as a balanced default. Values above 1.3 often cause unnatural text. See examples: [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT/blob/main/app.py), [ChatPDF](https://github.com/shibing624/ChatPDF/blob/main/rag.py)

# What are typical values for no_repeat_ngram_size?
---
Common values are 2-4. Size 3 blocks trigram repetition (most common), size 2 is more aggressive, size 4+ is more permissive. Value 0 disables the constraint. See examples: [amazon-science/polygon-transformer](https://github.com/amazon-science/polygon-transformer/blob/main/demo.py)

# How do repetition penalties interact with softmax normalization?
---
Repetition penalties modify logits before softmax, so reducing repeated token logits increases the relative probability of all other tokens. This creates indirect effects where non-repeated tokens become more likely even if they weren't directly boosted.

# When do repetition penalties become counterproductive?
---
When natural language requires legitimate repetition (emphasis, technical terms, proper nouns), when measuring specific token frequencies in experiments, or when penalty values are too high (>1.3) causing unnatural avoidance of common words and phrases.