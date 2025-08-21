# Torch Data Types (dtypes) Fundamentals

## What are torch dtypes and why do they matter?

Data types that specify how tensors store numerical values in memory. They determine precision (how many decimal places), memory usage (bytes per number), and computational speed. Like choosing between int vs float in regular programming, but for AI workloads.

## What are the main floating-point dtypes in PyTorch?

**float32** (torch.float): 32-bit precision, 4 bytes per number, standard default. **float16** (torch.half): 16-bit precision, 2 bytes per number, faster but less precise. **bfloat16** (torch.bfloat16): 16-bit with float32's range, Google's TPU-optimized format.

## What's the memory difference between float32 and float16?

Float16 uses exactly half the memory of float32. A 1000x1000 tensor that takes 4MB in float32 only takes 2MB in float16. This matters enormously for large language models where memory is often the limiting factor.

## What is the precision difference between these dtypes?

**float32**: ~7 decimal digits of precision. **float16**: ~3-4 decimal digits. **bfloat16**: ~2-3 decimal digits but same range as float32. Think float32 as a microscope, float16 as a magnifying glass, bfloat16 as binoculars.

# Dtype Consistency and Testing

## Why does dtype consistency matter for testing?

Mixed dtypes cause automatic conversions that can introduce numerical errors, making tests flaky. If your model expects float32 but gets float16 inputs, PyTorch will upcast, potentially changing results enough to break assertions.

## What happens when you mix dtypes in tensor operations?

PyTorch automatically promotes to the higher precision dtype. float16 + float32 → float32. This can be expensive (memory allocation + copying) and introduce subtle bugs where results differ from pure single-dtype operations.

## Why might a test pass with one dtype but fail with another?

Precision differences compound through calculations. A small rounding error in float16 might accumulate through matrix multiplications until it crosses a test threshold that would pass with float32's extra precision.

## How do you ensure dtype consistency in tests?

Use `.to(dtype)` explicitly on all tensors, set default dtype with `torch.set_default_dtype()`, or create tensors with explicit dtype parameters like `torch.zeros(shape, dtype=torch.float32)`.

# Choosing the Right Dtype

## When should you use float32?

**Testing and debugging**: Need reproducible, precise results. **Research**: Want to eliminate numerical issues as variables. **CPU inference**: float32 is often faster than float16 on CPU. **Small models**: Memory isn't constrained.

## When should you use float16?

**GPU training/inference**: Modern GPUs have specialized Tensor Cores for float16. **Large models**: Memory constraints force the choice. **Production inference**: Speed matters more than tiny precision differences. **Mixed precision training**: Combine with float32 for stable gradients.

## When should you use bfloat16?

**TPU workloads**: Google's hardware is optimized for it. **Training large models**: Better numerical stability than float16 during training. **When you need float32's range but not precision**: Scientific computing with large dynamic ranges.

## What's the "sweet spot" for each dtype?

**float32**: Training smaller models, research, debugging. **float16**: Large model inference, memory-constrained training. **bfloat16**: Very large model training, TPU deployment, when you hit float16 stability issues.

# Testing-Specific Considerations

## Why use float32 for testing even if production uses float16?

Testing should eliminate numerical precision as a source of variance. Using float32 ensures that test failures come from actual logic bugs, not accumulated floating-point errors that could mask real issues.

## How do you test dtype-specific behavior without compromising reliability?

Write separate test functions for each dtype, but use float32 for your core logic tests. Test dtype conversion behavior explicitly in dedicated tests, not mixed with business logic verification.

## What's the testing strategy for models that use mixed precision?

Test the core algorithms in float32 for precision, then have separate tests that verify mixed precision behavior works correctly. Like testing the recipe with precise measurements, then testing the scaled-up version.

## How do you handle dtype in parametrized tests?

Create base test logic in float32, then have dtype-specific tests that convert inputs/outputs and adjust tolerance levels. Don't parametrize your main logic tests over dtypes - keep that complexity separate.

# Common Pitfalls and Debugging

## What's the most common dtype-related bug?

Silent dtype promotion causing unexpected memory usage or performance degradation. You think you're running in float16 but operations are secretly promoting to float32, using double the memory.

## How do you debug dtype inconsistencies?

Add `print(tensor.dtype)` statements, use `torch.set_warn_always(True)` for promotion warnings, or write a helper function that validates all tensors in a computation have expected dtypes.

## Why might your model run out of memory after a dtype change?

Dtype promotion during operations. If your model mixes float16 and float32 tensors, intermediate results get promoted to float32, using more memory than expected. Always ensure consistent dtypes throughout.

## What's the "gradient underflow" problem with float16?

Very small gradients get rounded to zero in float16, stopping training. This is why mixed precision training keeps gradients in float32 but forward pass in float16 - best of both worlds.

# Performance and Hardware Considerations

## Which dtypes are hardware-accelerated?

**Modern GPUs** (A100, V100): Specialized Tensor Cores for float16 and bfloat16 matrix operations. **TPUs**: Optimized for bfloat16. **CPUs**: Generally fastest with float32, float16 might be slower due to conversion overhead.

## What's the speed difference between dtypes?

On GPU with Tensor Cores: float16/bfloat16 can be 2-4x faster than float32 for large matrix operations. On CPU: float32 is usually fastest. The speedup depends heavily on your specific hardware and operation sizes.

## Why might float16 be slower than float32 on some hardware?

Older GPUs or CPUs lack native float16 support, so operations get converted to float32, computed, then converted back. This conversion overhead can make float16 slower than just using float32 directly.

## How do you benchmark dtype performance correctly?

Test with realistic tensor sizes for your use case, warm up the GPU before timing, measure both memory usage and speed, and test on your actual deployment hardware - performance characteristics vary significantly between systems.