# ANE-LM

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs.

Every other inference engine on Mac (llama.cpp, MLX, etc.) runs on CPU and GPU. The Neural Engine sits idle. ANE-LM is the first to run full LLM inference directly on the ANE — including 4B+ parameter models.

## What's New: 9+ tok/s on Qwen3-4B

The original project ran small models (0.6B–0.8B). We've extended it to **Qwen3-4B** (2560 hidden, 36 layers, 32 Q-heads / 8 KV-heads) through single-chunk FFN compilation, fused kernels, and dispatch reduction:

- **9.1 tok/s** generation on M3 Max (up from 5.7 baseline)
- **113 ANE dispatches** per token (down from 216+)
- ~8s cached init, ~28s first run
- Fused O_proj+residual, fused FFN+residual, single-chunk FFN (150MB kernels)

Also includes **dynamic-weight matrix-vector multiply** APIs — the building blocks for weight-swapping without recompilation. See our [Apple Neural Engine field guide](https://github.com/skyfallsin/apple-neural-engine-field-guide) for the full reverse-engineering findings.

## Supported Models

- **Qwen3** (dense) — tested up to 4B parameters
- **Qwen3.5** (dense, text-only) — hybrid DeltaNet + full attention

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
# Single-shot generation
./build/ane-lm generate --model /path/to/Qwen3-4B --prompt "Hello" --max-tokens 100

# Interactive chat
./build/ane-lm chat --model /path/to/Qwen3-4B

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane-lm convert --model /path/to/Qwen3-4B
```

Download models in safetensors format from HuggingFace (e.g. `Qwen/Qwen3-4B`, `Qwen/Qwen3.5-0.8B`).

### Options

```
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--no-ane-cache       Disable persistent ANE compile cache
-v, --verbose        Show detailed initialization info
```

## Performance

| Model | Params | Gen tok/s | Init (cached) | Dispatches/token | FFN Strategy |
|-------|--------|-----------|---------------|------------------|---------------|
| Qwen3.5-0.8B | 0.8B | ~17 t/s | ~2s | 72 | Single fused |
| Qwen3-4B | 4B | **~9.1 t/s** | ~8s | 113 | Single fused + residual |

### Bottleneck: Memory Bandwidth

The ANE is **bandwidth-bound**, not compute-bound. Each batch=1 token reads the entire 7.4 GB weight set from unified memory. Profiling breakdown (M3 Max, Qwen3-4B):

| Component | ms/token | % | Effective BW |
|-----------|----------|---|-------------|
| FFN (ANE) | 58.7 | 54.8% | 91.7 GB/s |
| QKV proj (ANE) | 20.8 | 19.4% | 34.1 GB/s |
| O_proj+res (ANE) | 14.6 | 13.6% | 32.4 GB/s |
| LM head (ANE) | 8.3 | 7.7% | 101.7 GB/s |
| CPU (attn, norm, RoPE) | 4.9 | 4.5% | — |
| **Total** | **107.1** | 100% | **69.1 GB/s avg** |

The ANE achieves ~100 GB/s peak memory bandwidth on large kernels (FFN, LM head) but only ~33 GB/s on small kernels (QKV, O_proj) due to per-dispatch fixed overhead (~363µs). System peak is 400 GB/s, but the ANE's memory interface appears capped at ~100 GB/s — about 25% of system bandwidth.

**Current: 9.3 tok/s. Theoretical ceiling: 13.5 tok/s** (zero-overhead streaming at 100 GB/s). The 31% overhead gap breaks down as: per-dispatch fixed cost (38%), small-kernel bandwidth penalty (22%), CPU compute (5%).

## How It Works

ANE-LM compiles each weight matrix into an ANE convolution kernel with weights baked in at compile time. For a single token:

1. **Embedding lookup** (CPU)
2. **Per-layer** (×36 for Qwen3-4B):
   - RMSNorm (CPU) — ANE RMSNorm has 0.01 max error floor that amplifies through FFN
   - Fused QKV projection (ANE — single kernel, Q+K+V matmuls)
   - QK-norm + RoPE (CPU)
   - GQA attention + KV cache (CPU, softmax vectorized with vDSP)
   - O projection + residual add (ANE — single fused kernel)
   - RMSNorm (CPU)
   - SwiGLU FFN + residual add (ANE — single fused kernel, up to 150MB)
3. **Final norm** (CPU)
4. **LM head** (ANE — 5 chunks of 32768 vocab each)

Persistent compile cache stores compiled ANE programs on disk, so first-run compilation (~28s for 4B) only happens once.

## ANE Hardware Findings

Through systematic reverse-engineering (40+ targeted test programs), we've documented hardware constraints and working patterns that aren't in any public documentation. Key discoveries:

- **Memory bandwidth ceiling: ~100 GB/s** — the ANE's memory interface is capped at ~25% of M3 Max system bandwidth (400 GB/s). This is the fundamental bottleneck for batch=1 LLM inference.
- **fp16-only weights** — ANE hardware only processes fp16. CoreML's INT8 quantization is resolved to fp16 by Espresso before reaching the ANE compiler. INT8 halves disk size but cannot reduce inference bandwidth.
- **IOSurface W ≥ 32 (SP)** — all runtime inputs must have innermost dim ≥ 32 or eval silently fails.
- **`tile` poisons global ANE state** — any program using `tile` corrupts all subsequent evals in the process (status 0x1d). Only recovery is process restart.
- **Conv dynamic weights silently fail** — conv reads weights from a dedicated hardware bus, not IOSurface inputs. Compiles fine, produces garbage.
- **Per-dispatch fixed cost: ~363µs** — overhead from CPU↔ANE round-trips. At 113 dispatches/token, this adds 41ms (38% of token time). Fusing ops to reduce dispatch count is the primary optimization lever.
- **softmax works on runtime tensors** — all attention building blocks (softmax, exp, sub, reduce_max, mul) work, opening a path to ANE-based attention.
- **RMSNorm precision floor** — fp16 RMSNorm has ~0.01 max error at 2560 channels, amplified ~6400× through FFN. Must stay on CPU.

**📖 Full details: [Apple Neural Engine — Reverse-Engineering Field Guide](https://github.com/skyfallsin/apple-neural-engine-field-guide)**

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) — Original project: ANE runtime, Qwen3/3.5 inference, safetensors loader, tokenizer, chat template engine
- [maderix/ANE](https://github.com/maderix/ANE) — Training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++
