#pragma once

/**
 * GGUF model loader for ane.cpp
 *
 * Replaces safetensors loader + WeightCache + bf16→fp16 pipeline.
 * Single GGUF file as only weight source.
 *
 * Weight tensors are loaded into two destinations:
 *   1. CPU (fp32): norms, DeltaNet SSM params, embedding, lm_head — for decode/ANE
 *   2. ggml Metal backend: all projection weights — for GPU prefill via ggml_mul_mat
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace ane_lm {

/// Holds all weight tensors loaded from a GGUF file.
/// CPU weights are dequantized to fp32. GPU weights stay in native GGUF format
/// (Q8_0, Q4_K_M, F16, etc.) as ggml_tensor* in the Metal backend.
struct GGUFModel {
    // --- GPU weight tensors (ggml Metal backend, native quantization) ---
    // Per-layer projection weights for prefill
    struct LayerWeights {
        // Phase 1: QKV projections
        ggml_tensor* first_proj = nullptr;    // Linear: attn_qkv [qkv_dim, H]
                                               // Full:   concat(q,k,v) [q+2kv, H]
        ggml_tensor* z_proj = nullptr;         // Linear only: ssm_out/gate [z_dim, H]
        ggml_tensor* a_proj = nullptr;         // Linear only: ssm_alpha [n_val_heads, H]
        ggml_tensor* b_proj = nullptr;         // Linear only: ssm_beta [n_val_heads, H]

        // Phase 2: o_proj + FFN
        ggml_tensor* o_proj = nullptr;         // Linear: ssm_out [H, val_dim]
                                               // Full:   attn_output [H, qH]
        ggml_tensor* gate_proj = nullptr;      // ffn_gate [I, H]
        ggml_tensor* up_proj = nullptr;        // ffn_up [I, H]
        ggml_tensor* down_proj = nullptr;      // ffn_down [H, I]

        // Norm weights (F32 in ggml, used by GPU RMSNorm)
        ggml_tensor* input_norm = nullptr;     // attn_norm [H]
        ggml_tensor* post_norm = nullptr;      // post_attention_norm [H]

        int first_proj_rows = 0;  // output dim of first_proj
        int o_proj_in = 0;        // input dim of o_proj
    };

    std::vector<LayerWeights> layers;

    // Global weights
    ggml_tensor* token_embd = nullptr;         // [vocab, H]
    ggml_tensor* output_norm = nullptr;        // [H]
    ggml_tensor* lm_head = nullptr;            // [vocab, H] (or alias of token_embd)

    // --- CPU weight data (dequantized to fp32) ---
    // These are owned by this struct (malloc'd)
    float* embed_tokens_f32 = nullptr;         // [vocab × H] for embedding lookup
    float* lm_head_f32 = nullptr;              // [vocab × H] for lm_head (or alias)
    float* final_norm_f32 = nullptr;           // [H]

    struct CPULayerWeights {
        float* input_layernorm = nullptr;      // [H]
        float* post_attention_layernorm = nullptr; // [H]

        // DeltaNet-specific (linear attention layers only)
        float* conv1d_w = nullptr;             // [qkv_dim × kernel]
        float* A = nullptr;                    // [n_val_heads] (exp'd)
        float* dt_bias = nullptr;              // [n_val_heads]
        float* ssm_norm_w = nullptr;           // [val_dim]

        // Full-attention-specific
        float* q_norm = nullptr;               // [head_dim]
        float* k_norm = nullptr;               // [head_dim]
    };
    std::vector<CPULayerWeights> cpu_layers;

    // --- ggml state (must stay alive while tensors are in use) ---
    ggml_backend_t metal_backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context* weight_ctx = nullptr;        // ggml context for GPU weight tensors
    gguf_context* gguf_ctx = nullptr;          // GGUF file context (for tensor data)
    ggml_context* gguf_data_ctx = nullptr;     // ggml context with loaded GGUF data

    // Metadata
    int num_layers = 0;
    bool tie_word_embeddings = false;

    ~GGUFModel();
};

/// Load a GGUF model file. Returns nullptr on failure.
/// layer_types: per-layer type (0=linear_attention, 1=full_attention)
/// The Metal backend is created internally.
GGUFModel* load_gguf_model(
    const std::string& gguf_path,
    const std::vector<int>& layer_types,  // 0=linear, 1=full
    int hidden_size,
    int intermediate_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int lin_qkv_dim,
    int lin_total_val,
    int lin_num_val_heads,
    int full_q_dim,
    int full_kv_dim,
    int full_out_dim,
    int vocab_size,
    bool tie_word_embeddings
);

} // namespace ane_lm
