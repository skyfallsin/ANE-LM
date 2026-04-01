#include "gguf_loader.h"
#include "ggml-metal.h"
#include "ggml-alloc.h"
#include <ane_lm/common.h>

#include <cstring>
#include <cmath>

namespace ane_lm {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dequantize any ggml tensor to a newly malloc'd fp32 buffer.
static float* dequant_to_f32(const ggml_tensor* t) {
    const int64_t n = ggml_nelements(t);
    float* out = (float*)malloc(n * sizeof(float));
    if (!out) return nullptr;

    if (t->type == GGML_TYPE_F32) {
        memcpy(out, t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "[gguf] No dequantization for type %s\n", ggml_type_name(t->type));
            free(out);
            return nullptr;
        }
        traits->to_float(t->data, out, n);
    }
    return out;
}

/// Look up a tensor in the GGUF data context. Returns nullptr if not found.
static ggml_tensor* find_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        // Not an error — some tensors are optional per layer type
    }
    return t;
}

/// Required tensor lookup — prints error if missing.
static ggml_tensor* require_tensor(ggml_context* ctx, const char* name) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[gguf] ERROR: required tensor '%s' not found\n", name);
    }
    return t;
}

// ---------------------------------------------------------------------------
// GGUFModel destructor
// ---------------------------------------------------------------------------

GGUFModel::~GGUFModel() {
    // Free CPU weights
    free(embed_tokens_f32);
    if (lm_head_f32 != embed_tokens_f32) free(lm_head_f32);
    free(final_norm_f32);

    for (auto& cl : cpu_layers) {
        free(cl.input_layernorm);
        free(cl.post_attention_layernorm);
        free(cl.conv1d_w);
        free(cl.A);
        free(cl.dt_bias);
        free(cl.ssm_norm_w);
        free(cl.q_norm);
        free(cl.k_norm);
    }

    // Free ggml state (order matters)
    if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
    if (weight_ctx) ggml_free(weight_ctx);
    if (gguf_ctx) gguf_free(gguf_ctx);
    if (gguf_data_ctx) ggml_free(gguf_data_ctx);
    if (metal_backend) ggml_backend_free(metal_backend);
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

GGUFModel* load_gguf_model(
    const std::string& gguf_path,
    const std::vector<int>& layer_types,
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
    bool tie_word_embeddings)
{
    Timer total_timer;
    const int num_layers = (int)layer_types.size();
    const int H = hidden_size;
    const int I = intermediate_size;

    LOG("[gguf] Loading %s (%d layers)\n", gguf_path.c_str(), num_layers);

    // ---------------------------------------------------------------
    // Step 1: Load GGUF file (with tensor data into CPU memory)
    // ---------------------------------------------------------------
    ggml_context* data_ctx = nullptr;
    gguf_init_params params = {
        .no_alloc = false,  // load all tensor data
        .ctx = &data_ctx,
    };

    gguf_context* gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf || !data_ctx) {
        fprintf(stderr, "[gguf] Failed to load %s\n", gguf_path.c_str());
        return nullptr;
    }

    int64_t n_tensors = gguf_get_n_tensors(gguf);
    LOG("[gguf] Loaded %lld tensors from GGUF\n", (long long)n_tensors);

    // ---------------------------------------------------------------
    // Step 2: Initialize Metal backend
    // ---------------------------------------------------------------
    ggml_backend_t metal = ggml_backend_metal_init();
    if (!metal) {
        fprintf(stderr, "[gguf] Failed to initialize Metal backend\n");
        gguf_free(gguf);
        ggml_free(data_ctx);
        return nullptr;
    }
    LOG("[gguf] Metal backend: %s\n", ggml_backend_name(metal));

    // ---------------------------------------------------------------
    // Step 3: Create ggml context for GPU weight tensors
    // We create tensor metadata here, then allocate a Metal buffer
    // and copy data from the GGUF CPU tensors.
    // ---------------------------------------------------------------

    // Count how many GPU tensors we need:
    // Per layer: first_proj, o_proj, gate, up, down, input_norm, post_norm
    //   + linear: z_proj, a_proj, b_proj
    //   + full: (first_proj is concat of q,k,v — handled as one tensor)
    // Global: token_embd, output_norm, lm_head
    int n_gpu_tensors = 3; // token_embd, output_norm, lm_head
    for (int L = 0; L < num_layers; L++) {
        n_gpu_tensors += 7; // first_proj, o_proj, gate, up, down, input_norm, post_norm
        if (layer_types[L] == 0) {
            n_gpu_tensors += 3; // z_proj, a_proj, b_proj
        }
    }

    size_t ctx_size = ggml_tensor_overhead() * n_gpu_tensors + 1024;
    ggml_init_params gctx_params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* weight_ctx = ggml_init(gctx_params);
    if (!weight_ctx) {
        fprintf(stderr, "[gguf] Failed to create ggml weight context\n");
        ggml_backend_free(metal);
        gguf_free(gguf);
        ggml_free(data_ctx);
        return nullptr;
    }

    // ---------------------------------------------------------------
    // Step 4: Create GPU weight tensor metadata + load CPU weights
    // ---------------------------------------------------------------
    auto* model = new GGUFModel();
    model->num_layers = num_layers;
    model->tie_word_embeddings = tie_word_embeddings;
    model->metal_backend = metal;
    model->weight_ctx = weight_ctx;
    model->gguf_ctx = gguf;
    model->gguf_data_ctx = data_ctx;
    model->layers.resize(num_layers);
    model->cpu_layers.resize(num_layers);

    char name[256];

    // Helper: create a GPU tensor matching a GGUF source tensor's type and shape
    auto make_gpu_tensor = [&](const char* gguf_name, const char* label) -> ggml_tensor* {
        ggml_tensor* src = find_tensor(data_ctx, gguf_name);
        if (!src) return nullptr;
        ggml_tensor* dst = ggml_dup_tensor(weight_ctx, src);
        ggml_set_name(dst, label);
        return dst;
    };

    // Helper: create a GPU tensor for concatenated Q+K+V (full attention)
    // GGUF has separate q, k, v tensors — we create one [q_dim+2*kv_dim, H] tensor
    auto make_concat_qkv = [&](int L) -> ggml_tensor* {
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", L);
        ggml_tensor* q = require_tensor(data_ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", L);
        ggml_tensor* k = require_tensor(data_ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", L);
        ggml_tensor* v = require_tensor(data_ctx, name);
        if (!q || !k || !v) return nullptr;

        // All should have same ne[0] (=H) and same type
        int total_rows = (int)(q->ne[1] + k->ne[1] + v->ne[1]);
        ggml_tensor* dst = ggml_new_tensor_2d(weight_ctx, q->type, q->ne[0], total_rows);
        snprintf(name, sizeof(name), "blk.%d.first_proj", L);
        ggml_set_name(dst, name);
        return dst;
    };

    // --- Global weights ---
    model->token_embd = make_gpu_tensor("token_embd.weight", "token_embd");
    model->output_norm = make_gpu_tensor("output_norm.weight", "output_norm");

    if (tie_word_embeddings) {
        model->lm_head = model->token_embd;
    } else {
        model->lm_head = make_gpu_tensor("output.weight", "lm_head");
    }

    // CPU: dequantize embedding + norm for decode path
    {
        ggml_tensor* t = require_tensor(data_ctx, "token_embd.weight");
        if (t) model->embed_tokens_f32 = dequant_to_f32(t);

        t = require_tensor(data_ctx, "output_norm.weight");
        if (t) model->final_norm_f32 = dequant_to_f32(t);

        if (tie_word_embeddings) {
            model->lm_head_f32 = model->embed_tokens_f32;
        } else {
            t = find_tensor(data_ctx, "output.weight");
            if (t) model->lm_head_f32 = dequant_to_f32(t);
            else model->lm_head_f32 = model->embed_tokens_f32; // fallback
        }
    }

    if (!model->embed_tokens_f32 || !model->final_norm_f32) {
        fprintf(stderr, "[gguf] Missing required global weights\n");
        delete model;
        return nullptr;
    }

    // --- Per-layer weights ---
    for (int L = 0; L < num_layers; L++) {
        auto& lw = model->layers[L];
        auto& cl = model->cpu_layers[L];
        bool is_linear = (layer_types[L] == 0);

        // -- GPU projection weights --

        if (is_linear) {
            // Linear attention (DeltaNet) layer
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", L);
            lw.first_proj = make_gpu_tensor(name, name);
            if (lw.first_proj) {
                lw.first_proj_rows = (int)lw.first_proj->ne[1];
            }

            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", L);
            lw.z_proj = make_gpu_tensor(name, name);

            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", L);
            lw.a_proj = make_gpu_tensor(name, name);

            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", L);
            lw.b_proj = make_gpu_tensor(name, name);

            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", L);
            lw.o_proj = make_gpu_tensor(name, name);
            if (lw.o_proj) lw.o_proj_in = (int)lw.o_proj->ne[0];

        } else {
            // Full attention layer — concat Q/K/V into one tensor
            lw.first_proj = make_concat_qkv(L);
            if (lw.first_proj) {
                lw.first_proj_rows = (int)lw.first_proj->ne[1];
            }

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", L);
            lw.o_proj = make_gpu_tensor(name, name);
            if (lw.o_proj) lw.o_proj_in = (int)lw.o_proj->ne[0];
        }

        // FFN (same for both layer types)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", L);
        lw.gate_proj = make_gpu_tensor(name, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", L);
        lw.up_proj = make_gpu_tensor(name, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", L);
        lw.down_proj = make_gpu_tensor(name, name);

        // Norm weights (F32)
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", L);
        lw.input_norm = make_gpu_tensor(name, name);

        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", L);
        lw.post_norm = make_gpu_tensor(name, name);

        // -- CPU weights (dequantized to fp32) --

        // Norms
        {
            ggml_tensor* t;
            snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.input_layernorm = dequant_to_f32(t);

            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.post_attention_layernorm = dequant_to_f32(t);
        }

        if (is_linear) {
            // DeltaNet-specific CPU weights
            ggml_tensor* t;

            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.conv1d_w = dequant_to_f32(t);

            snprintf(name, sizeof(name), "blk.%d.ssm_a", L);
            t = require_tensor(data_ctx, name);
            if (t) {
                cl.A = dequant_to_f32(t);
                // Apply exp() as the original code does
                if (cl.A) {
                    for (int64_t i = 0; i < ggml_nelements(t); i++) {
                        cl.A[i] = expf(cl.A[i]);
                    }
                }
            }

            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.dt_bias = dequant_to_f32(t);

            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.ssm_norm_w = dequant_to_f32(t);
        } else {
            // Full-attention-specific CPU weights
            ggml_tensor* t;

            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.q_norm = dequant_to_f32(t);

            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", L);
            t = require_tensor(data_ctx, name);
            if (t) cl.k_norm = dequant_to_f32(t);
        }

        LOG("  Layer %d/%d (%s)...\r", L + 1, num_layers,
            is_linear ? "linear" : "full_attn");
    }

    // ---------------------------------------------------------------
    // Step 5: Allocate Metal buffer and copy weight data
    // ---------------------------------------------------------------
    LOG("[gguf] Allocating Metal buffer for GPU weights...\n");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(weight_ctx, metal);
    if (!buf) {
        fprintf(stderr, "[gguf] Failed to allocate Metal buffer for weights\n");
        delete model;
        return nullptr;
    }
    model->weight_buffer = buf;

    LOG("[gguf] Metal buffer: %.1f MB\n",
        ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));

    // Copy weight data from GGUF CPU tensors → Metal buffer tensors
    // For most tensors this is a direct memcpy of quantized data.
    // For concat QKV (full attention), we concatenate Q+K+V into one buffer.

    auto copy_weight = [&](ggml_tensor* dst, const char* src_name) -> bool {
        if (!dst) return true; // optional tensor
        ggml_tensor* src = find_tensor(data_ctx, src_name);
        if (!src) {
            fprintf(stderr, "[gguf] Cannot copy: source '%s' not found\n", src_name);
            return false;
        }
        if (ggml_nbytes(src) != ggml_nbytes(dst)) {
            fprintf(stderr, "[gguf] Size mismatch for '%s': src=%zu dst=%zu\n",
                    src_name, ggml_nbytes(src), ggml_nbytes(dst));
            return false;
        }
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
        return true;
    };

    // Global weights
    if (!copy_weight(model->token_embd, "token_embd.weight")) { delete model; return nullptr; }
    if (!copy_weight(model->output_norm, "output_norm.weight")) { delete model; return nullptr; }
    if (!tie_word_embeddings) {
        if (!copy_weight(model->lm_head, "output.weight")) { delete model; return nullptr; }
    }

    // Per-layer weights
    for (int L = 0; L < num_layers; L++) {
        auto& lw = model->layers[L];
        bool is_linear = (layer_types[L] == 0);

        if (is_linear) {
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", L);
            if (!copy_weight(lw.first_proj, name)) { delete model; return nullptr; }

            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", L);
            if (!copy_weight(lw.z_proj, name)) { delete model; return nullptr; }

            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", L);
            if (!copy_weight(lw.a_proj, name)) { delete model; return nullptr; }

            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", L);
            if (!copy_weight(lw.b_proj, name)) { delete model; return nullptr; }

            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", L);
            if (!copy_weight(lw.o_proj, name)) { delete model; return nullptr; }
        } else {
            // Full attention: concat Q+K+V into first_proj
            if (lw.first_proj) {
                snprintf(name, sizeof(name), "blk.%d.attn_q.weight", L);
                ggml_tensor* q = require_tensor(data_ctx, name);
                snprintf(name, sizeof(name), "blk.%d.attn_k.weight", L);
                ggml_tensor* k = require_tensor(data_ctx, name);
                snprintf(name, sizeof(name), "blk.%d.attn_v.weight", L);
                ggml_tensor* v = require_tensor(data_ctx, name);

                if (!q || !k || !v) { delete model; return nullptr; }

                size_t off = 0;
                ggml_backend_tensor_set(lw.first_proj, q->data, off, ggml_nbytes(q));
                off += ggml_nbytes(q);
                ggml_backend_tensor_set(lw.first_proj, k->data, off, ggml_nbytes(k));
                off += ggml_nbytes(k);
                ggml_backend_tensor_set(lw.first_proj, v->data, off, ggml_nbytes(v));
            }

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", L);
            if (!copy_weight(lw.o_proj, name)) { delete model; return nullptr; }
        }

        // FFN
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", L);
        if (!copy_weight(lw.gate_proj, name)) { delete model; return nullptr; }

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", L);
        if (!copy_weight(lw.up_proj, name)) { delete model; return nullptr; }

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", L);
        if (!copy_weight(lw.down_proj, name)) { delete model; return nullptr; }

        // Norms
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", L);
        if (!copy_weight(lw.input_norm, name)) { delete model; return nullptr; }

        snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", L);
        if (!copy_weight(lw.post_norm, name)) { delete model; return nullptr; }
    }

    double elapsed = total_timer.elapsed_ms();
    LOG("[gguf] Model loaded in %.1f ms (%.1f MB Metal buffer)\n",
        elapsed, ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));

    return model;
}

} // namespace ane_lm
