#include "qwen3.h"
#include "../../core/cpu_ops.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sys/stat.h>

namespace ane_lm {

using json = nlohmann::json;

static void apply_rope_qwen3(
    float* q, float* k,
    int n_q_heads, int n_kv_heads,
    int head_dim, int pos, float theta,
    const float* cos_row, const float* sin_row) {
    int half = head_dim / 2;
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + (size_t)h * head_dim : k + (size_t)(h - n_q_heads) * head_dim;
        for (int i = 0; i < half; i++) {
            float cos_a, sin_a;
            if (cos_row && sin_row) {
                cos_a = cos_row[i];
                sin_a = sin_row[i];
            } else {
                float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
                float angle = pos * freq;
                cos_a = cosf(angle);
                sin_a = sinf(angle);
            }
            float v0 = v[i];
            float v1 = v[i + half];
            v[i] = v0 * cos_a - v1 * sin_a;
            v[i + half] = v1 * cos_a + v0 * sin_a;
        }
    }
}

static int qwen3_prefill_batch_size() {
    static int batch = [] {
        const char* env = getenv("ANE_PREFILL_BATCH");
        int v = env ? atoi(env) : 4;
        if (v < 1) v = 1;
        if (v > ANE_SPATIAL) v = ANE_SPATIAL;
        return v;
    }();
    return batch;
}

static bool speculative_compare_enabled() {
    static bool enabled = getenv("ANE_SPECULATIVE_COMPARE") != nullptr;
    return enabled;
}

static void pack_w_lanes(std::vector<float>& packed,
                         const float* batch_data,
                         int batch,
                         int dim) {
    packed.assign((size_t)dim * ANE_SPATIAL, 0.0f);
    for (int c = 0; c < dim; c++) {
        size_t base = (size_t)c * ANE_SPATIAL;
        for (int b = 0; b < batch; b++) {
            packed[base + b] = batch_data[(size_t)b * dim + c];
        }
    }
}

static void unpack_w_lanes(float* batch_data,
                           const std::vector<float>& raw,
                           int batch,
                           int dim) {
    for (int c = 0; c < dim; c++) {
        size_t base = (size_t)c * ANE_SPATIAL;
        for (int b = 0; b < batch; b++) {
            batch_data[(size_t)b * dim + c] = raw[base + b];
        }
    }
}

static bool ane_matvec_batch(ANEKernel* kernel,
                             float* output,
                             const float* input,
                             int batch,
                             int in_dim,
                             int out_dim,
                             std::vector<float>& packed_in,
                             std::vector<float>& raw_out) {
    pack_w_lanes(packed_in, input, batch, in_dim);
    if (!ane_write_input_tiled(kernel, 0, packed_in.data(), 1, in_dim, 1, ANE_SPATIAL)) return false;
    float dummy = 0.0f;
    float* outputs[1] = { &dummy };
    int out_chs[1] = { 1 };
    if (!ane_eval_raw_outputs(kernel, outputs, out_chs)) return false;
    raw_out.resize((size_t)out_dim * ANE_SPATIAL);
    if (!ane_read_output_raw(kernel, 0, raw_out.data(), (int)raw_out.size())) return false;
    unpack_w_lanes(output, raw_out, batch, out_dim);
    return true;
}

static bool ane_binary_batch(ANEKernel* kernel,
                             int input0_dim,
                             int input1_dim,
                             int output_dim,
                             float* output,
                             const float* input0,
                             const float* input1,
                             int batch,
                             std::vector<float>& packed0,
                             std::vector<float>& packed1,
                             std::vector<float>& raw_out) {
    pack_w_lanes(packed0, input0, batch, input0_dim);
    pack_w_lanes(packed1, input1, batch, input1_dim);
    if (!ane_write_input_tiled(kernel, 0, packed0.data(), 1, input0_dim, 1, ANE_SPATIAL)) return false;
    if (!ane_write_input_tiled(kernel, 1, packed1.data(), 1, input1_dim, 1, ANE_SPATIAL)) return false;
    float dummy = 0.0f;
    float* outputs[1] = { &dummy };
    int out_chs[1] = { 1 };
    if (!ane_eval_raw_outputs(kernel, outputs, out_chs)) return false;
    raw_out.resize((size_t)output_dim * ANE_SPATIAL);
    if (!ane_read_output_raw(kernel, 0, raw_out.data(), (int)raw_out.size())) return false;
    unpack_w_lanes(output, raw_out, batch, output_dim);
    return true;
}

Qwen3Args Qwen3Args::from_json(const json& j) {
    Qwen3Args args;

    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.max_position_embeddings = tc.value("max_position_embeddings", args.max_position_embeddings);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);

    args.head_dim = tc.value("head_dim", args.head_dim);
    if (!tc.contains("head_dim") && args.num_attention_heads > 0) {
        args.head_dim = args.hidden_size / args.num_attention_heads;
    }

    args.tie_word_embeddings = tc.value(
        "tie_word_embeddings",
        j.value("tie_word_embeddings", args.tie_word_embeddings));

    if (tc.contains("rope_parameters") && tc["rope_parameters"].is_object()) {
        const auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
    }

    return args;
}

Qwen3Model::~Qwen3Model() {
    free(embed_tokens_);
    free(final_norm_);
    if (!tie_word_embeddings_) {
        free(lm_head_);
    }

    free(x_);
    free(x_norm_);
    free(logits_);
    free(speculative_logits_batch_);
    free(draft_x_);
    free(draft_x_norm_);
    free(draft_logits_);
    free(draft_logits_history_);
    free(scratch_qkv_);
    free(draft_scratch_qkv_);
    free(scratch_attn_);
    free(draft_scratch_attn_);
    free(rope_cos_);
    free(rope_sin_);

    for (auto& lw : layers_) {
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);
        free(lw.q_norm);
        free(lw.k_norm);
    }

    for (auto& kv : kv_caches_) {
        free(kv.k_cache);
        free(kv.v_cache);
    }
    for (auto& kv : draft_kv_caches_) {
        free(kv.k_cache);
        free(kv.v_cache);
    }

    for (auto& lk : ane_layers_) {
        ane_free_layer(&lk);
    }

    free_lm_head_ane();
}

void Qwen3Model::reset() {
    speculative_pending_ = false;
    speculative_pending_batch_ = 0;
    speculative_ready_ = false;
    draft_next_pos_ = 0;
    for (auto& kv : kv_caches_) {
        kv.len = 0;
        kv.start = 0;
        memset(kv.k_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
        memset(kv.v_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
    }
    for (auto& kv : draft_kv_caches_) {
        kv.len = 0;
        kv.start = 0;
        memset(kv.k_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
        memset(kv.v_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
    }
}

void Qwen3Model::apply_args(const Qwen3Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = head_dim_;
    max_pos_ = args.max_position_embeddings;
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    tie_word_embeddings_ = args.tie_word_embeddings;

    q_proj_dim_ = num_q_heads_ * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = q_proj_dim_;
}

bool Qwen3Model::load(const std::string& model_dir) {
    // 1. Read config.json and parse args
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen3Args args = Qwen3Args::from_json(j);
    apply_args(args);

    // 2. Open model weights (single-file or sharded)
    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // Infer dims from safetensors
    const SFTensor* embed = sf->find("model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid model.embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];
    if (head_dim_ <= 0 && num_q_heads_ > 0) {
        head_dim_ = hidden_size_ / num_q_heads_;
    }

    rot_dim_ = head_dim_;
    q_proj_dim_ = num_q_heads_ * head_dim_;
    kv_proj_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = q_proj_dim_;

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    const char* draft_layers_env = getenv("ANE_SPECULATIVE_DRAFT_LAYERS");
    draft_layers_ = draft_layers_env ? atoi(draft_layers_env) : 2;
    if (draft_layers_ < 1) draft_layers_ = 1;
    if (draft_layers_ > num_layers_) draft_layers_ = num_layers_;

    const char* spec_batch_env = getenv("ANE_SPECULATIVE_BATCH");
    speculative_batch_size_ = spec_batch_env ? atoi(spec_batch_env) : MAX_SPECULATIVE_BATCH;
    if (speculative_batch_size_ < 1) speculative_batch_size_ = 1;
    if (speculative_batch_size_ > MAX_SPECULATIVE_BATCH) speculative_batch_size_ = MAX_SPECULATIVE_BATCH;

    // 3. Init ANE
    ane_init();

    x_ = (float*)calloc(hidden_size_, sizeof(float));
    x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    logits_ = (float*)calloc(vocab_size_, sizeof(float));
    speculative_logits_batch_ = (float*)calloc((size_t)MAX_SPECULATIVE_BATCH * vocab_size_, sizeof(float));
    draft_x_ = (float*)calloc(hidden_size_, sizeof(float));
    draft_x_norm_ = (float*)calloc(hidden_size_, sizeof(float));
    draft_logits_ = (float*)calloc(vocab_size_, sizeof(float));
    draft_logits_history_ = (float*)calloc((size_t)(MAX_SPECULATIVE_BATCH + 1) * vocab_size_, sizeof(float));
    scratch_qkv_ = (float*)calloc((size_t)q_proj_dim_ + 2 * kv_proj_dim_, sizeof(float));
    draft_scratch_qkv_ = (float*)calloc((size_t)q_proj_dim_ + 2 * kv_proj_dim_, sizeof(float));
    scratch_attn_ = (float*)calloc(std::max(full_out_dim_, hidden_size_), sizeof(float));
    draft_scratch_attn_ = (float*)calloc(std::max(full_out_dim_, hidden_size_), sizeof(float));

    int half_rot = rot_dim_ / 2;
    rope_cache_len_ = std::min(std::max(max_pos_, 1), 16384);
    rope_cos_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));
    rope_sin_ = (float*)calloc((size_t)rope_cache_len_ * half_rot, sizeof(float));

    // Precompute RoPE trig table for common context lengths.
    if (rope_cos_ && rope_sin_ && half_rot > 0) {
        std::vector<float> inv_freq((size_t)half_rot);
        for (int j = 0, i = 0; i < rot_dim_; i += 2, j++) {
            inv_freq[(size_t)j] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < rope_cache_len_; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j = 0; j < half_rot; j++) {
                float angle = pos * inv_freq[(size_t)j];
                cos_row[j] = cosf(angle);
                sin_row[j] = sinf(angle);
            }
        }
    }

    layers_.resize(num_layers_);
    kv_caches_.resize(num_layers_);
    ane_layers_.resize(num_layers_);
    draft_kv_caches_.resize(draft_layers_);
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    full_spec_snapshots_.assign((size_t)num_layers_ * (MAX_SPECULATIVE_BATCH + 1), {});
    full_spec_saved_k_.assign((size_t)num_layers_ * MAX_SPECULATIVE_BATCH * kv_stride, 0.0f);
    full_spec_saved_v_.assign((size_t)num_layers_ * MAX_SPECULATIVE_BATCH * kv_stride, 0.0f);
    full_spec_saved_slot_.assign((size_t)num_layers_ * MAX_SPECULATIVE_BATCH, -1);
    full_spec_saved_valid_.assign((size_t)num_layers_ * MAX_SPECULATIVE_BATCH, 0);
    draft_spec_snapshots_.assign((size_t)draft_layers_ * (MAX_SPECULATIVE_BATCH + 1), {});
    draft_spec_saved_k_.assign((size_t)draft_layers_ * MAX_SPECULATIVE_BATCH * kv_stride, 0.0f);
    draft_spec_saved_v_.assign((size_t)draft_layers_ * MAX_SPECULATIVE_BATCH * kv_stride, 0.0f);
    draft_spec_saved_slot_.assign((size_t)draft_layers_ * MAX_SPECULATIVE_BATCH, -1);
    draft_spec_saved_valid_.assign((size_t)draft_layers_ * MAX_SPECULATIVE_BATCH, 0);

    for (int L = 0; L < num_layers_; L++) {
        auto& kv = kv_caches_[L];
        kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.len = 0;
        kv.start = 0;
        kv.capacity = KV_CACHE_CAPACITY;
    }
    for (int L = 0; L < draft_layers_; L++) {
        auto& kv = draft_kv_caches_[L];
        kv.k_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.v_cache = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
        kv.len = 0;
        kv.start = 0;
        kv.capacity = KV_CACHE_CAPACITY;
    }

    // 4. Load weights + compile ANE kernels
    if (!load_weights(sf.get())) {
        return false;
    }

    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) {
        return false;
    }

    return true;
}

bool Qwen3Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32(
        "model.embed_tokens.weight", (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    if (tie_word_embeddings_) {
        lm_head_ = embed_tokens_;
    } else {
        lm_head_ = sf->load_bf16_to_f32("lm_head.weight", (int64_t)vocab_size_ * hidden_size_);
        if (!lm_head_) return false;
    }

    final_norm_ = sf->load_bf16_to_f32("model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_bf16_to_f32(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", L);
        lw.q_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.q_norm) return false;

        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", L);
        lw.k_norm = sf->load_bf16_to_f32(name, head_dim_);
        if (!lw.k_norm) return false;
    }

    LOG("All Qwen3 weights loaded successfully\n");
    return true;
}

// Convert tensor name to blob path: "a.b.c" -> "<dir>/a/b/c.bin"
static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen3Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling Qwen3 ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");

    char name[256], name2[256], name3[256], name4[256];

    // --- Cross-layer fusion strategy ---
    // Layer 0: standalone QKV (first_proj) — no previous layer to fuse with
    // Layers 0..N-2: oproj_norm_qkv[L] fuses O_proj_L + input_ln_{L+1} + QKV_{L+1}
    // Layer N-1: standalone oproj_add — no next layer to fuse with
    // All layers: FFN (unchanged)

    int fused_count = 0;
    int same_layer_fused_count = 0;
    bool compile_cross_layer = getenv("CROSS_LAYER_FUSION") != nullptr;
    bool compile_same_layer = getenv("FUSE_OPROJ_FFN") != nullptr;

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d...\r", L + 1, num_layers_);

        // --- QKV projection ---
        if (L == 0 || !compile_cross_layer) {
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(name2, sizeof(name2), "model.layers.%d.self_attn.k_proj.weight", L);
            snprintf(name3, sizeof(name3), "model.layers.%d.self_attn.v_proj.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                    blob_path(blob_dir, name), q_proj_dim_,
                    blob_path(blob_dir, name2), kv_proj_dim_,
                    blob_path(blob_dir, name3), kv_proj_dim_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_3(
                    sf->get_bf16_ptr(name), q_proj_dim_,
                    sf->get_bf16_ptr(name2), kv_proj_dim_,
                    sf->get_bf16_ptr(name3), kv_proj_dim_, hidden_size_);
            }
            if (!ane_layers_[L].first_proj) {
                fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
                return false;
            }
        }

        // --- O_proj: fused cross-layer or standalone ---
        if (L < num_layers_ - 1 && compile_cross_layer) {
            // Try cross-layer fusion: O_proj_L + RMSNorm_{L+1} + QKV_{L+1}
            int nextL = L + 1;
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", L);
            snprintf(name2, sizeof(name2), "model.layers.%d.self_attn.q_proj.weight", nextL);
            snprintf(name3, sizeof(name3), "model.layers.%d.self_attn.k_proj.weight", nextL);
            snprintf(name4, sizeof(name4), "model.layers.%d.self_attn.v_proj.weight", nextL);

            if (use_blobs) {
                ane_layers_[L].oproj_norm_qkv = ane_compile_oproj_norm_qkv_blob(
                    blob_path(blob_dir, name),
                    layers_[nextL].input_layernorm,
                    blob_path(blob_dir, name2), q_proj_dim_,
                    blob_path(blob_dir, name3), kv_proj_dim_,
                    blob_path(blob_dir, name4), kv_proj_dim_,
                    hidden_size_, full_out_dim_, rms_eps_);
            } else {
                ane_layers_[L].oproj_norm_qkv = ane_compile_oproj_norm_qkv(
                    sf->get_bf16_ptr(name),
                    layers_[nextL].input_layernorm,
                    sf->get_bf16_ptr(name2), q_proj_dim_,
                    sf->get_bf16_ptr(name3), kv_proj_dim_,
                    sf->get_bf16_ptr(name4), kv_proj_dim_,
                    hidden_size_, full_out_dim_, rms_eps_);
            }

            if (ane_layers_[L].oproj_norm_qkv) {
                fused_count++;
            } else {
                fprintf(stderr, "ANE oproj_norm_qkv failed for layer %d, falling back\n", L);
            }

            // Always compile standalone oproj_add + next-layer QKV as fallback
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", L);
            if (use_blobs) {
                ane_layers_[L].oproj_add = ane_compile_oproj_add_blob(
                    blob_path(blob_dir, name), hidden_size_, full_out_dim_);
            } else {
                ane_layers_[L].oproj_add = ane_compile_oproj_add(
                    sf->get_bf16_ptr(name), hidden_size_, full_out_dim_);
            }
            if (!ane_layers_[nextL].first_proj) {
                snprintf(name2, sizeof(name2), "model.layers.%d.self_attn.q_proj.weight", nextL);
                snprintf(name3, sizeof(name3), "model.layers.%d.self_attn.k_proj.weight", nextL);
                snprintf(name4, sizeof(name4), "model.layers.%d.self_attn.v_proj.weight", nextL);
                if (use_blobs) {
                    ane_layers_[nextL].first_proj = ane_compile_fused_3_blob(
                        blob_path(blob_dir, name2), q_proj_dim_,
                        blob_path(blob_dir, name3), kv_proj_dim_,
                        blob_path(blob_dir, name4), kv_proj_dim_, hidden_size_);
                } else {
                    ane_layers_[nextL].first_proj = ane_compile_fused_3(
                        sf->get_bf16_ptr(name2), q_proj_dim_,
                        sf->get_bf16_ptr(name3), kv_proj_dim_,
                        sf->get_bf16_ptr(name4), kv_proj_dim_, hidden_size_);
                }
            }
        } else {
            // Standalone oproj_add (no cross-layer fusion)
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", L);
            if (use_blobs) {
                ane_layers_[L].oproj_add = ane_compile_oproj_add_blob(
                    blob_path(blob_dir, name), hidden_size_, full_out_dim_);
            } else {
                ane_layers_[L].oproj_add = ane_compile_oproj_add(
                    sf->get_bf16_ptr(name), hidden_size_, full_out_dim_);
            }
            if (!ane_layers_[L].oproj_add) {
                fprintf(stderr, "ANE oproj_add compile failed for layer %d\n", L);
                return false;
            }
        }

        // --- FFN (unchanged) ---
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.layers.%d.mlp.up_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.layers.%d.mlp.down_proj.weight", L);

        int ffn_chunks = ane_ffn_chunk_count(hidden_size_, intermediate_size_);
        if (ffn_chunks <= 1) {
            if (compile_same_layer && !compile_cross_layer) {
                snprintf(name4, sizeof(name4), "model.layers.%d.self_attn.o_proj.weight", L);
                ane_layers_[L].fused_oproj_ffn = ane_compile_fused_oproj_ffn(
                    sf->get_bf16_ptr(name4),
                    sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2), sf->get_bf16_ptr(name3),
                    layers_[L].post_attention_layernorm,
                    hidden_size_, full_out_dim_, intermediate_size_, rms_eps_);
                if (ane_layers_[L].fused_oproj_ffn) {
                    same_layer_fused_count++;
                } else {
                    fprintf(stderr, "ANE fused_oproj_ffn compile failed for layer %d, falling back\n", L);
                }
            }

            // Try fused FFN + residual add first
            if (use_blobs) {
                ane_layers_[L].ffn_resadd = ane_compile_fused_ffn_resadd_blob(
                    blob_path(blob_dir, name), blob_path(blob_dir, name2),
                    blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
            } else {
                ane_layers_[L].ffn_resadd = ane_compile_fused_ffn_resadd(
                    sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                    sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
            }
            if (!ane_layers_[L].ffn_resadd) {
                // Fall back to FFN without residual add
                if (use_blobs) {
                    ane_layers_[L].fused_ffn = ane_compile_fused_ffn_blob(
                        blob_path(blob_dir, name), blob_path(blob_dir, name2),
                        blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
                } else {
                    ane_layers_[L].fused_ffn = ane_compile_fused_ffn(
                        sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                        sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
                }
            }
        }
        if (!ane_layers_[L].ffn_resadd && !ane_layers_[L].fused_ffn) {
            if (ffn_chunks <= 1) ffn_chunks = 2;
            if (L == 0) LOG("  Using chunked FFN (%d chunks, inter=%d)\n", ffn_chunks, intermediate_size_);
            if (!use_blobs) {
                if (!ane_compile_chunked_ffn(&ane_layers_[L].chunked_ffn,
                        sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                        sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_, ffn_chunks)) {
                    fprintf(stderr, "ANE chunked FFN compile failed for layer %d\n", L);
                    return false;
                }
            } else {
                if (!ane_compile_chunked_ffn_blob(&ane_layers_[L].chunked_ffn,
                        blob_path(blob_dir, name), blob_path(blob_dir, name2),
                        blob_path(blob_dir, name3), hidden_size_, intermediate_size_, ffn_chunks)) {
                    fprintf(stderr, "ANE chunked FFN blob compile failed for layer %d\n", L);
                    return false;
                }
            }
        }
    }

    int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d, cross-layer fused=%d/%d, same-layer fused=%d/%d)\n",
        compiled + cached, compiled, cached, fused_count, num_layers_ - 1,
        same_layer_fused_count, num_layers_);

    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen3Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();
    const char* lm_name = tie_word_embeddings_ ? "model.embed_tokens.weight" : "lm_head.weight";

    const uint16_t* lm_bf16 = sf->get_bf16_ptr(lm_name);
    if (!lm_bf16) {
        fprintf(stderr, "ANE LM head: missing BF16 weights for %s\n", lm_name);
        return false;
    }

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;

    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: compiling %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = vocab_size_ - offset;
        if (rows > chunk) rows = chunk;

        LOG("    LM head chunk %d/%d...\r", c + 1, chunks);

        // For blob mode we still use BF16 pointer here because lm_head is chunked dynamically.
        (void)use_blobs;
        const uint16_t* chunk_w = lm_bf16 + (int64_t)offset * hidden_size_;
        lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        if (!lm_head_kernels_[c]) {
            fprintf(stderr, "\nANE LM head: compile failed at chunk %d/%d\n", c + 1, chunks);
            free_lm_head_ane();
            return false;
        }
    }

    LOG("    LM head chunk %d/%d done          \n", chunks, chunks);
    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Qwen3Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) {
        ane_free(k);
    }
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

void Qwen3Model::compute_logits_from_hidden_into(const float* hidden, float* out) {
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], out + offset, hidden, hidden_size_, rows)) {
                fprintf(stderr, "ANE LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(out, lm_head_, hidden, vocab_size_, hidden_size_);
        }
    } else {
        matvec(out, lm_head_, hidden, vocab_size_, hidden_size_);
    }
}

void Qwen3Model::compute_logits_batch_from_hidden(const float* hidden_batch,
                                                  int batch,
                                                  float* logits_batch,
                                                  std::vector<float>& packed_in,
                                                  std::vector<float>& raw_out) {
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        std::vector<float> chunk_logits;
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            chunk_logits.resize((size_t)batch * rows);
            if (!ane_matvec_batch(lm_head_kernels_[c], chunk_logits.data(), hidden_batch,
                                  batch, hidden_size_, rows, packed_in, raw_out)) {
                fprintf(stderr, "ANE LM head batched eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
            for (int b = 0; b < batch; b++) {
                memcpy(logits_batch + (size_t)b * vocab_size_ + offset,
                       chunk_logits.data() + (size_t)b * rows,
                       (size_t)rows * sizeof(float));
            }
        }
        if (ok) return;
        free_lm_head_ane();
    }

    for (int b = 0; b < batch; b++) {
        compute_logits_from_hidden_into(hidden_batch + (size_t)b * hidden_size_,
                                        logits_batch + (size_t)b * vocab_size_);
    }
}

float* Qwen3Model::compute_logits_from_hidden(const float* hidden) {
    if (hidden != x_) memcpy(x_, hidden, (size_t)hidden_size_ * sizeof(float));
    compute_logits_from_hidden_into(x_, logits_);
    return logits_;
}

int Qwen3Model::argmax_token(const float* logits, int vocab_size) const {
    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

void Qwen3Model::begin_speculative_window(std::vector<KVCache>& caches,
                                          std::vector<KVCacheSnapshot>& snapshots,
                                          std::vector<uint8_t>& saved_valid,
                                          int num_cache_layers) {
    for (int L = 0; L < num_cache_layers; L++) {
        snapshots[(size_t)L * (MAX_SPECULATIVE_BATCH + 1)] = { caches[L].len, caches[L].start };
    }
    std::fill(saved_valid.begin(), saved_valid.end(), 0);
}

void Qwen3Model::record_speculative_insert(std::vector<KVCache>& caches,
                                           std::vector<KVCacheSnapshot>& snapshots,
                                           std::vector<float>& saved_k,
                                           std::vector<float>& saved_v,
                                           std::vector<int>& saved_slot,
                                           std::vector<uint8_t>& saved_valid,
                                           int num_cache_layers,
                                           int layer_index,
                                           int token_index,
                                           const float* k_raw,
                                           const float* v_raw) {
    auto& cache = caches[layer_index];
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    size_t op_index = (size_t)layer_index * MAX_SPECULATIVE_BATCH + token_index;

    bool will_overwrite = (cache.len >= cache.capacity);
    if (will_overwrite) {
        int overwritten_slot = cache.start;
        saved_slot[op_index] = overwritten_slot;
        saved_valid[op_index] = 1;
        float* saved_k_ptr = saved_k.data() + op_index * kv_stride;
        float* saved_v_ptr = saved_v.data() + op_index * kv_stride;
        memcpy(saved_k_ptr, cache.k_cache + (size_t)overwritten_slot * kv_stride, kv_stride * sizeof(float));
        memcpy(saved_v_ptr, cache.v_cache + (size_t)overwritten_slot * kv_stride, kv_stride * sizeof(float));
    }

    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }

    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
    snapshots[(size_t)layer_index * (MAX_SPECULATIVE_BATCH + 1) + (token_index + 1)] = { cache.len, cache.start };
}

void Qwen3Model::finalize_speculative_window(std::vector<KVCache>& caches,
                                             std::vector<KVCacheSnapshot>& snapshots,
                                             std::vector<float>& saved_k,
                                             std::vector<float>& saved_v,
                                             std::vector<int>& saved_slot,
                                             std::vector<uint8_t>& saved_valid,
                                             int num_cache_layers,
                                             int accepted_tokens,
                                             int pending_batch) {
    if (accepted_tokens >= pending_batch) return;

    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    for (int L = 0; L < num_cache_layers; L++) {
        KVCacheSnapshot snap = snapshots[(size_t)L * (MAX_SPECULATIVE_BATCH + 1) + accepted_tokens];
        caches[L].len = snap.len;
        caches[L].start = snap.start;
    }

    for (int token_index = pending_batch - 1; token_index >= accepted_tokens; token_index--) {
        for (int L = 0; L < num_cache_layers; L++) {
            size_t op_index = (size_t)L * MAX_SPECULATIVE_BATCH + token_index;
            if (!saved_valid[op_index]) continue;
            int slot = saved_slot[op_index];
            float* saved_k_ptr = saved_k.data() + op_index * kv_stride;
            float* saved_v_ptr = saved_v.data() + op_index * kv_stride;
            memcpy(caches[L].k_cache + (size_t)slot * kv_stride, saved_k_ptr, kv_stride * sizeof(float));
            memcpy(caches[L].v_cache + (size_t)slot * kv_stride, saved_v_ptr, kv_stride * sizeof(float));
        }
    }
}

float* Qwen3Model::draft_forward(int token_id, int pos, bool speculative_insert, int token_index) {
    memcpy(draft_x_, embed_tokens_ + (int64_t)token_id * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = draft_scratch_attn_;
    int qkv_total = q_proj_dim_ + 2 * kv_proj_dim_;

    for (int L = 0; L < draft_layers_; L++) {
        rmsnorm(draft_x_norm_, draft_x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);
        if (!ane_matvec(ane_layers_[L].first_proj, draft_scratch_qkv_, draft_x_norm_,
                        hidden_size_, qkv_total)) {
            fprintf(stderr, "ANE draft first_proj eval failed at layer %d\n", L);
            return nullptr;
        }

        float* q_raw = draft_scratch_qkv_;
        float* k_raw = draft_scratch_qkv_ + q_proj_dim_;
        float* v_raw = draft_scratch_qkv_ + q_proj_dim_ + kv_proj_dim_;

        for (int h = 0; h < num_q_heads_; h++) {
            float* qh = q_raw + (size_t)h * head_dim_;
            rmsnorm(qh, qh, layers_[L].q_norm, head_dim_, rms_eps_);
        }
        for (int h = 0; h < num_kv_heads_; h++) {
            float* kh = k_raw + (size_t)h * head_dim_;
            rmsnorm(kh, kh, layers_[L].k_norm, head_dim_, rms_eps_);
        }

        const float* rope_cos_row = nullptr;
        const float* rope_sin_row = nullptr;
        if (pos >= 0 && pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
            int half_rot = rot_dim_ / 2;
            rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
            rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
        }
        apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                         head_dim_, pos, rope_theta_, rope_cos_row, rope_sin_row);

        if (speculative_insert) {
            record_speculative_insert(draft_kv_caches_, draft_spec_snapshots_,
                                      draft_spec_saved_k_, draft_spec_saved_v_,
                                      draft_spec_saved_slot_, draft_spec_saved_valid_,
                                      draft_layers_, L, token_index, k_raw, v_raw);
        } else {
            auto& cache = draft_kv_caches_[L];
            int slot;
            if (cache.len < cache.capacity) {
                slot = cache.start + cache.len;
                if (slot >= cache.capacity) slot -= cache.capacity;
                cache.len++;
            } else {
                slot = cache.start;
                cache.start++;
                if (cache.start >= cache.capacity) cache.start = 0;
            }
            size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
            memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
            memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
        }
        auto& cache = draft_kv_caches_[L];
        gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache,
                      num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                      cache.start, cache.len, cache.capacity);

        if (ane_layers_[L].oproj_add) {
            if (!ane_eval_oproj_add(ane_layers_[L].oproj_add,
                                    draft_x_, pre_oproj, draft_x_, full_out_dim_, hidden_size_)) {
                fprintf(stderr, "ANE draft oproj_add eval failed at layer %d\n", L);
                return nullptr;
            }
        } else {
            if (!ane_matvec(ane_layers_[L].o_proj, draft_x_norm_, pre_oproj, full_out_dim_, hidden_size_)) {
                fprintf(stderr, "ANE draft o_proj eval failed at layer %d\n", L);
                return nullptr;
            }
            for (int i = 0; i < hidden_size_; i++) draft_x_[i] += draft_x_norm_[i];
        }

        rmsnorm(draft_x_norm_, draft_x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
        if (ane_layers_[L].ffn_resadd) {
            if (!ane_eval_fused_ffn_resadd(ane_layers_[L].ffn_resadd, draft_x_, draft_x_norm_, draft_x_, hidden_size_)) {
                fprintf(stderr, "ANE draft ffn_resadd eval failed at layer %d\n", L);
                return nullptr;
            }
        } else {
            float* mlp_out = draft_scratch_attn_;
            if (ane_layers_[L].fused_ffn) {
                if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, draft_x_norm_, hidden_size_, hidden_size_)) {
                    fprintf(stderr, "ANE draft fused_ffn eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
                if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp_out, draft_x_norm_)) {
                    fprintf(stderr, "ANE draft chunked_ffn eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else {
                fprintf(stderr, "No draft FFN kernel for layer %d\n", L);
                return nullptr;
            }
            for (int i = 0; i < hidden_size_; i++) draft_x_[i] += mlp_out[i];
        }
    }

    rmsnorm(draft_x_norm_, draft_x_, final_norm_, hidden_size_, rms_eps_);
    compute_logits_from_hidden_into(draft_x_norm_, draft_logits_);
    return draft_logits_;
}

bool Qwen3Model::forward_full_attn_core(int L, float* x, float* pre_oproj, int pos) {
    auto& lw = layers_[L];
    auto& cache = kv_caches_[L];

    float* qkv_buf = scratch_qkv_;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x,
                    hidden_size_, q_proj_dim_ + 2 * kv_proj_dim_)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d\n", L);
        return false;
    }

    float* q_raw = qkv_buf;
    float* k_raw = qkv_buf + q_proj_dim_;
    float* v_raw = qkv_buf + q_proj_dim_ + kv_proj_dim_;

    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_raw + (size_t)h * head_dim_;
        rmsnorm(qh, qh, lw.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        float* kh = k_raw + (size_t)h * head_dim_;
        rmsnorm(kh, kh, lw.k_norm, head_dim_, rms_eps_);
    }

    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }

    apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                     head_dim_, pos, rope_theta_, rope_cos_row, rope_sin_row);

    int slot;
    if (cache.len < cache.capacity) {
        slot = cache.start + cache.len;
        if (slot >= cache.capacity) slot -= cache.capacity;
        cache.len++;
    } else {
        slot = cache.start;
        cache.start++;
        if (cache.start >= cache.capacity) cache.start = 0;
    }

    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

    gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache,
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                  cache.start, cache.len, cache.capacity);

    return true;
}

// Forward pass profiling accumulators (print with PROFILE_FORWARD=1 env var)
static struct FwdProfile {
    double rmsnorm_ms = 0, qkv_ms = 0, qk_norm_ms = 0, rope_ms = 0;
    double attention_ms = 0, o_proj_ms = 0, residual_ms = 0, ffn_ms = 0;
    double lm_head_ms = 0, total_ms = 0;
    int count = 0;
    bool enabled = false;
    bool checked = false;

    void check() {
        if (!checked) {
            checked = true;
            enabled = getenv("PROFILE_FORWARD") != nullptr;
        }
    }

    void print() {
        if (!enabled || count == 0) return;
        double n = (double)count;
        fprintf(stderr, "\n=== Forward pass profile (%d tokens, avg per token) ===\n", count);
        fprintf(stderr, "  rmsnorm:   %6.2f ms  (%.1f%%)\n", rmsnorm_ms/n, 100*rmsnorm_ms/total_ms);
        fprintf(stderr, "  qkv_proj:  %6.2f ms  (%.1f%%)\n", qkv_ms/n, 100*qkv_ms/total_ms);
        fprintf(stderr, "  qk_norm:   %6.2f ms  (%.1f%%)\n", qk_norm_ms/n, 100*qk_norm_ms/total_ms);
        fprintf(stderr, "  rope:      %6.2f ms  (%.1f%%)\n", rope_ms/n, 100*rope_ms/total_ms);
        fprintf(stderr, "  attention: %6.2f ms  (%.1f%%)\n", attention_ms/n, 100*attention_ms/total_ms);
        fprintf(stderr, "  o_proj:    %6.2f ms  (%.1f%%)\n", o_proj_ms/n, 100*o_proj_ms/total_ms);
        fprintf(stderr, "  residual:  %6.2f ms  (%.1f%%)\n", residual_ms/n, 100*residual_ms/total_ms);
        fprintf(stderr, "  ffn:       %6.2f ms  (%.1f%%)\n", ffn_ms/n, 100*ffn_ms/total_ms);
        fprintf(stderr, "  lm_head:   %6.2f ms  (%.1f%%)\n", lm_head_ms/n, 100*lm_head_ms/total_ms);
        fprintf(stderr, "  TOTAL:     %6.2f ms  (sum of above)\n", total_ms/n);
        fprintf(stderr, "==============================================\n");
    }

    ~FwdProfile() { print(); }
} g_fwd_prof;

float* Qwen3Model::prefill(const std::vector<int>& token_ids, int start_pos) {
    if (token_ids.empty()) return nullptr;
    Timer t_total;

    static bool fusion_enabled = getenv("CROSS_LAYER_FUSION") != nullptr;
    static bool same_layer_fusion_enabled = getenv("FUSE_OPROJ_FFN") != nullptr;
    int batch_size = qwen3_prefill_batch_size();
    if (batch_size <= 1 || token_ids.size() == 1 || fusion_enabled || same_layer_fusion_enabled) {
        return LLMModel::prefill(token_ids, start_pos);
    }

    g_fwd_prof.check();

    int qkv_total = q_proj_dim_ + 2 * kv_proj_dim_;
    std::vector<float> x_batch((size_t)batch_size * hidden_size_);
    std::vector<float> x_norm_batch((size_t)batch_size * hidden_size_);
    std::vector<float> qkv_batch((size_t)batch_size * qkv_total);
    std::vector<float> pre_oproj_batch((size_t)batch_size * full_out_dim_);
    std::vector<float> mlp_batch((size_t)batch_size * hidden_size_);
    std::vector<float> packed0;
    std::vector<float> packed1;
    std::vector<float> raw_out;

    float* last_hidden = nullptr;

    for (int base = 0; base < (int)token_ids.size(); base += batch_size) {
        int batch = std::min(batch_size, (int)token_ids.size() - base);
        for (int b = 0; b < batch; b++) {
            memcpy(x_batch.data() + (size_t)b * hidden_size_,
                   embed_tokens_ + (int64_t)token_ids[(size_t)base + b] * hidden_size_,
                   (size_t)hidden_size_ * sizeof(float));
        }

        for (int L = 0; L < num_layers_; L++) {
            {
                Timer t;
                for (int b = 0; b < batch; b++) {
                    rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                            x_batch.data() + (size_t)b * hidden_size_,
                            layers_[L].input_layernorm, hidden_size_, rms_eps_);
                }
                if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();
            }

            {
                Timer t;
                if (!ane_matvec_batch(ane_layers_[L].first_proj,
                                      qkv_batch.data(), x_norm_batch.data(),
                                      batch, hidden_size_, qkv_total,
                                      packed0, raw_out)) {
                    fprintf(stderr, "ANE first_proj batched eval failed at layer %d\n", L);
                    return nullptr;
                }
                if (g_fwd_prof.enabled) g_fwd_prof.qkv_ms += t.elapsed_ms();
            }

            {
                Timer t;
                for (int b = 0; b < batch; b++) {
                    float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
                    float* k_raw = q_raw + q_proj_dim_;
                    for (int h = 0; h < num_q_heads_; h++) {
                        float* qh = q_raw + (size_t)h * head_dim_;
                        rmsnorm(qh, qh, layers_[L].q_norm, head_dim_, rms_eps_);
                    }
                    for (int h = 0; h < num_kv_heads_; h++) {
                        float* kh = k_raw + (size_t)h * head_dim_;
                        rmsnorm(kh, kh, layers_[L].k_norm, head_dim_, rms_eps_);
                    }
                }
                if (g_fwd_prof.enabled) g_fwd_prof.qk_norm_ms += t.elapsed_ms();
            }

            {
                Timer t;
                for (int b = 0; b < batch; b++) {
                    int pos = start_pos + base + b;
                    float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
                    float* k_raw = q_raw + q_proj_dim_;
                    const float* rope_cos_row = nullptr;
                    const float* rope_sin_row = nullptr;
                    if (pos >= 0 && pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
                        int half_rot = rot_dim_ / 2;
                        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
                        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
                    }
                    apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                                     head_dim_, pos, rope_theta_, rope_cos_row, rope_sin_row);
                }
                if (g_fwd_prof.enabled) g_fwd_prof.rope_ms += t.elapsed_ms();
            }

            {
                Timer t;
                for (int b = 0; b < batch; b++) {
                    float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
                    float* k_raw = q_raw + q_proj_dim_;
                    float* v_raw = q_raw + q_proj_dim_ + kv_proj_dim_;
                    auto& cache = kv_caches_[L];
                    int slot;
                    if (cache.len < cache.capacity) {
                        slot = cache.start + cache.len;
                        if (slot >= cache.capacity) slot -= cache.capacity;
                        cache.len++;
                    } else {
                        slot = cache.start;
                        cache.start++;
                        if (cache.start >= cache.capacity) cache.start = 0;
                    }
                    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
                    memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
                    memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
                    gqa_attention(pre_oproj_batch.data() + (size_t)b * full_out_dim_,
                                  q_raw, cache.k_cache, cache.v_cache,
                                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                                  cache.start, cache.len, cache.capacity);
                }
                if (g_fwd_prof.enabled) g_fwd_prof.attention_ms += t.elapsed_ms();
            }

            {
                Timer t;
                if (ane_layers_[L].oproj_add) {
                    if (!ane_binary_batch(ane_layers_[L].oproj_add,
                                          full_out_dim_, hidden_size_, hidden_size_,
                                          x_batch.data(),
                                          pre_oproj_batch.data(), x_batch.data(), batch,
                                          packed0, packed1, raw_out)) {
                        fprintf(stderr, "ANE oproj_add batched eval failed at layer %d\n", L);
                        return nullptr;
                    }
                } else {
                    for (int b = 0; b < batch; b++) {
                        float* x = x_batch.data() + (size_t)b * hidden_size_;
                        float* pre = pre_oproj_batch.data() + (size_t)b * full_out_dim_;
                        if (ane_layers_[L].o_proj) {
                            if (!ane_matvec(ane_layers_[L].o_proj, mlp_batch.data() + (size_t)b * hidden_size_,
                                            pre, full_out_dim_, hidden_size_)) {
                                fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                                return nullptr;
                            }
                            for (int i = 0; i < hidden_size_; i++) x[i] += mlp_batch[(size_t)b * hidden_size_ + i];
                        }
                    }
                }
                if (g_fwd_prof.enabled) g_fwd_prof.o_proj_ms += t.elapsed_ms();
            }

            {
                Timer t;
                for (int b = 0; b < batch; b++) {
                    rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                            x_batch.data() + (size_t)b * hidden_size_,
                            layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
                }
                if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();
            }

            {
                Timer t;
                if (ane_layers_[L].ffn_resadd) {
                    if (!ane_binary_batch(ane_layers_[L].ffn_resadd,
                                          hidden_size_, hidden_size_, hidden_size_,
                                          x_batch.data(),
                                          x_norm_batch.data(), x_batch.data(), batch,
                                          packed0, packed1, raw_out)) {
                        fprintf(stderr, "ANE ffn_resadd batched eval failed at layer %d\n", L);
                        return nullptr;
                    }
                } else if (ane_layers_[L].fused_ffn) {
                    if (!ane_matvec_batch(ane_layers_[L].fused_ffn,
                                          mlp_batch.data(), x_norm_batch.data(),
                                          batch, hidden_size_, hidden_size_, packed0, raw_out)) {
                        fprintf(stderr, "ANE fused_ffn batched eval failed at layer %d\n", L);
                        return nullptr;
                    }
                    for (int b = 0; b < batch; b++) {
                        float* x = x_batch.data() + (size_t)b * hidden_size_;
                        float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                        for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
                    }
                } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
                    for (int b = 0; b < batch; b++) {
                        float* x = x_batch.data() + (size_t)b * hidden_size_;
                        float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
                        float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                        if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp, x_norm)) {
                            fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                            return nullptr;
                        }
                        for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
                    }
                } else {
                    fprintf(stderr, "No FFN kernel for layer %d\n", L);
                    return nullptr;
                }
                if (g_fwd_prof.enabled) g_fwd_prof.ffn_ms += t.elapsed_ms();
            }
        }

        last_hidden = x_batch.data() + (size_t)(batch - 1) * hidden_size_;
    }

    {
        Timer t;
        rmsnorm(x_, last_hidden, final_norm_, hidden_size_, rms_eps_);
        if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();
    }

    {
        Timer t;
        float* logits = compute_logits_from_hidden(x_);
        if (g_fwd_prof.enabled) {
            g_fwd_prof.lm_head_ms += t.elapsed_ms();
            g_fwd_prof.total_ms += t_total.elapsed_ms();
            g_fwd_prof.count += (int)token_ids.size();
        }
        return logits;
    }
}

bool Qwen3Model::init_speculative(const std::vector<int>& prompt_tokens, int start_pos) {
    static bool fusion_enabled = getenv("CROSS_LAYER_FUSION") != nullptr;
    static bool same_layer_fusion_enabled = getenv("FUSE_OPROJ_FFN") != nullptr;
    if (fusion_enabled || same_layer_fusion_enabled) {
        fprintf(stderr, "Speculative decode is not supported with fusion experiments enabled\n");
        return false;
    }
    if (prompt_tokens.empty()) return false;

    speculative_pending_ = false;
    speculative_pending_batch_ = 0;
    speculative_ready_ = false;
    draft_next_pos_ = start_pos;

    for (auto& kv : draft_kv_caches_) {
        kv.len = 0;
        kv.start = 0;
        memset(kv.k_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
        memset(kv.v_cache, 0, (size_t)kv.capacity * num_kv_heads_ * head_dim_ * sizeof(float));
    }

    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        if (!draft_forward(prompt_tokens[i], start_pos + i)) {
            return false;
        }
    }
    draft_next_pos_ = start_pos + (int)prompt_tokens.size();
    speculative_ready_ = true;
    return true;
}

bool Qwen3Model::init_external_draft_verify(const std::vector<int>& prompt_tokens, int start_pos) {
    static bool fusion_enabled = getenv("CROSS_LAYER_FUSION") != nullptr;
    static bool same_layer_fusion_enabled = getenv("FUSE_OPROJ_FFN") != nullptr;
    if (fusion_enabled || same_layer_fusion_enabled) {
        fprintf(stderr, "External draft verification is not supported with fusion experiments enabled\n");
        return false;
    }
    if (prompt_tokens.empty()) return false;
    (void)start_pos;
    speculative_pending_ = false;
    speculative_pending_batch_ = 0;
    return true;
}

bool Qwen3Model::draft_speculative_tokens(int max_tokens, int sampler_vocab,
                                          const SamplingParams& sampling,
                                          const std::vector<int>& recent_tokens,
                                          std::vector<int>& drafted_tokens,
                                          int seed_token) {
    drafted_tokens.clear();
    if (!speculative_ready_ || speculative_pending_) return false;

    int batch = std::min(max_tokens, speculative_batch_size_);
    if (batch <= 0) return false;

    begin_speculative_window(draft_kv_caches_, draft_spec_snapshots_, draft_spec_saved_valid_, draft_layers_);
    memcpy(draft_logits_history_, draft_logits_, (size_t)vocab_size_ * sizeof(float));
    std::vector<int> draft_context = recent_tokens;

    int produced = 0;
    if (seed_token >= 0 && batch > 0) {
        drafted_tokens.push_back(seed_token);
        draft_context.push_back(seed_token);
        if (!draft_forward(seed_token, draft_next_pos_, true, 0)) {
            return false;
        }
        memcpy(draft_logits_history_ + (size_t)vocab_size_, draft_logits_,
               (size_t)vocab_size_ * sizeof(float));
        produced = 1;
    }

    for (int i = produced; i < batch; i++) {
        const float* cur_logits = draft_logits_history_ + (size_t)i * vocab_size_;
        int token = sample_token(cur_logits, sampler_vocab, sampling, draft_context);
        drafted_tokens.push_back(token);
        draft_context.push_back(token);
        if (!draft_forward(token, draft_next_pos_ + i, true, i)) {
            return false;
        }
        memcpy(draft_logits_history_ + (size_t)(i + 1) * vocab_size_, draft_logits_,
               (size_t)vocab_size_ * sizeof(float));
    }

    return true;
}

const float* Qwen3Model::draft_logits_at(int position) const {
    if (!draft_logits_history_ || position < 0 || position > speculative_batch_size_) {
        return nullptr;
    }
    return draft_logits_history_ + (size_t)position * vocab_size_;
}

bool Qwen3Model::verify_speculative(const int* token_ids, int batch, int start_pos,
                                    float** logits_batch, int* logits_stride) {
    if (logits_batch) *logits_batch = nullptr;
    if (logits_stride) *logits_stride = 0;
    if (!token_ids || batch <= 0 || batch > speculative_batch_size_) return false;

    int qkv_total = q_proj_dim_ + 2 * kv_proj_dim_;
    std::vector<float> x_batch((size_t)batch * hidden_size_);
    std::vector<float> x_norm_batch((size_t)batch * hidden_size_);
    std::vector<float> qkv_batch((size_t)batch * qkv_total);
    std::vector<float> pre_oproj_batch((size_t)batch * full_out_dim_);
    std::vector<float> mlp_batch((size_t)batch * hidden_size_);
    std::vector<float> packed0;
    std::vector<float> packed1;
    std::vector<float> raw_out;

    for (int b = 0; b < batch; b++) {
        memcpy(x_batch.data() + (size_t)b * hidden_size_,
               embed_tokens_ + (int64_t)token_ids[b] * hidden_size_,
               (size_t)hidden_size_ * sizeof(float));
    }

    begin_speculative_window(kv_caches_, full_spec_snapshots_, full_spec_saved_valid_, num_layers_);

    for (int L = 0; L < num_layers_; L++) {
        for (int b = 0; b < batch; b++) {
            rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                    x_batch.data() + (size_t)b * hidden_size_,
                    layers_[L].input_layernorm, hidden_size_, rms_eps_);
        }

        if (!ane_matvec_batch(ane_layers_[L].first_proj,
                              qkv_batch.data(), x_norm_batch.data(),
                              batch, hidden_size_, qkv_total,
                              packed0, raw_out)) {
            fprintf(stderr, "ANE first_proj batched eval failed at layer %d\n", L);
            return false;
        }

        for (int b = 0; b < batch; b++) {
            float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
            float* k_raw = q_raw + q_proj_dim_;
            for (int h = 0; h < num_q_heads_; h++) {
                float* qh = q_raw + (size_t)h * head_dim_;
                rmsnorm(qh, qh, layers_[L].q_norm, head_dim_, rms_eps_);
            }
            for (int h = 0; h < num_kv_heads_; h++) {
                float* kh = k_raw + (size_t)h * head_dim_;
                rmsnorm(kh, kh, layers_[L].k_norm, head_dim_, rms_eps_);
            }
        }

        for (int b = 0; b < batch; b++) {
            int cur_pos = start_pos + b;
            float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
            float* k_raw = q_raw + q_proj_dim_;
            const float* rope_cos_row = nullptr;
            const float* rope_sin_row = nullptr;
            if (cur_pos >= 0 && cur_pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
                int half_rot = rot_dim_ / 2;
                rope_cos_row = rope_cos_ + (size_t)cur_pos * half_rot;
                rope_sin_row = rope_sin_ + (size_t)cur_pos * half_rot;
            }
            apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                             head_dim_, cur_pos, rope_theta_, rope_cos_row, rope_sin_row);
        }

        for (int b = 0; b < batch; b++) {
            float* q_raw = qkv_batch.data() + (size_t)b * qkv_total;
            float* k_raw = q_raw + q_proj_dim_;
            float* v_raw = q_raw + q_proj_dim_ + kv_proj_dim_;
            record_speculative_insert(kv_caches_, full_spec_snapshots_,
                                      full_spec_saved_k_, full_spec_saved_v_,
                                      full_spec_saved_slot_, full_spec_saved_valid_,
                                      num_layers_, L, b, k_raw, v_raw);
            auto& cache = kv_caches_[L];
            gqa_attention(pre_oproj_batch.data() + (size_t)b * full_out_dim_,
                          q_raw, cache.k_cache, cache.v_cache,
                          num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                          cache.start, cache.len, cache.capacity);
        }

        if (ane_layers_[L].oproj_add) {
            if (!ane_binary_batch(ane_layers_[L].oproj_add,
                                  full_out_dim_, hidden_size_, hidden_size_,
                                  x_batch.data(),
                                  pre_oproj_batch.data(), x_batch.data(), batch,
                                  packed0, packed1, raw_out)) {
                fprintf(stderr, "ANE oproj_add batched eval failed at layer %d\n", L);
                return false;
            }
        } else {
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* pre = pre_oproj_batch.data() + (size_t)b * full_out_dim_;
                if (!ane_matvec(ane_layers_[L].o_proj,
                                mlp_batch.data() + (size_t)b * hidden_size_,
                                pre, full_out_dim_, hidden_size_)) {
                    fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                    return false;
                }
                for (int i = 0; i < hidden_size_; i++) {
                    x[i] += mlp_batch[(size_t)b * hidden_size_ + i];
                }
            }
        }

        for (int b = 0; b < batch; b++) {
            rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                    x_batch.data() + (size_t)b * hidden_size_,
                    layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
        }

        if (ane_layers_[L].ffn_resadd) {
            if (!ane_binary_batch(ane_layers_[L].ffn_resadd,
                                  hidden_size_, hidden_size_, hidden_size_,
                                  x_batch.data(),
                                  x_norm_batch.data(), x_batch.data(), batch,
                                  packed0, packed1, raw_out)) {
                fprintf(stderr, "ANE ffn_resadd batched eval failed at layer %d\n", L);
                return false;
            }
        } else if (ane_layers_[L].fused_ffn) {
            if (!ane_matvec_batch(ane_layers_[L].fused_ffn,
                                  mlp_batch.data(), x_norm_batch.data(),
                                  batch, hidden_size_, hidden_size_, packed0, raw_out)) {
                fprintf(stderr, "ANE fused_ffn batched eval failed at layer %d\n", L);
                return false;
            }
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                for (int i = 0; i < hidden_size_; i++) {
                    x[i] += mlp[i];
                }
            }
        } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
                float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp, x_norm)) {
                    fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                    return false;
                }
                for (int i = 0; i < hidden_size_; i++) {
                    x[i] += mlp[i];
                }
            }
        } else {
            fprintf(stderr, "No FFN kernel for layer %d\n", L);
            return false;
        }
    }

    for (int b = 0; b < batch; b++) {
        rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                x_batch.data() + (size_t)b * hidden_size_,
                final_norm_, hidden_size_, rms_eps_);
    }
    compute_logits_batch_from_hidden(x_norm_batch.data(), batch, speculative_logits_batch_, packed0, raw_out);

    if (speculative_compare_enabled() && draft_layers_ == num_layers_) {
        for (int b = 0; b < batch; b++) {
            const float* draft_logits = draft_logits_history_ + (size_t)(b + 1) * vocab_size_;
            const float* verify_logits = speculative_logits_batch_ + (size_t)b * vocab_size_;
            float max_err = 0.0f;
            int max_err_idx = 0;
            for (int i = 0; i < vocab_size_; i++) {
                float err = fabsf(draft_logits[i] - verify_logits[i]);
                if (err > max_err) {
                    max_err = err;
                    max_err_idx = i;
                }
            }
            int draft_top = argmax_token(draft_logits, vocab_size_);
            int verify_top = argmax_token(verify_logits, vocab_size_);
            fprintf(stderr,
                    "spec compare step=%d pos=%d draft_top=%d verify_top=%d max_err=%.6f argmax_match=%s max_err_idx=%d\n",
                    b, start_pos + b, draft_top, verify_top, max_err,
                    (draft_top == verify_top ? "yes" : "no"), max_err_idx);
        }
    }

    speculative_pending_ = true;
    speculative_pending_batch_ = batch;
    if (logits_batch) *logits_batch = speculative_logits_batch_;
    if (logits_stride) *logits_stride = vocab_size_;
    return true;
}

void Qwen3Model::finalize_speculative(int accepted_tokens) {
    if (!speculative_pending_) return;
    if (accepted_tokens < 0) accepted_tokens = 0;
    if (accepted_tokens > speculative_pending_batch_) accepted_tokens = speculative_pending_batch_;

    finalize_speculative_window(kv_caches_, full_spec_snapshots_,
                                full_spec_saved_k_, full_spec_saved_v_,
                                full_spec_saved_slot_, full_spec_saved_valid_,
                                num_layers_, accepted_tokens, speculative_pending_batch_);
    finalize_speculative_window(draft_kv_caches_, draft_spec_snapshots_,
                                draft_spec_saved_k_, draft_spec_saved_v_,
                                draft_spec_saved_slot_, draft_spec_saved_valid_,
                                draft_layers_, accepted_tokens, speculative_pending_batch_);

    memcpy(draft_logits_,
           draft_logits_history_ + (size_t)accepted_tokens * vocab_size_,
           (size_t)vocab_size_ * sizeof(float));
    draft_next_pos_ += accepted_tokens;

    speculative_pending_ = false;
    speculative_pending_batch_ = 0;
    speculative_ready_ = true;
}

void Qwen3Model::finalize_external_draft_verify(int accepted_tokens) {
    if (!speculative_pending_) return;
    if (accepted_tokens < 0) accepted_tokens = 0;
    if (accepted_tokens > speculative_pending_batch_) accepted_tokens = speculative_pending_batch_;

    finalize_speculative_window(kv_caches_, full_spec_snapshots_,
                                full_spec_saved_k_, full_spec_saved_v_,
                                full_spec_saved_slot_, full_spec_saved_valid_,
                                num_layers_, accepted_tokens, speculative_pending_batch_);

    speculative_pending_ = false;
    speculative_pending_batch_ = 0;
}

bool Qwen3Model::accept_speculative_token(int token_id, int pos) {
    if (!draft_forward(token_id, pos)) return false;
    draft_next_pos_ = pos + 1;
    speculative_ready_ = true;
    return true;
}

float* Qwen3Model::forward(int token_id, int pos) {
    g_fwd_prof.check();
    Timer t_total;

    memcpy(x_, embed_tokens_ + (int64_t)token_id * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = scratch_attn_;
    int qkv_total = q_proj_dim_ + 2 * kv_proj_dim_;

    // --- Cross-layer fused forward pass ---
    // Layer 0: standalone QKV → attention → fused O_proj_0+QKV_1
    // Layers 1..N-2: attention (QKV pre-computed) → fused O_proj_L+QKV_{L+1}
    // Layer N-1: attention (QKV pre-computed) → standalone O_proj
    // All layers: post-attention FFN

    // Track whether scratch_qkv_ already has the QKV for the current layer
    bool qkv_precomputed = false;
    // Cross-layer fusion disabled by default: computing QKV_{L+1} before FFN_L
    // produces garbage because FFN is a massive nonlinear transformation.
    // Enable with CROSS_LAYER_FUSION=1 for speed testing (10.3 tok/s but broken output).
    static bool fusion_enabled = getenv("CROSS_LAYER_FUSION") != nullptr;
    // Disabled by default: faster on microbench/smoke tests, but generation turns to garbage
    // because ANE RMSNorm precision before FFN is still not good enough for full-model chaining.
    static bool same_layer_fusion_enabled = getenv("FUSE_OPROJ_FFN") != nullptr;

    for (int L = 0; L < num_layers_; L++) {

        // ---- Step 1: QKV projection ----
        if (!qkv_precomputed) {
            // Need to compute QKV for this layer (standalone path)
            Timer t;
            rmsnorm(x_norm_, x_, layers_[L].input_layernorm, hidden_size_, rms_eps_);
            if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();

            if (g_fwd_prof.enabled) t.reset();
            if (!ane_matvec(ane_layers_[L].first_proj, scratch_qkv_, x_norm_,
                            hidden_size_, qkv_total)) {
                fprintf(stderr, "ANE first_proj eval failed at layer %d\n", L);
                return nullptr;
            }
            if (g_fwd_prof.enabled) g_fwd_prof.qkv_ms += t.elapsed_ms();
        }
        // else: scratch_qkv_ already has QKV from previous layer's fused kernel

        // ---- Step 2: QK-norm + RoPE ----
        float* q_raw = scratch_qkv_;
        float* k_raw = scratch_qkv_ + q_proj_dim_;
        float* v_raw = scratch_qkv_ + q_proj_dim_ + kv_proj_dim_;

        if (g_fwd_prof.enabled) {
            Timer t;
            for (int h = 0; h < num_q_heads_; h++) {
                float* qh = q_raw + (size_t)h * head_dim_;
                rmsnorm(qh, qh, layers_[L].q_norm, head_dim_, rms_eps_);
            }
            for (int h = 0; h < num_kv_heads_; h++) {
                float* kh = k_raw + (size_t)h * head_dim_;
                rmsnorm(kh, kh, layers_[L].k_norm, head_dim_, rms_eps_);
            }
            g_fwd_prof.qk_norm_ms += t.elapsed_ms();
        } else {
            for (int h = 0; h < num_q_heads_; h++) {
                float* qh = q_raw + (size_t)h * head_dim_;
                rmsnorm(qh, qh, layers_[L].q_norm, head_dim_, rms_eps_);
            }
            for (int h = 0; h < num_kv_heads_; h++) {
                float* kh = k_raw + (size_t)h * head_dim_;
                rmsnorm(kh, kh, layers_[L].k_norm, head_dim_, rms_eps_);
            }
        }

        {
            Timer t;
            const float* rope_cos_row = nullptr;
            const float* rope_sin_row = nullptr;
            if (pos >= 0 && pos < rope_cache_len_ && rope_cos_ && rope_sin_) {
                int half_rot = rot_dim_ / 2;
                rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
                rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
            }
            apply_rope_qwen3(q_raw, k_raw, num_q_heads_, num_kv_heads_,
                             head_dim_, pos, rope_theta_, rope_cos_row, rope_sin_row);
            if (g_fwd_prof.enabled) g_fwd_prof.rope_ms += t.elapsed_ms();
        }

        // ---- Step 3: KV cache + attention ----
        {
            Timer t;
            auto& cache = kv_caches_[L];
            int slot;
            if (cache.len < cache.capacity) {
                slot = cache.start + cache.len;
                if (slot >= cache.capacity) slot -= cache.capacity;
                cache.len++;
            } else {
                slot = cache.start;
                cache.start++;
                if (cache.start >= cache.capacity) cache.start = 0;
            }
            size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
            memcpy(cache.k_cache + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
            memcpy(cache.v_cache + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
            gqa_attention(pre_oproj, q_raw, cache.k_cache, cache.v_cache,
                          num_q_heads_, num_kv_heads_, head_dim_, head_dim_,
                          cache.start, cache.len, cache.capacity);
            if (g_fwd_prof.enabled) g_fwd_prof.attention_ms += t.elapsed_ms();
        }

        bool same_layer_fused = false;

        // ---- Step 4: O_proj + residual (fused or standalone) ----
        {
            Timer t;
            if (same_layer_fusion_enabled && ane_layers_[L].fused_oproj_ffn) {
                if (!ane_eval_fused_oproj_ffn(ane_layers_[L].fused_oproj_ffn,
                        scratch_attn_, x_, pre_oproj, x_, full_out_dim_, hidden_size_)) {
                    fprintf(stderr, "ANE fused_oproj_ffn eval failed at layer %d\n", L);
                    return nullptr;
                }
                qkv_precomputed = false;
                same_layer_fused = true;
            } else if (ane_layers_[L].oproj_norm_qkv && fusion_enabled) {
                // Cross-layer fused: O_proj_L + add + RMSNorm_{L+1} + QKV_{L+1}
                // Output 1: scratch_qkv_ = QKV for next layer
                // Output 2: x_ = x + O_proj(attn_out) (updated residual)
                if (!ane_eval_oproj_norm_qkv(ane_layers_[L].oproj_norm_qkv,
                        scratch_qkv_, x_, pre_oproj, x_,
                        full_out_dim_, hidden_size_, qkv_total)) {
                    fprintf(stderr, "ANE oproj_norm_qkv eval failed at layer %d\n", L);
                    return nullptr;
                }
                qkv_precomputed = true;
            } else if (ane_layers_[L].oproj_add) {
                // Standalone: conv(O_proj, attn) + x_residual
                if (!ane_eval_oproj_add(ane_layers_[L].oproj_add,
                        x_, pre_oproj, x_, full_out_dim_, hidden_size_)) {
                    fprintf(stderr, "ANE oproj_add eval failed at layer %d\n", L);
                    return nullptr;
                }
                qkv_precomputed = false;
            } else {
                // Fallback: separate O_proj + CPU residual add
                float* attn_out = x_norm_;
                if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, full_out_dim_, hidden_size_)) {
                    fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                    return nullptr;
                }
                for (int i = 0; i < hidden_size_; i++) x_[i] += attn_out[i];
                qkv_precomputed = false;
            }
            if (g_fwd_prof.enabled) g_fwd_prof.o_proj_ms += t.elapsed_ms();
        }

        // ---- Step 5: Post-attention RMSNorm (CPU, for FFN precision) ----
        if (!same_layer_fused) {
            Timer t;
            rmsnorm(x_norm_, x_, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
            if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();
        }

        // ---- Step 6: FFN + residual add (ANE) ----
        {
            Timer t;
            if (same_layer_fused) {
                for (int i = 0; i < hidden_size_; i++) x_[i] += scratch_attn_[i];
            } else if (ane_layers_[L].ffn_resadd) {
                if (!ane_eval_fused_ffn_resadd(ane_layers_[L].ffn_resadd, x_, x_norm_, x_, hidden_size_)) {
                    fprintf(stderr, "ANE ffn_resadd eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else {
                float* mlp_out = scratch_attn_;
                if (ane_layers_[L].fused_ffn) {
                    if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, x_norm_, hidden_size_, hidden_size_)) {
                        fprintf(stderr, "ANE fused_ffn eval failed at layer %d\n", L);
                        return nullptr;
                    }
                } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
                    if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp_out, x_norm_)) {
                        fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                        return nullptr;
                    }
                } else {
                    fprintf(stderr, "No FFN kernel for layer %d\n", L);
                    return nullptr;
                }
                for (int i = 0; i < hidden_size_; i++) x_[i] += mlp_out[i];
            }
            if (g_fwd_prof.enabled) g_fwd_prof.ffn_ms += t.elapsed_ms();
        }
    }

    // Final norm
    {
        Timer t;
        rmsnorm(x_, x_, final_norm_, hidden_size_, rms_eps_);
        if (g_fwd_prof.enabled) g_fwd_prof.rmsnorm_ms += t.elapsed_ms();
    }

    // LM head
    {
        Timer t;
        compute_logits_from_hidden(x_);
        if (g_fwd_prof.enabled) g_fwd_prof.lm_head_ms += t.elapsed_ms();
    }

    if (g_fwd_prof.enabled) {
        g_fwd_prof.total_ms += t_total.elapsed_ms();
        g_fwd_prof.count++;
    }

    return logits_;
}

} // namespace ane_lm
