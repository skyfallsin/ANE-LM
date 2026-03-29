#pragma once

#include "qwen3_5.h"
#include "../../core/ane_runtime.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace ane_lm {

struct Qwen3Args {
    int hidden_size = 1024;
    int num_hidden_layers = 28;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int head_dim = 128;
    int intermediate_size = 3072;
    int vocab_size = 151936;
    int max_position_embeddings = 40960;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;

    static Qwen3Args from_json(const nlohmann::json& config);
};

class Qwen3Model : public LLMModel {
public:
    ~Qwen3Model() override;
    bool load(const std::string& model_dir) override;
    float* forward(int token_id, int pos) override;
    float* prefill(const std::vector<int>& token_ids, int start_pos = 0) override;
    bool supports_speculative_decode() const override { return true; }
    bool init_speculative(const std::vector<int>& prompt_tokens, int start_pos = 0) override;
    int speculative_batch_size() const override { return speculative_batch_size_; }
    bool draft_speculative_tokens(int max_tokens, int sampler_vocab,
                                  const SamplingParams& sampling,
                                  const std::vector<int>& recent_tokens,
                                  std::vector<int>& drafted_tokens,
                                  int seed_token = -1) override;
    const float* draft_logits_at(int position) const override;
    bool verify_speculative(const int* token_ids, int batch, int start_pos,
                            float** logits_batch, int* logits_stride) override;
    void finalize_speculative(int accepted_tokens) override;
    bool accept_speculative_token(int token_id, int pos) override;
    bool supports_external_draft_decode() const override { return true; }
    bool init_external_draft_verify(const std::vector<int>& prompt_tokens, int start_pos = 0) override;
    void finalize_external_draft_verify(int accepted_tokens) override;
    void reset() override;
    int vocab_size() const override { return vocab_size_; }

private:
    // Config
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int vocab_size_ = 0;
    int num_layers_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rot_dim_ = 0;
    int max_pos_ = 0;
    float rope_theta_ = 0;
    float rms_eps_ = 0;
    bool tie_word_embeddings_ = true;

    int q_proj_dim_ = 0;
    int kv_proj_dim_ = 0;
    int full_out_dim_ = 0;

    int rope_cache_len_ = 0;

    static constexpr int KV_CACHE_CAPACITY = 2048;
    static constexpr int LM_HEAD_ANE_CHUNK_MAX = 32768;
    static constexpr int MAX_SPECULATIVE_BATCH = ANE_SPATIAL;

    struct LayerWeights {
        float* input_layernorm = nullptr;
        float* post_attention_layernorm = nullptr;
        float* q_norm = nullptr;
        float* k_norm = nullptr;
    };

    struct KVCache {
        float* k_cache = nullptr;
        float* v_cache = nullptr;
        int len = 0;
        int start = 0;
        int capacity = 0;
    };

    struct KVCacheSnapshot {
        int len = 0;
        int start = 0;
    };

    std::vector<LayerWeights> layers_;
    std::vector<KVCache> kv_caches_;
    std::vector<LayerANEKernels> ane_layers_;

    float* embed_tokens_ = nullptr;
    float* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    std::vector<ANEKernel*> lm_head_kernels_;
    int lm_head_chunk_ = LM_HEAD_ANE_CHUNK_MAX;
    bool ane_lm_head_enabled_ = false;

    float* x_ = nullptr;
    float* x_norm_ = nullptr;
    float* logits_ = nullptr;
    float* speculative_logits_batch_ = nullptr;
    float* draft_x_ = nullptr;
    float* draft_x_norm_ = nullptr;
    float* draft_logits_ = nullptr;
    float* draft_logits_history_ = nullptr;
    float* scratch_qkv_ = nullptr;
    float* draft_scratch_qkv_ = nullptr;
    float* scratch_attn_ = nullptr;
    float* draft_scratch_attn_ = nullptr;
    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;

    int speculative_batch_size_ = MAX_SPECULATIVE_BATCH;
    int draft_layers_ = 2;
    int draft_next_pos_ = 0;
    int speculative_pending_batch_ = 0;
    bool speculative_ready_ = false;
    bool speculative_pending_ = false;

    std::vector<KVCache> draft_kv_caches_;

    std::vector<KVCacheSnapshot> full_spec_snapshots_;
    std::vector<float> full_spec_saved_k_;
    std::vector<float> full_spec_saved_v_;
    std::vector<int> full_spec_saved_slot_;
    std::vector<uint8_t> full_spec_saved_valid_;

    std::vector<KVCacheSnapshot> draft_spec_snapshots_;
    std::vector<float> draft_spec_saved_k_;
    std::vector<float> draft_spec_saved_v_;
    std::vector<int> draft_spec_saved_slot_;
    std::vector<uint8_t> draft_spec_saved_valid_;

    void apply_args(const Qwen3Args& args);
    bool load_weights(ModelWeights* sf);
    bool compile_ane(ModelWeights* sf, const std::string& blob_dir);
    bool compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir);
    void free_lm_head_ane();
    void compute_logits_from_hidden_into(const float* hidden, float* out);
    void compute_logits_batch_from_hidden(const float* hidden_batch,
                                          int batch,
                                          float* logits_batch,
                                          std::vector<float>& packed_in,
                                          std::vector<float>& raw_out);
    float* compute_logits_from_hidden(const float* hidden);
    float* draft_forward(int token_id, int pos,
                         bool speculative_insert = false,
                         int token_index = -1);
    int argmax_token(const float* logits, int vocab_size) const;
    void begin_speculative_window(std::vector<KVCache>& caches,
                                  std::vector<KVCacheSnapshot>& snapshots,
                                  std::vector<uint8_t>& saved_valid,
                                  int num_cache_layers);
    void record_speculative_insert(std::vector<KVCache>& caches,
                                   std::vector<KVCacheSnapshot>& snapshots,
                                   std::vector<float>& saved_k,
                                   std::vector<float>& saved_v,
                                   std::vector<int>& saved_slot,
                                   std::vector<uint8_t>& saved_valid,
                                   int num_cache_layers,
                                   int layer_index,
                                   int token_index,
                                   const float* k_raw,
                                   const float* v_raw);
    void finalize_speculative_window(std::vector<KVCache>& caches,
                                     std::vector<KVCacheSnapshot>& snapshots,
                                     std::vector<float>& saved_k,
                                     std::vector<float>& saved_v,
                                     std::vector<int>& saved_slot,
                                     std::vector<uint8_t>& saved_valid,
                                     int num_cache_layers,
                                     int accepted_tokens,
                                     int pending_batch);

    bool forward_full_attn_core(int L, float* x, float* pre_oproj, int pos);
};

} // namespace ane_lm
