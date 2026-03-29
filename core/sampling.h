#pragma once

#include <vector>

namespace ane_lm {

struct SamplingParams {
    float temperature = 0.6f;
    float repetition_penalty = 1.2f;
    int repetition_context_size = 256;
    float frequency_penalty = 0.1f;
};

void compute_sampling_probs(float* probs_out, const float* logits, int vocab_size,
                            const SamplingParams& params,
                            const std::vector<int>& recent_tokens = {});

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens = {});

} // namespace ane_lm
