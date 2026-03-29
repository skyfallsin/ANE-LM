#include "sampling.h"
#include "cpu_ops.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace ane_lm {

void compute_sampling_probs(float* probs_out, const float* logits, int vocab_size,
                            const SamplingParams& params,
                            const std::vector<int>& recent_tokens) {
    memcpy(probs_out, logits, vocab_size * sizeof(float));

    if (!recent_tokens.empty()) {
        int start = std::max(0, (int)recent_tokens.size() - params.repetition_context_size);

        std::unordered_map<int, int> freq;
        for (int j = start; j < (int)recent_tokens.size(); j++) {
            int tok = recent_tokens[j];
            if (tok >= 0 && tok < vocab_size) {
                freq[tok]++;
            }
        }

        for (auto& [tok, count] : freq) {
            if (params.repetition_penalty > 1.0f) {
                if (probs_out[tok] > 0.0f) {
                    probs_out[tok] /= params.repetition_penalty;
                } else {
                    probs_out[tok] *= params.repetition_penalty;
                }
            }
            if (params.frequency_penalty > 0.0f) {
                probs_out[tok] -= params.frequency_penalty * count;
            }
        }
    }

    if (params.temperature <= 0.0f) {
        int max_i = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (probs_out[i] > probs_out[max_i]) max_i = i;
        }
        memset(probs_out, 0, vocab_size * sizeof(float));
        probs_out[max_i] = 1.0f;
        return;
    }

    float inv_t = 1.0f / params.temperature;
    for (int i = 0; i < vocab_size; i++) probs_out[i] *= inv_t;
    softmax(probs_out, vocab_size);
}

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens) {
    float* adjusted = (float*)malloc(vocab_size * sizeof(float));
    compute_sampling_probs(adjusted, logits, vocab_size, params, recent_tokens);

    float r = (float)drand48();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += adjusted[i];
        if (cum >= r) { free(adjusted); return i; }
    }
    free(adjusted);
    return vocab_size - 1;
}

} // namespace ane_lm
