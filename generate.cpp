#include "generate.h"
#include "core/sampling.h"
#include <ane_lm/common.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

namespace ane_lm {

static bool speculative_decode_enabled() {
    return getenv("ANE_SPECULATIVE_DECODE") != nullptr;
}

static bool speculative_stats_enabled() {
    return getenv("ANE_SPECULATIVE_STATS") != nullptr;
}

static bool is_stop_token(int token, const Tokenizer& tokenizer) {
    return token == tokenizer.eos_id() || token == tokenizer.im_end_id();
}

static bool is_utf8_continuation(uint8_t b) {
    return (b & 0xC0u) == 0x80u;
}

static size_t longest_common_prefix_len(const std::string& a, const std::string& b) {
    size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

// Move cut position to a UTF-8 codepoint boundary at or before cut.
static size_t utf8_boundary_at_or_before(const std::string& s, size_t cut) {
    if (cut >= s.size()) return s.size();
    while (cut > 0 && is_utf8_continuation(static_cast<uint8_t>(s[cut]))) {
        cut--;
    }
    return cut;
}

void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& messages,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback,
    DraftModelContext* draft)
{
    std::string formatted;
    if (tokenizer.has_chat_template()) {
        formatted = tokenizer.apply_chat_template(messages, true, enable_thinking);
    } else {
        for (auto& [role, content] : messages) {
            (void)role;
            formatted += content + "\n";
        }
    }
    std::vector<int> prompt_tokens = tokenizer.encode(formatted);

    bool use_external_speculative =
        draft && draft->model && draft->tokenizer &&
        model.supports_external_draft_decode() &&
        draft->model->supports_speculative_decode();

    if (use_external_speculative) {
        std::vector<int> draft_prompt_tokens = draft->tokenizer->encode(formatted);
        if (draft_prompt_tokens != prompt_tokens) {
            fprintf(stderr, "Draft model tokenizer mismatch — disabling external speculation\n");
            use_external_speculative = false;
        } else if (draft->tokenizer->eos_id() != tokenizer.eos_id() ||
                   draft->tokenizer->im_end_id() != tokenizer.im_end_id()) {
            fprintf(stderr, "Draft model stop-token mismatch — disabling external speculation\n");
            use_external_speculative = false;
        }
    }

    Timer prefill_timer;
    float* logits = model.prefill(prompt_tokens, 0);
    if (!logits) {
        fprintf(stderr, "Forward failed during prefill\n");
        return;
    }
    double prefill_ms = prefill_timer.elapsed_ms();
    double prompt_tps = prompt_tokens.size() / (prefill_ms / 1000.0);

    bool use_internal_speculative = !use_external_speculative &&
        speculative_decode_enabled() && model.supports_speculative_decode();
    if (use_internal_speculative && !model.init_speculative(prompt_tokens, 0)) {
        use_internal_speculative = false;
    }
    if (use_external_speculative) {
        if (!draft->model->init_speculative(prompt_tokens, 0) ||
            !model.init_external_draft_verify(prompt_tokens, 0)) {
            fprintf(stderr, "Failed to initialize external speculative draft path\n");
            use_external_speculative = false;
        }
    }

    int sampler_vocab = std::min(model.vocab_size(), tokenizer.vocab_size());
    if (use_external_speculative) {
        sampler_vocab = std::min(sampler_vocab, draft->model->vocab_size());
        sampler_vocab = std::min(sampler_vocab, draft->tokenizer->vocab_size());
    }
    if (sampler_vocab <= 0) {
        fprintf(stderr, "Invalid sampler vocab size: %d\n", sampler_vocab);
        return;
    }

    Timer gen_timer;
    int n_generated = 0;
    std::vector<int> generated_tokens;
    std::string emitted_text;
    std::string prev_decoded;
    bool has_prev_decoded = false;

    auto emit_token = [&](int token) {
        n_generated++;
        generated_tokens.push_back(token);
        std::string current_decoded = tokenizer.decode(generated_tokens);

        std::string piece;
        if (has_prev_decoded) {
            size_t lcp = longest_common_prefix_len(prev_decoded, current_decoded);
            size_t stable_len = utf8_boundary_at_or_before(prev_decoded, lcp);
            std::string stable_decoded = prev_decoded.substr(0, stable_len);
            if (stable_decoded.size() >= emitted_text.size() &&
                stable_decoded.compare(0, emitted_text.size(), emitted_text) == 0) {
                piece = stable_decoded.substr(emitted_text.size());
                emitted_text = std::move(stable_decoded);
            } else {
                size_t p = longest_common_prefix_len(stable_decoded, emitted_text);
                p = utf8_boundary_at_or_before(stable_decoded, p);
                piece = stable_decoded.substr(p);
                emitted_text = std::move(stable_decoded);
            }
        }
        prev_decoded = std::move(current_decoded);
        has_prev_decoded = true;

        if (callback) {
            GenerationResponse r;
            r.text = piece;
            r.token = token;
            r.prompt_tokens = (int)prompt_tokens.size();
            r.prompt_tps = prompt_tps;
            r.generation_tokens = n_generated;
            r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
            callback(r);
        }
    };

    auto sample_from_probs = [](const float* probs, int vocab_size) {
        float r = (float)drand48();
        float cum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cum += probs[i];
            if (cum >= r) return i;
        }
        return vocab_size - 1;
    };

    std::vector<float> target_probs((size_t)sampler_vocab);
    std::vector<float> draft_probs((size_t)sampler_vocab);
    std::vector<float> diff_probs((size_t)sampler_vocab);

    auto speculative_accept_or_sample = [&](const float* target_logits,
                                            const float* draft_logits,
                                            const std::vector<int>& context,
                                            int drafted_token,
                                            int* correction_token) {
        compute_sampling_probs(target_probs.data(), target_logits, sampler_vocab, sampling, context);
        compute_sampling_probs(draft_probs.data(), draft_logits, sampler_vocab, sampling, context);

        float accept_prob = 0.0f;
        if (drafted_token >= 0 && drafted_token < sampler_vocab) {
            float p = target_probs[(size_t)drafted_token];
            float q = draft_probs[(size_t)drafted_token];
            if (q > 0.0f) {
                accept_prob = std::min(1.0f, p / q);
            }
        }

        if ((float)drand48() < accept_prob) {
            *correction_token = drafted_token;
            return true;
        }

        float total = 0.0f;
        for (int i = 0; i < sampler_vocab; i++) {
            float diff = target_probs[(size_t)i] - draft_probs[(size_t)i];
            if (diff < 0.0f) diff = 0.0f;
            diff_probs[(size_t)i] = diff;
            total += diff;
        }

        if (total <= 1e-8f) {
            *correction_token = sample_from_probs(target_probs.data(), sampler_vocab);
            return false;
        }

        float inv_total = 1.0f / total;
        for (int i = 0; i < sampler_vocab; i++) {
            diff_probs[(size_t)i] *= inv_total;
        }
        *correction_token = sample_from_probs(diff_probs.data(), sampler_vocab);
        return false;
    };

    bool show_speculative_stats = speculative_stats_enabled();
    int speculative_attempts = 0;
    int speculative_accepts = 0;

    int limit = (max_tokens > 0) ? max_tokens : INT_MAX;
    while (n_generated < limit) {
        if (use_external_speculative) {
            int draft_batch = std::min(limit - n_generated,
                                       std::min(model.speculative_batch_size(),
                                                draft->model->speculative_batch_size()));
            if (draft_batch > 0) {
                std::vector<int> drafted_tokens;
                if (!draft->model->draft_speculative_tokens(draft_batch, sampler_vocab, sampling,
                                                            generated_tokens, drafted_tokens)) {
                    fprintf(stderr, "External speculative drafting failed at token %d\n", n_generated);
                    return;
                }
                if (!drafted_tokens.empty()) {
                    float* logits_batch = nullptr;
                    int logits_stride = 0;
                    if (!model.verify_speculative(drafted_tokens.data(), (int)drafted_tokens.size(),
                                                  (int)prompt_tokens.size() + n_generated,
                                                  &logits_batch, &logits_stride)) {
                        fprintf(stderr, "External speculative verification failed at token %d\n", n_generated);
                        return;
                    }

                    speculative_attempts += (int)drafted_tokens.size();
                    int accepted = 0;
                    int correction_token = -1;
                    bool correction_emitted = false;
                    bool stop_requested = false;
                    float* current_logits = logits;
                    std::vector<int> proposal_context = generated_tokens;

                    for (int i = 0; i < (int)drafted_tokens.size() && n_generated < limit; i++) {
                        const float* draft_logits = draft->model->draft_logits_at(i);
                        if (!draft_logits) {
                            break;
                        }

                        int sampled_token = -1;
                        bool accepted_token = speculative_accept_or_sample(
                            current_logits, draft_logits, proposal_context, drafted_tokens[i], &sampled_token);

                        if (accepted_token) {
                            accepted++;
                            speculative_accepts++;
                            if (is_stop_token(drafted_tokens[i], tokenizer)) {
                                stop_requested = true;
                                break;
                            }
                            emit_token(drafted_tokens[i]);
                            proposal_context.push_back(drafted_tokens[i]);
                            current_logits = logits_batch + (size_t)i * logits_stride;
                            continue;
                        }

                        correction_token = sampled_token;
                        if (is_stop_token(correction_token, tokenizer)) {
                            stop_requested = true;
                        } else {
                            emit_token(correction_token);
                            correction_emitted = true;
                        }
                        break;
                    }

                    model.finalize_external_draft_verify(accepted);
                    draft->model->finalize_speculative(accepted);

                    if (stop_requested) {
                        break;
                    }
                    if (correction_emitted) {
                        if (n_generated >= limit) {
                            break;
                        }
                        int pos = (int)prompt_tokens.size() + n_generated - 1;
                        logits = model.forward(correction_token, pos);
                        if (!logits) {
                            fprintf(stderr, "Forward failed during external speculative correction at token %d\n", n_generated);
                            return;
                        }
                        if (!draft->model->accept_speculative_token(correction_token, pos)) {
                            fprintf(stderr, "External draft-state correction advance failed at token %d\n", n_generated);
                            return;
                        }
                        continue;
                    }
                    if (accepted > 0) {
                        logits = current_logits;
                        continue;
                    }
                }
            }
        }

        if (use_internal_speculative) {
            int draft_batch = std::min(limit - n_generated, model.speculative_batch_size());
            if (draft_batch > 0) {
                int seed_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);
                if (is_stop_token(seed_token, tokenizer)) {
                    break;
                }

                std::vector<int> drafted_tokens;
                if (!model.draft_speculative_tokens(draft_batch, sampler_vocab, sampling,
                                                   generated_tokens, drafted_tokens, seed_token)) {
                    fprintf(stderr, "Speculative drafting failed at token %d\n", n_generated);
                    return;
                }
                if (!drafted_tokens.empty()) {
                    float* logits_batch = nullptr;
                    int logits_stride = 0;
                    if (!model.verify_speculative(drafted_tokens.data(), (int)drafted_tokens.size(),
                                                  (int)prompt_tokens.size() + n_generated,
                                                  &logits_batch, &logits_stride)) {
                        fprintf(stderr, "Speculative verification failed at token %d\n", n_generated);
                        return;
                    }

                    speculative_attempts += (int)drafted_tokens.size();
                    int accepted = 0;
                    int correction_token = -1;
                    bool correction_emitted = false;
                    bool stop_requested = false;
                    float* current_logits = logits;
                    std::vector<int> proposal_context = generated_tokens;

                    accepted++;
                    speculative_accepts++;
                    emit_token(seed_token);
                    proposal_context.push_back(seed_token);
                    current_logits = logits_batch;

                    for (int i = 1; i < (int)drafted_tokens.size() && n_generated < limit; i++) {
                        const float* draft_logits = model.draft_logits_at(i);
                        if (!draft_logits) {
                            break;
                        }

                        int sampled_token = -1;
                        bool accepted_token = speculative_accept_or_sample(
                            current_logits, draft_logits, proposal_context, drafted_tokens[i], &sampled_token);

                        if (accepted_token) {
                            accepted++;
                            speculative_accepts++;
                            if (is_stop_token(drafted_tokens[i], tokenizer)) {
                                stop_requested = true;
                                break;
                            }
                            emit_token(drafted_tokens[i]);
                            proposal_context.push_back(drafted_tokens[i]);
                            current_logits = logits_batch + (size_t)i * logits_stride;
                            continue;
                        }

                        correction_token = sampled_token;
                        if (is_stop_token(correction_token, tokenizer)) {
                            stop_requested = true;
                        } else {
                            emit_token(correction_token);
                            correction_emitted = true;
                        }
                        break;
                    }

                    model.finalize_speculative(accepted);

                    if (stop_requested) {
                        break;
                    }
                    if (correction_emitted) {
                        if (n_generated >= limit) {
                            break;
                        }
                        int pos = (int)prompt_tokens.size() + n_generated - 1;
                        logits = model.forward(correction_token, pos);
                        if (!logits) {
                            fprintf(stderr, "Forward failed during speculative correction at token %d\n", n_generated);
                            return;
                        }
                        if (!model.accept_speculative_token(correction_token, pos)) {
                            fprintf(stderr, "Speculative draft-state correction advance failed at token %d\n", n_generated);
                            return;
                        }
                        continue;
                    }
                    if (accepted > 0) {
                        logits = current_logits;
                        continue;
                    }
                }
            }
        }

        int next_token = sample_token(logits, sampler_vocab, sampling, generated_tokens);
        if (is_stop_token(next_token, tokenizer)) {
            break;
        }

        emit_token(next_token);
        if (n_generated >= limit) {
            break;
        }

        int pos = (int)prompt_tokens.size() + n_generated - 1;
        logits = model.forward(next_token, pos);
        if (!logits) {
            fprintf(stderr, "Forward failed during generation at token %d\n", n_generated);
            return;
        }
        if (use_external_speculative) {
            if (!draft->model->accept_speculative_token(next_token, pos)) {
                fprintf(stderr, "External draft-state advance failed at token %d\n", n_generated);
                return;
            }
        } else if (use_internal_speculative && !model.accept_speculative_token(next_token, pos)) {
            fprintf(stderr, "Speculative draft-state advance failed at token %d\n", n_generated);
            return;
        }
    }

    if (callback && has_prev_decoded) {
        std::string final_decoded = prev_decoded;
        std::string tail;
        if (final_decoded.size() >= emitted_text.size() &&
            final_decoded.compare(0, emitted_text.size(), emitted_text) == 0) {
            tail = final_decoded.substr(emitted_text.size());
        } else {
            size_t p = 0;
            while (p < final_decoded.size() && p < emitted_text.size() &&
                   final_decoded[p] == emitted_text[p]) p++;
            tail = final_decoded.substr(p);
        }

        if (!tail.empty()) {
            GenerationResponse r;
            r.text = tail;
            r.token = generated_tokens.back();
            r.prompt_tokens = (int)prompt_tokens.size();
            r.prompt_tps = prompt_tps;
            r.generation_tokens = n_generated;
            r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
            callback(r);
        }
    }

    if (show_speculative_stats && (use_internal_speculative || use_external_speculative)) {
        double accept_pct = speculative_attempts > 0
            ? (100.0 * speculative_accepts / speculative_attempts)
            : 0.0;
        fprintf(stderr, "\nSpeculative decode: %d/%d accepted (%.1f%%)\n",
                speculative_accepts, speculative_attempts, accept_pct);
    }

    if (callback) {
        GenerationResponse r;
        r.token = -1;
        r.prompt_tokens = (int)prompt_tokens.size();
        r.prompt_tps = prompt_tps;
        r.generation_tokens = n_generated;
        r.generation_tps = n_generated / (gen_timer.elapsed_ms() / 1000.0);
        callback(r);
    }
}

// Single-prompt overload wraps into messages vector
void stream_generate(
    LLMModel& model,
    Tokenizer& tokenizer,
    const std::string& prompt,
    int max_tokens,
    bool enable_thinking,
    const SamplingParams& sampling,
    std::function<void(const GenerationResponse&)> callback,
    DraftModelContext* draft)
{
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", prompt}
    };
    stream_generate(model, tokenizer, messages, max_tokens, enable_thinking, sampling, std::move(callback), draft);
}

} // namespace ane_lm
