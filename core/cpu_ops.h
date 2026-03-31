#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>
#include <Accelerate/Accelerate.h>

namespace ane_lm {

// ============ Activations ============

inline float silu_f(float x) { return x / (1.0f + expf(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float softplus_f(float x) { return logf(1.0f + expf(x)); }

void silu_vec_inplace(float* x, int n, float* tmp);
void mul_sigmoid_inplace(float* y, const float* z, int n, float* tmp);

// ============ RMSNorm ============

void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps = 1e-6f);
void rmsnorm_gated(float* out, const float* x, const float* z, const float* weight, int dim);

// ============ RoPE ============

void apply_rope_cached(float* q, float* k, int n_q_heads, int n_kv_heads,
                       int head_dim, int q_head_stride, int k_head_stride,
                       int rot_dim, int pos, float theta,
                       const float* cos_row, const float* sin_row);

// ============ Softmax ============

void softmax(float* x, int n);

// ============ Matrix-Vector Multiply ============

void matvec(float* y, const float* W, const float* x, int out_dim, int in_dim);

// ============ L2 Normalization ============

void l2_normalize(float* x, int dim);

// ============ Conv1d ============

void conv1d_update(float* y, float* conv_state, int* state_pos, const float* x,
                   const float* w, int channels, int kernel_size);

// Batch conv1d for prefill: process N tokens at once, output to y_batch[N*channels]
// x_batch[N*channels] is the input (e.g. from Proj), stride between tokens = x_stride
// Updates conv_state and state_pos for subsequent decode tokens.
void conv1d_batch(float* y_batch, float* conv_state, int* state_pos,
                  const float* x_batch, int x_stride,
                  const float* w, int channels, int kernel_size, int N);

// ============ SSM Recurrence ============

void ssm_step(float* y, float* state, const float* q, const float* k,
              const float* v, float decay, float beta, int key_dim, int value_dim);

// ============ GQA Attention ============

void gqa_attention(float* out, const float* q,
                   const float* k_cache, const float* v_cache,
                   int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                   int cache_start, int cache_len, int cache_capacity);

} // namespace ane_lm
