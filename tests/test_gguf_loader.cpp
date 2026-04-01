/**
 * test_gguf_loader.cpp — Validate GGUF model loading for ane.cpp
 *
 * Tests:
 *   1. Load full Qwen3.5-4B model from Q8_0 GGUF
 *   2. Verify all tensor shapes match expected dimensions
 *   3. Verify CPU dequantized weights are non-zero and reasonable
 *   4. Run a single ggml_mul_mat with a loaded weight tensor
 *
 * Usage:
 *   cmake --build build --target test_gguf_loader
 *   ./build/test_gguf_loader [path/to/model.gguf]
 */

#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-alloc.h"

#include <Accelerate/Accelerate.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>

using namespace ane_lm;

// Qwen3.5-4B model config
static const int HIDDEN       = 2560;
static const int INTERMEDIATE = 9216;
static const int NUM_LAYERS   = 32;
static const int VOCAB        = 248320;
static const int HEAD_DIM     = 256;
static const int NUM_Q_HEADS  = 32;
static const int NUM_KV_HEADS = 4;
static const int LIN_QKV_DIM  = 8192;   // key_dim*num_key_heads + key_dim*num_key_heads + val_dim*num_val_heads
static const int LIN_TOTAL_VAL = 4096;
static const int LIN_NUM_VAL  = 32;
static const int FULL_Q_DIM   = 8192;   // num_q_heads * head_dim (32*256)... wait that's too big
// Actually: full_q_dim = num_q_heads * head_dim * 2 (gated) = 32 * 256 * 2 = 16384? No.
// Let me compute from the GGUF: blk.3.attn_q.weight has n_elts=655360 = 2560*256 → 2560 rows * 256 cols
// Wait no — ne=[2560, 8192] so it's [K=2560, M=8192]. So q_proj is 8192 output dim.

static const int FULL_KV_DIM  = 1024;   // num_kv_heads * head_dim = 4*256
static const int FULL_OUT_DIM = 4096;   // attn_output = [2560, 4096] → 4096 input dim

// Layer types for Qwen3.5-4B: pattern is 3 linear, 1 full, repeated
static std::vector<int> get_layer_types() {
    // From config.json layer_types array
    std::vector<int> types = {
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1,
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
    };
    return types;
}

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Test 1: Load model
// ---------------------------------------------------------------------------

static GGUFModel* g_model = nullptr;

static bool test_load(const char* path) {
    printf("\n=== Test 1: Load GGUF model ===\n");

    auto types = get_layer_types();
    g_model = load_gguf_model(
        path, types,
        HIDDEN, INTERMEDIATE,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        LIN_QKV_DIM, LIN_TOTAL_VAL, LIN_NUM_VAL,
        FULL_Q_DIM, FULL_KV_DIM, FULL_OUT_DIM,
        VOCAB, true  // tie_word_embeddings
    );

    if (!g_model) {
        printf("FAIL: load_gguf_model returned nullptr\n");
        return false;
    }

    printf("  Loaded %d layers, Metal buffer %.1f MB\n",
           g_model->num_layers,
           ggml_backend_buffer_get_size(g_model->weight_buffer) / (1024.0*1024.0));
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Verify tensor shapes
// ---------------------------------------------------------------------------

static bool test_shapes() {
    printf("\n=== Test 2: Verify tensor shapes ===\n");
    if (!g_model) { printf("SKIP: no model\n"); return true; }

    int errors = 0;
    auto check = [&](const char* desc, ggml_tensor* t, int64_t ne0, int64_t ne1) {
        if (!t) {
            printf("  FAIL: %s is NULL\n", desc);
            errors++;
            return;
        }
        if (t->ne[0] != ne0 || t->ne[1] != ne1) {
            printf("  FAIL: %s shape [%lld,%lld] expected [%lld,%lld]\n",
                   desc, t->ne[0], t->ne[1], ne0, ne1);
            errors++;
        }
    };

    check("token_embd", g_model->token_embd, HIDDEN, VOCAB);
    check("output_norm", g_model->output_norm, HIDDEN, 1);

    // Check layer 0 (linear attention)
    auto& l0 = g_model->layers[0];
    check("L0.first_proj (attn_qkv)", l0.first_proj, HIDDEN, LIN_QKV_DIM);
    check("L0.z_proj (attn_gate)", l0.z_proj, HIDDEN, LIN_TOTAL_VAL);
    check("L0.a_proj (ssm_alpha)", l0.a_proj, HIDDEN, LIN_NUM_VAL);
    check("L0.b_proj (ssm_beta)", l0.b_proj, HIDDEN, LIN_NUM_VAL);
    check("L0.gate_proj", l0.gate_proj, HIDDEN, INTERMEDIATE);
    check("L0.up_proj", l0.up_proj, HIDDEN, INTERMEDIATE);
    check("L0.down_proj", l0.down_proj, INTERMEDIATE, HIDDEN);
    check("L0.input_norm", l0.input_norm, HIDDEN, 1);
    check("L0.post_norm", l0.post_norm, HIDDEN, 1);

    // Check layer 3 (full attention)
    auto& l3 = g_model->layers[3];
    // first_proj = concat(q[8192], k[1024], v[1024]) = [2560, 10240]
    int expected_qkv = FULL_Q_DIM + 2 * FULL_KV_DIM;
    check("L3.first_proj (Q+K+V)", l3.first_proj, HIDDEN, expected_qkv);
    check("L3.gate_proj", l3.gate_proj, HIDDEN, INTERMEDIATE);
    check("L3.down_proj", l3.down_proj, INTERMEDIATE, HIDDEN);

    printf("  L0 first_proj_rows=%d, o_proj_in=%d\n", l0.first_proj_rows, l0.o_proj_in);
    printf("  L3 first_proj_rows=%d, o_proj_in=%d\n", l3.first_proj_rows, l3.o_proj_in);

    if (errors > 0) {
        printf("  FAIL: %d shape mismatches\n", errors);
        return false;
    }
    printf("  All shapes verified\n");
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: Verify CPU dequantized weights
// ---------------------------------------------------------------------------

static bool test_cpu_weights() {
    printf("\n=== Test 3: Verify CPU dequantized weights ===\n");
    if (!g_model) { printf("SKIP: no model\n"); return true; }

    int errors = 0;
    auto check_f32 = [&](const char* desc, float* ptr, int64_t n) {
        if (!ptr) {
            printf("  FAIL: %s is NULL\n", desc);
            errors++;
            return;
        }
        // Check not all zeros
        float sum = 0;
        for (int64_t i = 0; i < std::min(n, (int64_t)1000); i++) sum += fabsf(ptr[i]);
        if (sum < 1e-10f) {
            printf("  FAIL: %s appears to be all zeros\n", desc);
            errors++;
        }
    };

    check_f32("embed_tokens", g_model->embed_tokens_f32, (int64_t)VOCAB * HIDDEN);
    check_f32("final_norm", g_model->final_norm_f32, HIDDEN);

    // Layer 0 (linear)
    auto& cl0 = g_model->cpu_layers[0];
    check_f32("L0.input_layernorm", cl0.input_layernorm, HIDDEN);
    check_f32("L0.post_attention_layernorm", cl0.post_attention_layernorm, HIDDEN);
    check_f32("L0.conv1d_w", cl0.conv1d_w, LIN_QKV_DIM * 4);
    check_f32("L0.A (exp'd)", cl0.A, LIN_NUM_VAL);
    check_f32("L0.dt_bias", cl0.dt_bias, LIN_NUM_VAL);
    check_f32("L0.ssm_norm_w", cl0.ssm_norm_w, 128);

    // Layer 3 (full)
    auto& cl3 = g_model->cpu_layers[3];
    check_f32("L3.q_norm", cl3.q_norm, HEAD_DIM);
    check_f32("L3.k_norm", cl3.k_norm, HEAD_DIM);

    // Verify exp(A) was applied
    if (cl0.A) {
        bool all_positive = true;
        for (int i = 0; i < LIN_NUM_VAL; i++) {
            if (cl0.A[i] <= 0) { all_positive = false; break; }
        }
        if (!all_positive) {
            printf("  FAIL: L0.A should be all positive after exp()\n");
            errors++;
        } else {
            printf("  L0.A[0..3] = %.4f %.4f %.4f %.4f (exp'd, all positive ✓)\n",
                   cl0.A[0], cl0.A[1], cl0.A[2], cl0.A[3]);
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d weight check errors\n", errors);
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: Run ggml_mul_mat with loaded weight
// ---------------------------------------------------------------------------

static bool test_mul_mat() {
    printf("\n=== Test 4: ggml_mul_mat with loaded ffn_down weight ===\n");
    if (!g_model) { printf("SKIP: no model\n"); return true; }

    auto& lw = g_model->layers[0];
    ggml_tensor* w = lw.down_proj;
    if (!w) { printf("FAIL: down_proj is NULL\n"); return false; }

    const int K = (int)w->ne[0]; // I=9216
    const int M = (int)w->ne[1]; // H=2560
    const int N = 64;

    printf("  Weight: down_proj [%d, %d] (%s)\n", K, M, ggml_type_name(w->type));

    // Build a graph using the already-loaded weight tensor
    size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    ggml_init_params gp = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context* ctx = ggml_init(gp);

    // Activation tensor (new, needs allocation)
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(a, "act");

    // Build graph: out = w @ a (using the weight tensor from the model)
    // But w is already in the model's Metal buffer. We need to reference it
    // in our graph. ggml_mul_mat just creates the op node — the tensors'
    // buffer pointers are already set.
    ggml_tensor* out = ggml_mul_mat(ctx, w, a);
    ggml_set_name(out, "out");

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Allocate only the NEW tensors (activation + output) on Metal
    // The weight tensor w already has its buffer from model loading
    ggml_backend_buffer_t act_buf = ggml_backend_alloc_ctx_tensors(ctx, g_model->metal_backend);
    if (!act_buf) {
        printf("FAIL: could not allocate activation buffer\n");
        ggml_free(ctx);
        return false;
    }

    // Fill activation with random data
    std::vector<float> a_data(N * K);
    srand(42);
    for (auto& v : a_data) v = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));

    // Compute
    double t0 = now_ms();
    ggml_status status = ggml_backend_graph_compute(g_model->metal_backend, graph);
    double t1 = now_ms();

    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute returned %d\n", (int)status);
        ggml_backend_buffer_free(act_buf);
        ggml_free(ctx);
        return false;
    }

    // Benchmark
    double t2 = now_ms();
    for (int i = 0; i < 10; i++) {
        ggml_backend_graph_compute(g_model->metal_backend, graph);
    }
    double t3 = now_ms();

    printf("  Compute: %.2f ms (first), %.2f ms (avg of 10)\n", t1-t0, (t3-t2)/10.0);

    double gflops = (2.0 * N * M * K) / ((t3-t2)/10.0 * 1e6);
    printf("  Throughput: %.1f GFLOPS\n", gflops);

    // Read back result and sanity check (not all zeros)
    std::vector<float> out_data(N * M);
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size() * sizeof(float));

    float sum = 0;
    for (int i = 0; i < N * M; i++) sum += fabsf(out_data[i]);
    printf("  Output mean abs: %.6f (should be > 0)\n", sum / (N * M));

    ggml_backend_buffer_free(act_buf);
    ggml_free(ctx);

    if (sum < 1e-10f) {
        printf("  FAIL: output is all zeros\n");
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    printf("GGUF Loader Test\n");
    printf("================\n");

    std::string path = "hf-models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf";
    if (argc > 1) path = argv[1];

    int pass = 0, fail = 0;

    if (test_load(path.c_str())) pass++; else fail++;
    if (test_shapes()) pass++; else fail++;
    if (test_cpu_weights()) pass++; else fail++;
    if (test_mul_mat()) pass++; else fail++;

    delete g_model;

    printf("\n================\n");
    printf("Results: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
