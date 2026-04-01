/**
 * test_ggml_metal.cpp — Proof-of-concept: ggml-metal integration for ane.cpp
 *
 * Validates:
 *   1. ggml builds and links correctly as a CMake subdirectory
 *   2. Metal backend initializes on Apple Silicon
 *   3. GGUF file loads and tensor data is accessible
 *   4. ggml_mul_mat on Metal produces correct results (vs cblas reference)
 *
 * Usage:
 *   cmake --build build --target test_ggml_metal
 *   ./build/test_ggml_metal [path/to/model.gguf]
 *
 * Default GGUF: hf-models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "gguf.h"

#include <Accelerate/Accelerate.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

/// Dequantize a ggml tensor to fp32 on the CPU.
/// Handles F16, F32, Q8_0, Q4_K_M, etc. via ggml_internal_get_type_traits.
static std::vector<float> dequantize_tensor_to_f32(const ggml_tensor* t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);

    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        // Use ggml's built-in dequantization
        const auto* traits = ggml_get_type_traits(t->type);
        if (!traits->to_float) {
            fprintf(stderr, "No dequantization for type %s\n", ggml_type_name(t->type));
            std::fill(out.begin(), out.end(), 0.0f);
            return out;
        }
        traits->to_float(t->data, out.data(), n);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Test 1: Metal backend init
// ---------------------------------------------------------------------------

static bool test_metal_init() {
    printf("\n=== Test 1: Metal backend initialization ===\n");

    ggml_backend_t metal = ggml_backend_metal_init();
    if (!metal) {
        printf("FAIL: ggml_backend_metal_init() returned NULL\n");
        return false;
    }

    printf("  Backend name: %s\n", ggml_backend_name(metal));
    printf("  Apple GPU family 7+: %s\n",
           ggml_backend_metal_supports_family(metal, 7) ? "yes" : "no");

    ggml_backend_free(metal);
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 2: Load GGUF and inspect tensors
// ---------------------------------------------------------------------------

static bool test_gguf_load(const char* gguf_path) {
    printf("\n=== Test 2: GGUF tensor loading ===\n");
    printf("  File: %s\n", gguf_path);

    // Load GGUF with tensor metadata only (no_alloc = true → don't load data yet)
    struct ggml_context* meta_ctx = nullptr;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &meta_ctx,
    };

    struct gguf_context* gguf = gguf_init_from_file(gguf_path, params);
    if (!gguf) {
        printf("FAIL: gguf_init_from_file returned NULL\n");
        return false;
    }

    int64_t n_tensors = gguf_get_n_tensors(gguf);
    printf("  Tensors: %lld\n", n_tensors);
    printf("  Data offset: %zu bytes\n", gguf_get_data_offset(gguf));

    // Find and print first few tensor names + shapes
    for (int64_t i = 0; i < std::min(n_tensors, (int64_t)8); i++) {
        const char* name = gguf_get_tensor_name(gguf, i);
        enum ggml_type type = gguf_get_tensor_type(gguf, i);
        size_t size = gguf_get_tensor_size(gguf, i);

        // Look up the tensor in the ggml context to get shape
        struct ggml_tensor* t = ggml_get_tensor(meta_ctx, name);
        if (t) {
            printf("  [%lld] %-40s  type=%-6s  shape=[%lld, %lld]  size=%zuB\n",
                   i, name, ggml_type_name(type),
                   t->ne[0], t->ne[1], size);
        }
    }

    // Verify our target tensor exists
    const char* target = "blk.0.ffn_down.weight";
    int64_t idx = gguf_find_tensor(gguf, target);
    if (idx < 0) {
        printf("FAIL: tensor '%s' not found\n", target);
        gguf_free(gguf);
        ggml_free(meta_ctx);
        return false;
    }
    printf("  Target tensor '%s' found at index %lld\n", target, idx);

    gguf_free(gguf);
    ggml_free(meta_ctx);
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 3: ggml_mul_mat on Metal — random data, verify vs cblas
// ---------------------------------------------------------------------------

static bool test_mul_mat_random(ggml_backend_t metal) {
    printf("\n=== Test 3: ggml_mul_mat on Metal (random F32, verify vs cblas) ===\n");

    // Dimensions matching Qwen3.5-4B ffn_down: C = act[N,K] @ W[M,K]^T = [N,M]
    // Where K=intermediate_size=9216, M=hidden_size=2560, N=batch
    const int M = 2560;   // weight rows (output dim)
    const int K = 9216;   // weight cols (input dim)
    const int N = 64;     // batch size (tokens)

    printf("  Dimensions: act[%d,%d] @ weight[%d,%d]^T → out[%d,%d]\n", N, K, M, K, N, M);

    // --- Build ggml graph ---
    size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    struct ggml_init_params ctx_params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(ctx_params);

    // Weight: [K, M] in ggml notation (ne0=K, ne1=M)
    struct ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(w, "weight");

    // Activation: [K, N] in ggml notation (ne0=K, ne1=N)
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(a, "act");

    // Result: [M, N]
    struct ggml_tensor* out = ggml_mul_mat(ctx, w, a);
    ggml_set_name(out, "out");

    // Build forward graph
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // Allocate tensors on Metal backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, metal);
    if (!buf) {
        printf("FAIL: ggml_backend_alloc_ctx_tensors returned NULL\n");
        ggml_free(ctx);
        return false;
    }
    printf("  Metal buffer allocated: %.2f MB\n",
           ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));

    // --- Fill with random data ---
    std::vector<float> w_data(M * K);
    std::vector<float> a_data(N * K);

    srand(42);
    for (auto& v : w_data) v = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    for (auto& v : a_data) v = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;

    ggml_backend_tensor_set(w, w_data.data(), 0, w_data.size() * sizeof(float));
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));

    // --- Compute on Metal ---
    double t0 = now_ms();
    enum ggml_status status = ggml_backend_graph_compute(metal, graph);
    double t1 = now_ms();

    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: ggml_backend_graph_compute returned %d\n", (int)status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    printf("  Metal compute: %.2f ms\n", t1 - t0);

    // Read back result
    std::vector<float> out_metal(N * M);
    ggml_backend_tensor_get(out, out_metal.data(), 0, out_metal.size() * sizeof(float));

    // --- Compute reference with cblas ---
    // C = A @ W^T  where A=[N,K] row-major, W=[M,K] row-major
    // cblas_sgemm: C[N,M] = A[N,K] * W^T[K,M]
    std::vector<float> out_ref(N * M, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, M, K,
                1.0f,
                a_data.data(), K,    // A [N, K]
                w_data.data(), K,    // B [M, K] → B^T
                0.0f,
                out_ref.data(), M);  // C [N, M]

    // --- Compare ---
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < N * M; i++) {
        float err = fabsf(out_metal[i] - out_ref[i]);
        max_abs_err = fmaxf(max_abs_err, err);
        if (fabsf(out_ref[i]) > 1e-6f) {
            float rel = err / fabsf(out_ref[i]);
            max_rel_err = fmaxf(max_rel_err, rel);
        }
        sum_sq_err += (double)(out_metal[i] - out_ref[i]) * (out_metal[i] - out_ref[i]);
        sum_sq_ref += (double)out_ref[i] * out_ref[i];
    }
    double rmse = sqrt(sum_sq_err / (N * M));
    double nrmse = sqrt(sum_sq_err / sum_sq_ref);

    printf("  Results (%d elements):\n", N * M);
    printf("    max_abs_err = %.6e\n", max_abs_err);
    printf("    max_rel_err = %.6e\n", max_rel_err);
    printf("    RMSE        = %.6e\n", rmse);
    printf("    NRMSE       = %.6e\n", nrmse);
    printf("    out_metal[0..3] = %.6f %.6f %.6f %.6f\n",
           out_metal[0], out_metal[1], out_metal[2], out_metal[3]);
    printf("    out_ref[0..3]   = %.6f %.6f %.6f %.6f\n",
           out_ref[0], out_ref[1], out_ref[2], out_ref[3]);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    // ggml-metal uses simdgroup matmul which accumulates in fp32 but inputs
    // may be fp16-rounded internally. NRMSE ~3e-4 is expected.
    if (nrmse > 1e-3) {
        printf("  FAIL: NRMSE %.6e exceeds threshold 1e-3\n", nrmse);
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 4: ggml_mul_mat on Metal with Q8_0 weights from GGUF
// ---------------------------------------------------------------------------

static bool test_mul_mat_gguf_q8(const char* gguf_path, ggml_backend_t metal) {
    printf("\n=== Test 4: ggml_mul_mat on Metal with Q8_0 GGUF weight ===\n");
    printf("  File: %s\n", gguf_path);

    // Load GGUF with data into a CPU ggml_context
    struct ggml_context* data_ctx = nullptr;
    struct gguf_init_params params = {
        .no_alloc = false,   // allocate and load tensor data
        .ctx = &data_ctx,
    };

    struct gguf_context* gguf = gguf_init_from_file(gguf_path, params);
    if (!gguf || !data_ctx) {
        printf("FAIL: gguf_init_from_file returned NULL\n");
        return false;
    }

    // Find our target weight tensor
    const char* tensor_name = "blk.0.ffn_down.weight";
    struct ggml_tensor* w_gguf = ggml_get_tensor(data_ctx, tensor_name);
    if (!w_gguf) {
        printf("FAIL: tensor '%s' not found in context\n", tensor_name);
        gguf_free(gguf);
        ggml_free(data_ctx);
        return false;
    }

    printf("  Weight tensor: %s\n", tensor_name);
    printf("    type  = %s\n", ggml_type_name(w_gguf->type));
    printf("    shape = [%lld, %lld]  (ne0=K=%lld, ne1=M=%lld)\n",
           w_gguf->ne[0], w_gguf->ne[1], w_gguf->ne[0], w_gguf->ne[1]);
    printf("    size  = %.2f MB\n", ggml_nbytes(w_gguf) / (1024.0 * 1024.0));

    const int K = (int)w_gguf->ne[0];  // 2560 (hidden_size)
    const int M = (int)w_gguf->ne[1];  // 9216 (intermediate_size)
    // Wait — ffn_down goes from I → H, so shape is [H, I]=[2560, 9216]
    // Actually in GGUF, ne0 is the first dim. Let me just use what the file says.
    const int N = 64;  // batch size

    printf("  Activation: [%d, %d] (N=%d, K=%d)\n", N, K, N, K);

    // --- Dequantize weight to F32 for cblas reference ---
    printf("  Dequantizing Q8_0 weight to F32 for reference...\n");
    std::vector<float> w_f32 = dequantize_tensor_to_f32(w_gguf);
    printf("    Dequantized %zu elements\n", w_f32.size());

    // --- Build ggml graph with Q8_0 weight on Metal ---
    size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    struct ggml_init_params gctx_params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context* gctx = ggml_init(gctx_params);

    // Weight: Q8_0 [K, M]
    struct ggml_tensor* w = ggml_new_tensor_2d(gctx, w_gguf->type, K, M);
    ggml_set_name(w, "weight_q8");

    // Activation: F32 [K, N]
    struct ggml_tensor* a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, N);
    ggml_set_name(a, "act");

    // Result: [M, N]
    struct ggml_tensor* out = ggml_mul_mat(gctx, w, a);
    ggml_set_name(out, "out");

    struct ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, out);

    // Allocate on Metal
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, metal);
    if (!buf) {
        printf("FAIL: buffer alloc failed\n");
        ggml_free(gctx);
        gguf_free(gguf);
        ggml_free(data_ctx);
        return false;
    }

    // Copy Q8_0 weight data from GGUF to Metal buffer
    ggml_backend_tensor_set(w, w_gguf->data, 0, ggml_nbytes(w_gguf));

    // Create random activation
    std::vector<float> a_data(N * K);
    srand(123);
    for (auto& v : a_data) v = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));

    // --- Compute on Metal ---
    double t0 = now_ms();
    enum ggml_status status = ggml_backend_graph_compute(metal, graph);
    double t1 = now_ms();

    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: graph compute returned %d\n", (int)status);
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        gguf_free(gguf);
        ggml_free(data_ctx);
        return false;
    }

    // Warmup done, now benchmark
    double t2 = now_ms();
    for (int i = 0; i < 10; i++) {
        ggml_backend_graph_compute(metal, graph);
    }
    double t3 = now_ms();
    printf("  Metal compute: %.2f ms (first), %.2f ms (avg of 10)\n",
           t1 - t0, (t3 - t2) / 10.0);

    // GFLOPS estimate
    double gflops = (2.0 * N * M * K) / ((t3 - t2) / 10.0 * 1e6);
    printf("  Throughput: %.1f GFLOPS\n", gflops);

    // Read back
    std::vector<float> out_metal(N * M);
    ggml_backend_tensor_get(out, out_metal.data(), 0, out_metal.size() * sizeof(float));

    // --- cblas reference using dequantized weights ---
    std::vector<float> out_ref(N * M, 0.0f);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, M, K,
                1.0f,
                a_data.data(), K,
                w_f32.data(), K,
                0.0f,
                out_ref.data(), M);

    // --- Compare (Q8_0 will have quantization error) ---
    float max_abs_err = 0.0f;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < N * M; i++) {
        float err = fabsf(out_metal[i] - out_ref[i]);
        max_abs_err = fmaxf(max_abs_err, err);
        sum_sq_err += (double)(out_metal[i] - out_ref[i]) * (out_metal[i] - out_ref[i]);
        sum_sq_ref += (double)out_ref[i] * out_ref[i];
    }
    double nrmse = sqrt(sum_sq_err / sum_sq_ref);

    printf("  Results (%d elements):\n", N * M);
    printf("    max_abs_err = %.6e\n", max_abs_err);
    printf("    NRMSE       = %.6e  (Q8_0 quantization error expected)\n", nrmse);
    printf("    out_metal[0..3] = %.6f %.6f %.6f %.6f\n",
           out_metal[0], out_metal[1], out_metal[2], out_metal[3]);
    printf("    out_ref[0..3]   = %.6f %.6f %.6f %.6f\n",
           out_ref[0], out_ref[1], out_ref[2], out_ref[3]);

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    gguf_free(gguf);
    ggml_free(data_ctx);

    // Q8_0 should be very close — typical NRMSE < 1e-3
    if (nrmse > 5e-3) {
        printf("  FAIL: NRMSE %.6e exceeds threshold 5e-3\n", nrmse);
        return false;
    }
    printf("  PASS\n");
    return true;
}

// ---------------------------------------------------------------------------
// Test 5: Benchmark — ggml-metal Q8_0 mul_mat at various batch sizes
// ---------------------------------------------------------------------------

static bool test_benchmark(const char* gguf_path, ggml_backend_t metal) {
    printf("\n=== Test 5: ggml-metal Q8_0 mul_mat benchmark ===\n");

    // Load GGUF
    struct ggml_context* data_ctx = nullptr;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx = &data_ctx,
    };
    struct gguf_context* gguf = gguf_init_from_file(gguf_path, params);
    if (!gguf || !data_ctx) {
        printf("SKIP: could not load GGUF\n");
        return true;
    }

    struct ggml_tensor* w_gguf = ggml_get_tensor(data_ctx, "blk.0.ffn_down.weight");
    if (!w_gguf) {
        printf("SKIP: tensor not found\n");
        gguf_free(gguf);
        ggml_free(data_ctx);
        return true;
    }

    const int K = (int)w_gguf->ne[0];
    const int M = (int)w_gguf->ne[1];

    int batch_sizes[] = {1, 16, 64, 128, 256, 512, 1024};
    int n_batch = sizeof(batch_sizes) / sizeof(batch_sizes[0]);

    printf("  Weight: %s [%d, %d] (%s)\n",
           "blk.0.ffn_down.weight", K, M, ggml_type_name(w_gguf->type));
    printf("  %-8s  %8s  %10s\n", "batch", "ms/call", "GFLOPS");
    printf("  %-8s  %8s  %10s\n", "-----", "-------", "------");

    for (int bi = 0; bi < n_batch; bi++) {
        const int N = batch_sizes[bi];

        size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
        struct ggml_init_params gp = {
            .mem_size   = ctx_size,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context* gctx = ggml_init(gp);

        struct ggml_tensor* w = ggml_new_tensor_2d(gctx, w_gguf->type, K, M);
        struct ggml_tensor* a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor* out = ggml_mul_mat(gctx, w, a);

        struct ggml_cgraph* graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(graph, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, metal);
        if (!buf) {
            ggml_free(gctx);
            continue;
        }

        // Copy weight data
        ggml_backend_tensor_set(w, w_gguf->data, 0, ggml_nbytes(w_gguf));

        // Random activations
        std::vector<float> a_data(N * K);
        for (auto& v : a_data) v = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size() * sizeof(float));

        // Warmup
        ggml_backend_graph_compute(metal, graph);

        // Benchmark
        const int iters = (N <= 16) ? 100 : 20;
        double t0 = now_ms();
        for (int i = 0; i < iters; i++) {
            ggml_backend_graph_compute(metal, graph);
        }
        double t1 = now_ms();
        double ms = (t1 - t0) / iters;
        double gflops = (2.0 * N * M * K) / (ms * 1e6);

        printf("  %-8d  %8.3f  %10.1f\n", N, ms, gflops);

        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }

    gguf_free(gguf);
    ggml_free(data_ctx);
    printf("  DONE\n");
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    printf("ggml-metal integration proof-of-concept\n");
    printf("========================================\n");

    // Default GGUF path
    std::string gguf_path = "hf-models/Qwen3.5-4B/Qwen3.5-4B-Q8_0.gguf";
    if (argc > 1) {
        gguf_path = argv[1];
    }

    int pass = 0, fail = 0;

    // Test 1: Metal init
    if (test_metal_init()) pass++; else fail++;

    // Test 2: GGUF loading
    if (test_gguf_load(gguf_path.c_str())) pass++; else fail++;

    // Init Metal backend for remaining tests
    ggml_backend_t metal = ggml_backend_metal_init();
    if (!metal) {
        printf("\nFATAL: Cannot initialize Metal backend for tests 3-5\n");
        printf("\nResults: %d passed, %d failed\n", pass, fail + 3);
        return 1;
    }

    // Test 3: Random F32 mul_mat
    if (test_mul_mat_random(metal)) pass++; else fail++;

    // Test 4: Q8_0 mul_mat from GGUF
    if (test_mul_mat_gguf_q8(gguf_path.c_str(), metal)) pass++; else fail++;

    // Test 5: Benchmark
    if (test_benchmark(gguf_path.c_str(), metal)) pass++; else fail++;

    ggml_backend_free(metal);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
