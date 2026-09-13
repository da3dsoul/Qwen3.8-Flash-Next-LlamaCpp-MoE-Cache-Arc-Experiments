// Standalone check for the two qwen4exp hyper-connection SYCL fusions
// (ggml_sycl_hc_stream_mean_fused / ggml_sycl_hc_combine_fused).
//
// It rebuilds exactly the op sequence llama_model_qwen4exp::graph::build_hc_mix and
// build_hc_combine emit, runs it on SYCL and on CPU, and reports the difference.
// The fusions only fire when the graph matches, so a mismatch here means either the
// matcher accepted a shape it should not have, or a kernel computes the wrong thing.
//
// Not part of llama.cpp's own test suite: it needs the SYCL backend and the qwen4exp
// graph shape, neither of which test-backend-ops covers. Build with:
//   icpx -fsycl -std=c++17 -I ggml/include -o hc_fusion_test docker/hc_fusion_test.cpp \
//        -lggml -lggml-base
//
// Usage: hc_fusion_test [n_embd] [hc] [n_tokens]
// GGML_SYCL_ENABLE_FUSION=0 runs the same graph unfused, for an A/B of the two paths.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

struct hc_inputs {
    int64_t n_embd;
    int64_t hc;
    int64_t nt;
    std::vector<float> xn;        // [hc*n_embd, nt]
    std::vector<float> up;        // [hc*n_embd, nt]
    std::vector<float> residual;  // [n_embd, hc, nt]
    std::vector<float> block_out; // [n_embd, nt]
    std::vector<float> inject;    // [hc, nt]
};

// mirrors build_hc_mix's tail and build_hc_combine, op for op
static void build_graph(ggml_context * ctx, ggml_cgraph * gf, const hc_inputs & in,
                        ggml_tensor ** t_xn, ggml_tensor ** t_up, ggml_tensor ** t_res,
                        ggml_tensor ** t_blk, ggml_tensor ** t_inj,
                        ggml_tensor ** out_mixed, ggml_tensor ** out_comb) {
    const int64_t n_embd = in.n_embd;
    const int64_t hc     = in.hc;
    const int64_t nt     = in.nt;
    const int64_t hc_dim = hc * n_embd;

    *t_xn  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    *t_up  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    *t_res = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    *t_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, nt);
    *t_inj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, nt);
    ggml_set_name(*t_xn,  "xn");
    ggml_set_name(*t_up,  "up");
    ggml_set_name(*t_res, "residual");
    ggml_set_name(*t_blk, "block_out");
    ggml_set_name(*t_inj, "inject");
    ggml_set_input(*t_xn);
    ggml_set_input(*t_up);
    ggml_set_input(*t_res);
    ggml_set_input(*t_blk);
    ggml_set_input(*t_inj);

    // build_hc_mix tail
    ggml_tensor * gate  = ggml_sigmoid(ctx, *t_up);
    ggml_tensor * gated = ggml_mul(ctx, *t_xn, gate);
    gated = ggml_reshape_3d(ctx, gated, n_embd, hc, nt);

    ggml_tensor * mixed = ggml_view_2d(ctx, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    mixed = ggml_cont(ctx, mixed);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx, mixed, s);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float) hc);
    ggml_set_name(mixed, "hc_mixed");
    ggml_set_output(mixed);

    // build_hc_combine
    ggml_tensor * w = ggml_scale(ctx, *t_inj, 1.0f / (float) hc);
    w = ggml_sigmoid(ctx, w);
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx, *t_blk, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx, b, n_embd, hc, nt, 1);

    ggml_tensor * comb = ggml_add(ctx, *t_res, ggml_mul(ctx, b, w));
    ggml_set_name(comb, "hc_combine");
    ggml_set_output(comb);

    ggml_build_forward_expand(gf, mixed);
    ggml_build_forward_expand(gf, comb);

    *out_mixed = mixed;
    *out_comb  = comb;
}

static bool run_on(ggml_backend_t backend, const hc_inputs & in,
                   std::vector<float> & mixed_out, std::vector<float> & comb_out) {
    const size_t mem = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor *t_xn, *t_up, *t_res, *t_blk, *t_inj, *mixed, *comb;
    build_graph(ctx, gf, in, &t_xn, &t_up, &t_res, &t_blk, &t_inj, &mixed, &comb);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "alloc failed\n");
        return false;
    }

    ggml_backend_tensor_set(t_xn,  in.xn.data(),        0, in.xn.size()        * sizeof(float));
    ggml_backend_tensor_set(t_up,  in.up.data(),        0, in.up.size()        * sizeof(float));
    ggml_backend_tensor_set(t_res, in.residual.data(),  0, in.residual.size()  * sizeof(float));
    ggml_backend_tensor_set(t_blk, in.block_out.data(), 0, in.block_out.size() * sizeof(float));
    ggml_backend_tensor_set(t_inj, in.inject.data(),    0, in.inject.size()    * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return false;
    }

    mixed_out.resize(ggml_nelements(mixed));
    comb_out.resize(ggml_nelements(comb));
    ggml_backend_tensor_get(mixed, mixed_out.data(), 0, mixed_out.size() * sizeof(float));
    ggml_backend_tensor_get(comb,  comb_out.data(),  0, comb_out.size()  * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

static int compare(const char * what, const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        printf("%-12s SIZE MISMATCH %zu vs %zu\n", what, a.size(), b.size());
        return 1;
    }
    // near-zero elements make an elementwise relative error meaningless, so score the
    // difference against the magnitude of the tensor as a whole (NMSE), the way
    // test-backend-ops does
    double max_abs  = 0.0;
    double sum_d2   = 0.0;
    double sum_a2   = 0.0;
    size_t n_exact  = 0;
    uint64_t hash_a = 1469598103934665603ull;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        max_abs = std::fmax(max_abs, std::fabs(d));
        sum_d2 += d * d;
        sum_a2 += (double) b[i] * (double) b[i];
        n_exact += (a[i] == b[i]);
        uint32_t bits;
        memcpy(&bits, &a[i], sizeof(bits));
        hash_a = (hash_a ^ bits) * 1099511628211ull;
    }
    const double nmse = sum_a2 > 0.0 ? sum_d2 / sum_a2 : sum_d2;
    printf("%-12s n=%-8zu bit-identical=%zu/%zu  max_abs=%.3e  nmse=%.3e  hash=%016llx\n",
           what, a.size(), n_exact, a.size(), max_abs, nmse, (unsigned long long) hash_a);
    return nmse > 1e-10 ? 1 : 0;
}

int main(int argc, char ** argv) {
    hc_inputs in;
    in.n_embd = argc > 1 ? atoll(argv[1]) : 2560;
    in.hc     = argc > 2 ? atoll(argv[2]) : 4;
    in.nt     = argc > 3 ? atoll(argv[3]) : 1;

    printf("hc_fusion_test: n_embd=%lld hc=%lld n_tokens=%lld  GGML_SYCL_ENABLE_FUSION=%s\n",
           (long long) in.n_embd, (long long) in.hc, (long long) in.nt,
           getenv("GGML_SYCL_ENABLE_FUSION") ? getenv("GGML_SYCL_ENABLE_FUSION") : "(default 1)");

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    auto fill = [&](std::vector<float> & v, size_t n) {
        v.resize(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = d(rng);
        }
    };
    fill(in.xn,        in.hc * in.n_embd * in.nt);
    fill(in.up,        in.hc * in.n_embd * in.nt);
    fill(in.residual,  in.n_embd * in.hc * in.nt);
    fill(in.block_out, in.n_embd * in.nt);
    fill(in.inject,    in.hc * in.nt);

    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    std::vector<float> cpu_mixed, cpu_comb;
    if (!run_on(cpu, in, cpu_mixed, cpu_comb)) {
        return 1;
    }

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d_i = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(d_i), "SYCL0") == 0) {
            dev = d_i;
            break;
        }
    }
    if (!dev) {
        fprintf(stderr, "no SYCL0 device\n");
        return 1;
    }
    ggml_backend_t sycl = ggml_backend_dev_init(dev, nullptr);
    std::vector<float> sycl_mixed, sycl_comb;
    if (!run_on(sycl, in, sycl_mixed, sycl_comb)) {
        return 1;
    }

    int fail = 0;
    fail |= compare("hc_mixed",   sycl_mixed, cpu_mixed);
    fail |= compare("hc_combine", sycl_comb,  cpu_comb);

    ggml_backend_free(sycl);
    ggml_backend_free(cpu);

    printf(fail ? "FAIL\n" : "OK\n");
    return fail;
}
