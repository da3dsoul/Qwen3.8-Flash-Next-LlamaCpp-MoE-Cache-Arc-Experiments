// In-graph equivalence harness for QSA sparse flash attention.
//
// With LLAMA_QSA_SPARSE_CHECK=1 the model builds two attention outputs per QSA layer from the same
// q/k/v and the same top-k mask in the same graph: kqv_out (sparse hint set, so the SYCL backend
// restricts the call to the finite-mask columns) and kqv_out_ref (hint 0, so it stays dense).
// Nothing about this stack is reproducible across processes, so the two must be compared inside one
// run. Restriction is exact up to floating point reassociation, so this reports NMSE and max abs
// diff rather than demanding bit-identity.
//
// build (inside the build container):
//   icpx -fsycl -O2 -I/app/include -I/app/ggml/include -I/app/common \
//        staging/work/qsa_sparse_equiv.cpp -L/app/build/bin -lllama -lggml -lggml-base -o qsa_sparse_equiv

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct run_ctx {
    int    step     = 0;
    double nmse_max = 0.0;
    double absd_max = 0.0;
    long   n_cmp    = 0;
    long   n_bad    = 0;
};

static std::map<std::string, std::vector<float>> g_sparse;
static std::map<std::string, std::vector<float>> g_dense;

static bool want_tensor(const char * name) {
    return strncmp(name, "kqv_out-", 8) == 0 || strncmp(name, "kqv_out_ref-", 12) == 0;
}

static void compare(run_ctx * rc, const std::string & il) {
    auto a = g_sparse.find(il);
    auto b = g_dense.find(il);

    if (a == g_sparse.end() || b == g_dense.end()) {
        return;
    }

    const std::vector<float> & s = a->second;
    const std::vector<float> & d = b->second;

    if (s.size() != d.size()) {
        printf("step %4d layer %-4s SIZE MISMATCH %zu vs %zu\n", rc->step, il.c_str(), s.size(), d.size());
        rc->n_bad++;
        g_sparse.erase(a);
        g_dense.erase(b);
        return;
    }

    double num = 0.0;
    double den = 0.0;
    double amx = 0.0;

    for (size_t i = 0; i < s.size(); ++i) {
        const double e = (double) s[i] - (double) d[i];
        num += e*e;
        den += (double) d[i]*(double) d[i];
        if (std::fabs(e) > amx) {
            amx = std::fabs(e);
        }
    }

    const double nmse = den > 0.0 ? num/den : num;

    rc->n_cmp++;
    if (nmse > rc->nmse_max) {
        rc->nmse_max = nmse;
    }
    if (amx > rc->absd_max) {
        rc->absd_max = amx;
    }
    if (!(nmse <= 5e-4)) {
        rc->n_bad++;
    }

    printf("step %4d layer %-4s n=%8zu nmse=%.6e max_abs_diff=%.6e%s\n",
            rc->step, il.c_str(), s.size(), nmse, amx, nmse <= 5e-4 ? "" : "   OVER");

    g_sparse.erase(a);
    g_dense.erase(b);
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    run_ctx * rc = (run_ctx *) user_data;

    if (ask) {
        return want_tensor(t->name);
    }

    if (!want_tensor(t->name) || t->type != GGML_TYPE_F32) {
        return true;
    }

    const size_t n = ggml_nelements(t);

    std::vector<float> v(n);
    ggml_backend_tensor_get(t, v.data(), 0, n*sizeof(float));

    const std::string name = t->name;

    if (name.compare(0, 12, "kqv_out_ref-") == 0) {
        g_dense[name.substr(12)] = std::move(v);
        compare(rc, name.substr(12));
    } else {
        g_sparse[name.substr(8)] = std::move(v);
        compare(rc, name.substr(8));
    }

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <n_prompt> <n_decode> [n_ctx] [n_ubatch] [n_cpu_moe]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const int    n_prompt   = atoi(argv[2]);
    const int    n_decode   = atoi(argv[3]);
    const int    n_ctx      = argc > 4 ? atoi(argv[4]) : 32768;
    const int    n_ubatch   = argc > 5 ? atoi(argv[5]) : 2048;
    const int    n_cpu_moe  = argc > 6 ? atoi(argv[6]) : 24;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    std::vector<llama_model_tensor_buft_override> ovr;
    std::vector<std::string> pats;

    const int n_layer_all = 48;
    for (int i = n_layer_all - n_cpu_moe; i < n_layer_all; ++i) {
        char buf[128];
        snprintf(buf, sizeof(buf), "blk\\.%d\\.ffn_(up|down|gate)_(exps|shexp)", i);
        pats.push_back(buf);
    }
    for (auto & p : pats) {
        ovr.push_back({ p.c_str(), ggml_backend_cpu_buffer_type() });
    }
    ovr.push_back({ nullptr, nullptr });
    mparams.tensor_buft_overrides = ovr.data();

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    run_ctx rc;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx;
    cparams.n_batch           = n_prompt > n_ubatch ? n_prompt : n_ubatch;
    cparams.n_ubatch          = n_ubatch;
    cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k            = GGML_TYPE_Q4_0;
    cparams.type_v            = GGML_TYPE_Q4_0;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &rc;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    std::vector<llama_token> toks(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        toks[i] = 1000 + (i*7919) % 20000;
    }

    llama_batch batch = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    int pos = n_prompt;

    for (int i = 0; i < n_decode; ++i) {
        rc.step = i + 1;

        llama_token t = 1000 + (pos*7919) % 20000;

        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b1)) {
            fprintf(stderr, "decode %d failed\n", i);
            return 1;
        }
        pos++;
    }

    printf("\nSUMMARY comparisons=%ld over_tolerance=%ld nmse_max=%.6e max_abs_diff=%.6e -> %s\n",
            rc.n_cmp, rc.n_bad, rc.nmse_max, rc.absd_max,
            (rc.n_cmp > 0 && rc.n_bad == 0) ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return (rc.n_cmp > 0 && rc.n_bad == 0) ? 0 : 1;
}
