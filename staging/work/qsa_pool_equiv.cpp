// Exact-equivalence harness for the QSA pooled-key cache.
//
// The cache is pure memoization: the pooled/normed/roped block keys it hands the scorer must be
// bit-identical to the ones the old path re-derives every token. This runs a fixed token sequence
// (no sampling, so both arms take the same path regardless of the stack's known run-to-run drift)
// and hashes every indexer tensor of every QSA layer at every step. Run once with the cache and
// once with LLAMA_QSA_NO_POOL_CACHE=1 and diff the output: it must match line for line.
//
// build (inside the build container):
//   icpx -fsycl -O2 -I/app/include -I/app/ggml/include -I/app/common \
//        staging/work/qsa_pool_equiv.cpp -L/app/build/bin -lllama -lggml -lggml-base -o qsa_pool_equiv

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>

struct hash_ctx {
    int step = 0;
};

static bool want_tensor(const char * name) {
    static const char * pre[] = {
        "indexer_k-", "indexer_k_ref-", "indexer_score-",
        "indexer_top_k_blocks-", "indexer_score_tokens-",
    };

    for (const char * p : pre) {
        if (strncmp(name, p, strlen(p)) == 0) {
            return true;
        }
    }

    return false;
}

static std::map<std::string, std::vector<float>> g_ref;

// indexer_k_ref-<il> is the from-scratch derivation, indexer_k-<il> what the cache handed the
// scorer. They run in the same graph on the same inputs, so a correct cache makes them
// bit-identical over every block the bias can select.
static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    hash_ctx * hc = (hash_ctx *) user_data;

    if (ask) {
        return want_tensor(t->name);
    }

    if (!want_tensor(t->name)) {
        return true;
    }

    const size_t nb = ggml_nbytes(t);

    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < nb; ++i) {
        h ^= buf[i];
        h *= 1099511628211ull;
    }

    const std::string name = t->name;

    if (name.compare(0, 14, "indexer_k_ref-") == 0) {
        const size_t n = nb/sizeof(float);
        std::vector<float> v(n);
        memcpy(v.data(), buf.data(), nb);
        g_ref[name.substr(14)] = std::move(v);

        return true;
    }

    if (name.compare(0, 10, "indexer_k-") == 0) {
        auto it = g_ref.find(name.substr(10));
        if (it != g_ref.end()) {
            const std::vector<float> & ref = it->second;
            const size_t row  = (size_t) t->ne[0];
            const size_t rows = ref.size()/row;

            std::vector<float> cur(nb/sizeof(float));
            memcpy(cur.data(), buf.data(), nb);

            long long first = -1;
            long long ndiff = 0;

            for (size_t r = 0; r < rows && r*row < cur.size(); ++r) {
                bool d = false;
                for (size_t i = 0; i < row; ++i) {
                    if (memcmp(&ref[r*row + i], &cur[r*row + i], sizeof(float)) != 0) {
                        d = true;
                        break;
                    }
                }
                if (d) {
                    if (first < 0) {
                        first = (long long) r;
                    }
                    ndiff++;
                }
            }

            printf("step %4d %-20s rows=%6zu first_diff_row=%8lld n_diff_rows=%8lld\n",
                    hc->step, t->name, rows, first, ndiff);

            g_ref.erase(it);

            return true;
        }
    }

    printf("step %4d %-28s ne=%6lld x %6lld x %4lld type=%-6s hash=%016llx\n",
            hc->step, t->name,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
            ggml_type_name(t->type), (unsigned long long) h);

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <n_prompt> <n_decode> [n_ctx] [n_ubatch] [n_cpu_moe]\n", argv[0]);
        return 1;
    }

    const char *  model_path = argv[1];
    const int     n_prompt   = atoi(argv[2]);
    const int     n_decode   = atoi(argv[3]);
    const int     n_ctx      = argc > 4 ? atoi(argv[4]) : 8192;
    const int     n_ubatch   = argc > 5 ? atoi(argv[5]) : 512;
    const int     n_cpu_moe  = argc > 6 ? atoi(argv[6]) : 24;
    const bool    mutate     = argc > 7 && atoi(argv[7]) != 0;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;

    // keep the MoE weights of the last n_cpu_moe layers on the host, as every benchmark here does
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

    hash_ctx hc;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx;
    cparams.n_batch           = n_prompt > n_ubatch ? n_prompt : n_ubatch;
    cparams.n_ubatch          = n_ubatch;
    cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k            = GGML_TYPE_Q4_0;
    cparams.type_v            = GGML_TYPE_Q4_0;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &hc;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    // a fixed, sampling-free token stream: both arms must walk the same positions
    std::vector<llama_token> toks(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        toks[i] = 1000 + (i*7919) % 20000;
    }

    llama_batch batch = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    int pos = n_prompt;

    for (int i = 0; i < n_decode; ++i) {
        hc.step = i + 1;

        // exercise the cache lifecycle: drop a tail and refill it, then shift the whole sequence.
        // both free or move cells, which is what the fingerprint alone cannot see
        if (mutate && i == n_decode/3) {
            fprintf(stderr, "### seq_rm from pos %d\n", pos - 100);
            llama_memory_seq_rm(mem, 0, pos - 100, -1);
            pos -= 100;

            std::vector<llama_token> refill(100);
            for (int k = 0; k < 100; ++k) {
                refill[k] = 1000 + ((pos + k)*104729) % 20000;
            }

            llama_batch br = llama_batch_get_one(refill.data(), 100);
            if (llama_decode(ctx, br)) {
                fprintf(stderr, "refill failed\n");
                return 1;
            }
            pos += 100;
        }

        if (mutate && i == 2*n_decode/3) {
            fprintf(stderr, "### seq_rm head + shift by -64\n");
            llama_memory_seq_rm (mem, 0, 0, 64);
            llama_memory_seq_add(mem, 0, 64, -1, -64);
            pos -= 64;
        }

        llama_token t = 1000 + (pos*7919) % 20000;

        llama_batch b1 = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b1)) {
            fprintf(stderr, "decode %d failed\n", i);
            return 1;
        }
        pos++;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
