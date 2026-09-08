// Bounded attention window on a hybrid GDN model (TENSELERATE, LLAMA_ATTN_WINDOW).
//
// Two properties of the window, checked on the generated tiny qwen35 model:
//
//   1. Identity below the window: with LLAMA_ATTN_WINDOW >= the prompt length the
//      logits are bit-identical to the unbounded model - the window changes nothing
//      until a sequence outgrows it.
//   2. Unbounded sequences: with a window much smaller than the sequence, decoding
//      continues past the window AND past the model's training context (n_ctx_train)
//      with finite logits, on a context whose KV cells for the attention layers are
//      only window + ubatch wide. The recurrent layers carry the rest.
//
// The window is a process-wide model-load setting, so each case loads its own model.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static std::vector<llama_token> make_tokens(int n_vocab, uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(1, n_vocab - 1);
    std::vector<llama_token> toks(n);
    for (auto & t : toks) {
        t = dist(rng);
    }
    return toks;
}

// decode tokens[0..n) in chunks of n_chunk, return the logits of the last token
static bool decode_all(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t n_chunk, std::vector<float> & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch batch = llama_batch_init(n_chunk, 0, 1);
    for (uint32_t i0 = 0; i0 < tokens.size(); i0 += n_chunk) {
        common_batch_clear(batch);
        const uint32_t i1 = std::min<uint32_t>(tokens.size(), i0 + n_chunk);
        for (uint32_t i = i0; i < i1; ++i) {
            common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, i + 1 == tokens.size());
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at pos %u\n", i0);
            llama_batch_free(batch);
            return false;
        }
    }
    llama_batch_free(batch);
    const float * logits = llama_get_logits_ith(ctx, -1);
    out.assign(logits, logits + n_vocab);
    return true;
}

struct run {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
};

static run load(common_params & params, const char * window, uint32_t n_ctx, uint32_t n_batch) {
    if (window) {
        setenv("LLAMA_ATTN_WINDOW", window, 1);
    } else {
        unsetenv("LLAMA_ATTN_WINDOW");
    }
    run r;
    auto mparams = common_model_params_to_llama(params);
    r.model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!r.model) {
        return r;
    }
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = n_batch;
    r.ctx = llama_init_from_model(r.model, cparams);
    return r;
}

static void unload(run & r) {
    if (r.ctx)   llama_free(r.ctx);
    if (r.model) llama_model_free(r.model);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    int rc = 0;

    // --- case 1: identity below the window ------------------------------------
    {
        const uint32_t n_prompt = 96;
        std::vector<float> ref, win;

        run a = load(params, nullptr, 256, 32);
        if (!a.model || !a.ctx) { fprintf(stderr, "FAIL: load (unbounded)\n"); return 1; }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(a.model));
        const auto toks = make_tokens(n_vocab, n_prompt, 42);
        if (!decode_all(a.ctx, toks, 32, ref)) { fprintf(stderr, "FAIL: decode (unbounded)\n"); return 1; }
        unload(a);

        run b = load(params, "128", 256, 32);   // window 128 > prompt 96
        if (!b.model || !b.ctx) { fprintf(stderr, "FAIL: load (window 128)\n"); return 1; }
        if (!decode_all(b.ctx, toks, 32, win)) { fprintf(stderr, "FAIL: decode (window 128)\n"); return 1; }
        unload(b);

        size_t n_diff = 0;
        float  max_abs = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = std::fabs(ref[i] - win[i]);
            if (d > 1e-5f) ++n_diff;
            max_abs = std::max(max_abs, d);
        }
        if (n_diff != 0) {
            fprintf(stderr, "FAIL identity: %zu/%zu logits differ (max |d| = %g) with window >= prompt\n", n_diff, ref.size(), max_abs);
            rc = 1;
        } else {
            printf("OK   identity: window 128 == unbounded on a %u-token prompt (max |d| = %g)\n", n_prompt, max_abs);
        }
    }

    // --- case 2: sequence past the window and past n_ctx_train -----------------
    {
        run c = load(params, "32", 2048, 64);
        if (!c.model || !c.ctx) { fprintf(stderr, "FAIL: load (window 32)\n"); return 1; }
        const int      n_vocab     = llama_vocab_n_tokens(llama_model_get_vocab(c.model));
        const uint32_t n_ctx_train = llama_model_n_ctx_train(c.model);
        const uint32_t n_seq       = std::min<uint32_t>(2000, n_ctx_train + 512);
        const auto toks = make_tokens(n_vocab, n_seq, 7);
        std::vector<float> out;
        if (!decode_all(c.ctx, toks, 64, out)) {
            fprintf(stderr, "FAIL long: decode stopped before %u tokens (n_ctx_train = %u)\n", n_seq, n_ctx_train);
            rc = 1;
        } else {
            bool finite = true;
            for (float v : out) {
                if (!std::isfinite(v)) { finite = false; break; }
            }
            if (!finite) {
                fprintf(stderr, "FAIL long: non-finite logits at %u tokens\n", n_seq);
                rc = 1;
            } else {
                printf("OK   long: %u tokens decoded with window 32 (n_ctx_train = %u), logits finite\n", n_seq, n_ctx_train);
            }
        }
        unload(c);
    }

    llama_backend_free();
    printf(rc == 0 ? "PASS\n" : "FAILED\n");
    return rc;
}
