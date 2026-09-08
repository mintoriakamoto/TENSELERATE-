// Sequence fork on a hybrid GDN model (TENSELERATE, "fork, don't fetch").
//
// A sequence on this model is window KV cells plus one recurrent state cell.
// llama_memory_seq_cp shares both by reference (KV: a sequence bit on the
// cells; recurrent: the tail cell, copied on the first write). The server's
// fork-instead-of-fetch path relies on three properties, checked here on the
// generated tiny qwen35 with a unified KV cache:
//
//   1. child == fresh: a sequence forked after prefix A and continued with B
//      produces the same logits as a fresh sequence over A+B, bit-exact.
//   2. donor intact: the donor, continued with C after the fork, equals a fresh
//      sequence over A+C - the child's writes did not leak into the donor's
//      recurrent state (copy-on-write).
//   3. chained: a fork of the child (after B) continued with D equals fresh
//      A+B+D - forks compose.
//
// Batch shapes are kept identical between forked and fresh runs so the same
// kernels see the same shapes.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
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

// decode `toks` on `seq` starting at position `pos0`, one batch; return last-token logits
static bool decode_seq(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos pos0, std::vector<float> & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        return false;
    }
    const float * logits = llama_get_logits_ith(ctx, -1);
    out.assign(logits, logits + n_vocab);
    return true;
}

static bool same(const std::vector<float> & a, const std::vector<float> & b, const char * what) {
    size_t n_diff = 0;
    float  max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > 0.0f) ++n_diff;
        max_abs = std::max(max_abs, d);
    }
    if (n_diff != 0) {
        fprintf(stderr, "FAIL %s: %zu/%zu logits differ (max |d| = %g)\n", what, n_diff, a.size(), max_abs);
        return false;
    }
    printf("OK   %s: bit-exact (%zu logits)\n", what, a.size());
    return true;
}

static llama_context * make_ctx(common_params & params, llama_model * model, uint32_t n_seq_max) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 128;
    cparams.n_ubatch   = 128;
    cparams.kv_unified = true;   // the server's --kv-unified: seq_cp shares cells instead of copying
    return llama_init_from_model(model, cparams);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "FAIL: model load\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const uint32_t nA = 48, nB = 16, nC = 16, nD = 16;
    const auto A = make_tokens(n_vocab, nA, 1);
    const auto B = make_tokens(n_vocab, nB, 2);
    const auto C = make_tokens(n_vocab, nC, 3);
    const auto D = make_tokens(n_vocab, nD, 4);

    int rc = 0;
    std::vector<float> tmp, child_B, donor_C, grandchild_D, fresh_AB, fresh_AC, fresh_ABD;

    // --- forked run: seq 0 = donor, seq 1 = child, seq 2 = grandchild -------------
    {
        llama_context * ctx = make_ctx(params, model, 3);
        if (!ctx) { fprintf(stderr, "FAIL: context\n"); return 1; }
        llama_memory_t mem = llama_get_memory(ctx);

        if (!decode_seq(ctx, 0, A, 0, tmp)) { fprintf(stderr, "FAIL: decode A\n"); return 1; }

        llama_memory_seq_cp(mem, 0, 1, -1, -1);                         // fork: donor 0 -> child 1
        if (!decode_seq(ctx, 1, B, nA, child_B)) { fprintf(stderr, "FAIL: decode B on child\n"); return 1; }
        if (!decode_seq(ctx, 0, C, nA, donor_C)) { fprintf(stderr, "FAIL: decode C on donor\n"); return 1; }

        llama_memory_seq_cp(mem, 1, 2, -1, -1);                         // fork the child: 1 -> grandchild 2
        if (!decode_seq(ctx, 2, D, nA + nB, grandchild_D)) { fprintf(stderr, "FAIL: decode D on grandchild\n"); return 1; }

        llama_free(ctx);
    }

    // --- fresh runs, same batch shapes ----------------------------------------------
    {
        llama_context * ctx = make_ctx(params, model, 1);
        if (!decode_seq(ctx, 0, A, 0, tmp) || !decode_seq(ctx, 0, B, nA, fresh_AB)) { fprintf(stderr, "FAIL: fresh A+B\n"); return 1; }
        llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(params, model, 1);
        if (!decode_seq(ctx, 0, A, 0, tmp) || !decode_seq(ctx, 0, C, nA, fresh_AC)) { fprintf(stderr, "FAIL: fresh A+C\n"); return 1; }
        llama_free(ctx);
    }
    {
        llama_context * ctx = make_ctx(params, model, 1);
        if (!decode_seq(ctx, 0, A, 0, tmp) || !decode_seq(ctx, 0, B, nA, tmp) || !decode_seq(ctx, 0, D, nA + nB, fresh_ABD)) { fprintf(stderr, "FAIL: fresh A+B+D\n"); return 1; }
        llama_free(ctx);
    }

    if (!same(child_B,      fresh_AB,  "child == fresh(A+B)"))        rc = 1;
    if (!same(donor_C,      fresh_AC,  "donor intact == fresh(A+C)")) rc = 1;
    if (!same(grandchild_D, fresh_ABD, "chained fork == fresh(A+B+D)")) rc = 1;

    llama_model_free(model);
    llama_backend_free();
    printf(rc == 0 ? "PASS\n" : "FAILED\n");
    return rc;
}
