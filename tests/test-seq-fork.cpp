// On-card sequence fork on a hybrid GDN model (TENSELERATE).
//
// A sequence on this architecture is KV cells for the attention layers plus one
// recurrent state cell for the GDN layers. Both are shareable by reference:
// llama_memory_seq_cp adds a sequence bit to the existing KV cells and points the
// destination's recurrent tail at the source's cell, which is copied on the fork's
// first write. Nothing crosses PCIe, so a delegated agent can start from a live
// parent's whole context - including the recurrent memory of everything that has
// fallen out of the attention window - for the price of one in-VRAM state copy.
//
// This test pins the property the server hook relies on:
//
//   1. A fork of sequence A at position P, continued over tokens T, produces
//      logits bit-identical to a fresh sequence decoded over prefix+T.
//   2. The parent is unaffected by the child's decoding (copy-on-write, not
//      shared mutable state).
//
// Run: test-seq-fork -m models/qwen35-dense.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<llama_token> rand_tokens(int n_vocab, uint32_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(1, n_vocab - 1);
    std::vector<llama_token> t(n);
    for (auto & x : t) x = dist(rng);
    return t;
}

// decode tokens into seq, starting at position pos0; return the last token's logits
static bool decode_seq(llama_context * ctx, llama_seq_id seq,
                       const std::vector<llama_token> & toks, llama_pos pos0,
                       std::vector<float> & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_batch batch = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) return false;
    const float * l = llama_get_logits_ith(ctx, -1);
    out.assign(l, l + n_vocab);
    return true;
}

static size_t n_differ(const std::vector<float> & a, const std::vector<float> & b, float & max_abs) {
    size_t n = 0; max_abs = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > 0.0f) ++n;
        max_abs = std::max(max_abs, d);
    }
    return n;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    common_init();
    llama_backend_init();

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 4;
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 256;
    cparams.kv_unified = true;      // the fork shares cells within one stream
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "FAIL: context\n"); llama_model_free(model); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);

    const auto prefix = rand_tokens(n_vocab, 64, 11);   // the "system prompt" / parent context
    const auto tail_a = rand_tokens(n_vocab, 24, 22);   // what the child adds
    const auto tail_b = rand_tokens(n_vocab, 24, 33);   // what the parent adds afterwards

    int rc = 0;

    // parent (seq 0): decode the prefix
    std::vector<float> lg_parent_prefix;
    if (!decode_seq(ctx, 0, prefix, 0, lg_parent_prefix)) { fprintf(stderr, "FAIL: parent prefix\n"); return 1; }

    // reference (seq 2): the same prefix, decoded independently
    std::vector<float> lg_ref_prefix;
    if (!decode_seq(ctx, 2, prefix, 0, lg_ref_prefix)) { fprintf(stderr, "FAIL: reference prefix\n"); return 1; }

    // --- fork the parent into seq 1, then continue the fork over tail_a ------
    llama_memory_seq_rm(mem, 1, -1, -1);
    llama_memory_seq_cp(mem, 0, 1, -1, -1);

    std::vector<float> lg_fork, lg_ref;
    if (!decode_seq(ctx, 1, tail_a, (llama_pos) prefix.size(), lg_fork)) {
        fprintf(stderr, "FAIL: fork continuation\n"); return 1;
    }
    if (!decode_seq(ctx, 2, tail_a, (llama_pos) prefix.size(), lg_ref)) {
        fprintf(stderr, "FAIL: reference continuation\n"); return 1;
    }

    float max_abs = 0.0f;
    size_t nd = n_differ(lg_fork, lg_ref, max_abs);
    if (nd != 0) {
        fprintf(stderr, "FAIL fork identity: %zu/%zu logits differ (max |d| = %g)\n", nd, lg_ref.size(), max_abs);
        rc = 1;
    } else {
        printf("OK   fork identity: forked seq == independently decoded seq (%zu tokens shared, max |d| = %g)\n",
               prefix.size(), max_abs);
    }

    // --- the parent must be unaffected by the child's decoding --------------
    // continue the PARENT over tail_b, and compare against a fresh sequence that
    // decoded the same prefix and never had a fork taken from it.
    std::vector<float> lg_parent_after;
    if (!decode_seq(ctx, 0, tail_b, (llama_pos) prefix.size(), lg_parent_after)) {
        fprintf(stderr, "FAIL: parent continuation\n"); return 1;
    }

    // the control decodes with the IDENTICAL batch split as the parent (prefix,
    // then tail_b), so the only difference between the two is whether a fork was
    // taken from the sequence. Decoding the same tokens as one larger batch
    // instead changes the reduction order and costs ~3e-08 of float noise, which
    // would make this check about batch shape rather than about the fork.
    llama_memory_seq_rm(mem, 3, -1, -1);
    std::vector<float> tmp;
    if (!decode_seq(ctx, 3, prefix, 0, tmp)) { fprintf(stderr, "FAIL: control prefix\n"); return 1; }
    if (!decode_seq(ctx, 3, tail_b, (llama_pos) prefix.size(), tmp)) { fprintf(stderr, "FAIL: control continuation\n"); return 1; }

    nd = n_differ(lg_parent_after, tmp, max_abs);
    if (nd != 0) {
        fprintf(stderr, "FAIL parent isolation: %zu/%zu logits differ (max |d| = %g) - the fork mutated the parent\n",
                nd, tmp.size(), max_abs);
        rc = 1;
    } else {
        printf("OK   parent isolation: parent unaffected by the fork's decoding (max |d| = %g)\n", max_abs);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf(rc == 0 ? "PASS\n" : "FAILED\n");
    return rc;
}
