#pragma once

// TENSELERATE: bounded attention window for hybrid (recurrent + attention) models.
//
// LLAMA_ATTN_WINDOW=N turns every non-recurrent layer into a sliding-window
// layer of N tokens; LLAMA_ATTN_SINKS=K (default 4) pins the first K positions
// of each sequence in attention forever. The recurrent layers have no positions
// and a fixed state, so the sequence is unbounded while KV per slot is O(N).
// Off unless the variable is set. Called once from the architecture's
// load_arch_hparams(); create_memory() then selects hybrid-iswa memory from
// hparams.swa_type, and the graph picks the iSWA attention input.
//
// Kept in its own file so the upstream sources carry one call each; see
// docs/upstream-sync.md ("fork footprint") and docs/bounded-window-serving.md.

struct llama_hparams;

// returns true if the window was enabled from the environment
bool tenselerate_attn_window_apply(llama_hparams & hparams, const char * caller);
