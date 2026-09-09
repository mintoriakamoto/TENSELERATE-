#pragma once

// TENSELERATE: Ampere back-ports of scheduling choices upstream gates to Ada+.
//
// Several llama.cpp CUDA decisions are gated on compute capability but are pure
// *scheduling*, not instructions: the code compiles and runs on sm_80, only the
// heuristic excludes it. On the CMP 170HX (GA100, sm_80, FP16 tensor cores
// restored but dp4a throttled) those heuristics were tuned on cards whose
// balance is nothing like this one, so they are worth re-measuring rather than
// inheriting. Each is opt-in and off by default; see docs/ampere-backports.md.
//
//   GGML_CUDA_FATTN_ADA_GATE=1
//       Take the Ada+ flash-attention kernel choice on Ampere. The one that
//       matters here: with quantized KV, Ada+ keeps batch width 2 on the vector
//       kernel while Ampere falls to the MMA kernel at width 2 - and width 2 is
//       exactly an MTP depth-1 verify pass.
//
//   GGML_CUDA_FATTN_STREAM_K=1|0
//       Force stream-k work decomposition on or off, overriding the
//       "Ada+ or tile efficiency < 75%" heuristic. Stream-k splits the KV loop
//       across SMs and fixes up partials; it is scheduling, available on sm_80.

#include "common.cuh"

#include <cstdlib>

// the compute capability fattn kernel selection should *behave* as, which is the
// real one unless the Ada-gate back-port is enabled on an Ampere-class device
static inline int tenselerate_fattn_cc(const int cc) {
    if (cc >= GGML_CUDA_CC_ADA_LOVELACE || !GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return cc;
    }
    if (cc < GGML_CUDA_CC_AMPERE) {
        return cc;
    }
    const char * env = getenv("GGML_CUDA_FATTN_ADA_GATE");
    if (env != nullptr && env[0] == '1') {
        return GGML_CUDA_CC_ADA_LOVELACE;
    }
    return cc;
}

// -1 = follow upstream's heuristic, 0 = force off, 1 = force on
static inline int tenselerate_fattn_stream_k_override() {
    const char * env = getenv("GGML_CUDA_FATTN_STREAM_K");
    if (env == nullptr) {
        return -1;
    }
    if (env[0] == '0') {
        return 0;
    }
    if (env[0] == '1') {
        return 1;
    }
    return -1;
}
