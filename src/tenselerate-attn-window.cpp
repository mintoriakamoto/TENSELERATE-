#include "tenselerate-attn-window.h"

#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>

bool tenselerate_attn_window_apply(llama_hparams & hparams, const char * caller) {
    const char * env = getenv("LLAMA_ATTN_WINDOW");
    if (env == nullptr) {
        return false;
    }

    const long w = strtol(env, nullptr, 10);
    if (w <= 0) {
        return false;
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.n_swa    = (uint32_t) w;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = il < hparams.n_layer() && !hparams.is_recr_impl[il];
    }

    // the window layers are the model's own attention layers: same rope
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    uint32_t n_sink = 4;
    if (const char * env_sink = getenv("LLAMA_ATTN_SINKS")) {
        n_sink = (uint32_t) std::max(0L, strtol(env_sink, nullptr, 10));
    }
    llama_hparams::n_swa_sink = n_sink;

    LLAMA_LOG_INFO("%s: LLAMA_ATTN_WINDOW: attention layers bounded to %u tokens + %u sinks (hybrid-iswa)\n",
            caller, hparams.n_swa, n_sink);

    return true;
}
