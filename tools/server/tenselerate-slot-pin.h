#pragma once

#include <cstddef>
#include <string>
#include <vector>

// TENSELERATE: pinned slots - a slot that may donate its context but is never
// selected to serve a request.
//
// The prefix-template pattern (issue #67, feature 2) is: warm one slot with the
// system prompt once, then let every new agent fork from it by reference
// (`fork_from`, or the automatic donor search behind LLAMA_SERVER_SLOT_FORK).
// Zero bytes cross the 2 GB/s link and a new agent starts in one decode step
// instead of a ~40 s prefill.
//
// That pattern is broken by slot selection as it stands, in a way that is
// silent and gets worse the better the template works. Both the LRU fallback
// and the fork-target search pick the idle slot with the OLDEST t_last_used. A
// template is warmed once and then never serves anything, so its timestamp is
// the oldest by construction: it is the FIRST slot either path evicts. The
// template survives exactly until one request arrives that does not share its
// prefix, and after that every agent silently pays the full prefill again with
// nothing in the log to say why.
//
// Pinning fixes it: a pinned slot is skipped by both selectors, so it can only
// ever be a donor.
//
//   LLAMA_SERVER_PIN_SLOTS=0        pin slot 0 as the template
//   LLAMA_SERVER_PIN_SLOTS=0,1      pin two (e.g. two different system prompts)
//   unset / empty                   nothing is pinned; behaviour unchanged
//
// Pinning every slot would leave nothing to serve from, so a spec that names
// them all is rejected at startup rather than deadlocking the first request.

// Parse a comma-separated slot-id list. Returns the ids in the spec; sets
// `error` non-empty when the spec is malformed or would pin every slot, in
// which case the caller must refuse to start. n_slots <= 0 skips that check
// (used by the tests to check parsing alone).
std::vector<int> tenselerate_parse_pinned_slots(const std::string & spec,
                                                int                 n_slots,
                                                std::string       & error);

// Read LLAMA_SERVER_PIN_SLOTS. Same contract as above.
std::vector<int> tenselerate_pinned_slots_from_env(int n_slots, std::string & error);

// Is this slot id pinned? Cheap: the list is a handful of entries at most.
bool tenselerate_slot_is_pinned(const std::vector<int> & pinned, int id);
