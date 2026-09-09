#include "tenselerate-slot-pin.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <sstream>

std::vector<int> tenselerate_parse_pinned_slots(const std::string & spec,
                                                int                 n_slots,
                                                std::string       & error) {
    error.clear();

    std::vector<int> out;
    std::set<int>    seen;

    std::stringstream ss(spec);
    std::string       item;

    while (std::getline(ss, item, ',')) {
        // trim; a trailing comma or stray spaces should not be an error
        const auto b = item.find_first_not_of(" \t");
        if (b == std::string::npos) {
            continue;
        }
        const auto e = item.find_last_not_of(" \t");
        item = item.substr(b, e - b + 1);

        size_t consumed = 0;
        int    id       = 0;
        try {
            id = std::stoi(item, &consumed);
        } catch (const std::exception &) {
            error = "not a slot id: '" + item + "'";
            return {};
        }
        if (consumed != item.size()) {
            error = "not a slot id: '" + item + "'";
            return {};
        }
        if (id < 0) {
            error = "negative slot id: '" + item + "'";
            return {};
        }
        if (n_slots > 0 && id >= n_slots) {
            error = "slot id " + std::to_string(id) + " is out of range (" +
                    std::to_string(n_slots) + " slots)";
            return {};
        }
        if (seen.insert(id).second) {
            out.push_back(id);
        }
    }

    // Pinning every slot leaves nothing to serve from. Refusing at startup is
    // far kinder than a server that accepts a request and never answers it.
    if (n_slots > 0 && (int) out.size() >= n_slots) {
        error = "pinning all " + std::to_string(n_slots) +
                " slots would leave none to serve requests";
        return {};
    }

    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> tenselerate_pinned_slots_from_env(int n_slots, std::string & error) {
    error.clear();

    const char * env = getenv("LLAMA_SERVER_PIN_SLOTS");
    if (env == nullptr || env[0] == '\0') {
        return {};
    }
    return tenselerate_parse_pinned_slots(env, n_slots, error);
}

bool tenselerate_slot_is_pinned(const std::vector<int> & pinned, int id) {
    return std::find(pinned.begin(), pinned.end(), id) != pinned.end();
}
