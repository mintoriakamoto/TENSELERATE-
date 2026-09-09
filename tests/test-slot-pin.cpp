// TENSELERATE: pinned slot spec parsing.
//
// A pinned slot donates its context but is never selected to serve, which is
// what makes the prefix-template pattern of issue #67 survive. The guards that
// use this list are two `continue`s in slot selection and need a live server to
// exercise; what is testable here is the part that decides which ids are pinned
// at all, including the two refusals that must happen at startup rather than as
// a server that accepts a request and never answers it.

#include "tenselerate-slot-pin.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        g_fail++;
    }
}

static void expect_ok(const std::string & spec, int n_slots,
                      const std::vector<int> & want, const char * what) {
    std::string err;
    const auto got = tenselerate_parse_pinned_slots(spec, n_slots, err);
    if (!err.empty()) {
        printf("FAIL: %s - unexpected error: %s\n", what, err.c_str());
        g_fail++;
        return;
    }
    if (got != want) {
        printf("FAIL: %s - got [", what);
        for (size_t i = 0; i < got.size(); i++) printf("%s%d", i ? "," : "", got[i]);
        printf("], want [");
        for (size_t i = 0; i < want.size(); i++) printf("%s%d", i ? "," : "", want[i]);
        printf("]\n");
        g_fail++;
    }
}

static void expect_err(const std::string & spec, int n_slots, const char * what) {
    std::string err;
    const auto got = tenselerate_parse_pinned_slots(spec, n_slots, err);
    if (err.empty()) {
        printf("FAIL: %s - expected an error, got %zu ids\n", what, got.size());
        g_fail++;
        return;
    }
    if (!got.empty()) {
        printf("FAIL: %s - error set but %zu ids returned\n", what, got.size());
        g_fail++;
    }
}

int main() {
    // the ordinary cases
    expect_ok("",       8, {},        "empty spec pins nothing");
    expect_ok("0",      8, {0},       "a single id");
    expect_ok("0,2",    8, {0, 2},    "two ids");
    expect_ok("2,0",    8, {0, 2},    "ids come back sorted");
    expect_ok("0,0,1",  8, {0, 1},    "duplicates collapse");
    expect_ok(" 0 , 1 ", 8, {0, 1},   "surrounding spaces are tolerated");
    expect_ok("0,",     8, {0},       "a trailing comma is not an error");
    expect_ok("0,,1",   8, {0, 1},    "an empty field is skipped");

    // malformed specs must be refused, not silently ignored: a typo that
    // pinned nothing would look like it worked until the template vanished
    expect_err("x",     8, "a non-numeric id is rejected");
    expect_err("0,x",   8, "a non-numeric id among valid ones is rejected");
    expect_err("1two",  8, "trailing garbage after a number is rejected");
    expect_err("-1",    8, "a negative id is rejected");
    expect_err("8",     8, "an id past the last slot is rejected");
    expect_err("0,8",   8, "an out-of-range id among valid ones is rejected");

    // pinning everything would leave nothing to serve from: refuse at startup
    // rather than deadlock on the first request
    expect_err("0",         1, "pinning the only slot is refused");
    expect_err("0,1",       2, "pinning both of two slots is refused");
    expect_err("0,1,2,3",   4, "pinning all four is refused");
    expect_ok ("0,1,2",     4, {0, 1, 2}, "pinning all but one is allowed");

    // n_slots <= 0 means "parse only", used before the slot count is known
    expect_ok("0,99", 0, {0, 99}, "no range check when the slot count is unknown");

    // membership
    std::string err;
    const auto pinned = tenselerate_parse_pinned_slots("0,3", 8, err);
    check(err.empty(), "membership fixture parses");
    check(tenselerate_slot_is_pinned(pinned, 0),  "slot 0 is pinned");
    check(tenselerate_slot_is_pinned(pinned, 3),  "slot 3 is pinned");
    check(!tenselerate_slot_is_pinned(pinned, 1), "slot 1 is not pinned");
    check(!tenselerate_slot_is_pinned(pinned, 8), "an unknown id is not pinned");
    check(!tenselerate_slot_is_pinned({}, 0),     "nothing is pinned when the list is empty");

    if (g_fail == 0) {
        printf("test-slot-pin: all checks passed\n");
        return 0;
    }
    printf("test-slot-pin: %d check(s) failed\n", g_fail);
    return 1;
}
