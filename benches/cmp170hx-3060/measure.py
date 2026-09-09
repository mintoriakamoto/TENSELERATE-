#!/usr/bin/env python3
"""
measure - throughput / draft-acceptance / loop probe for a local llama-server.

Standard library only. Talks to the OpenAI-compatible endpoint on loopback,
`stream: false`, and reads the `timings` object llama-server attaches to every
completion (tools/server/server-task.cpp, result_timings::to_json):

    timings.prompt_n / prompt_ms / prompt_per_second
    timings.predicted_n / predicted_ms / predicted_per_second
    timings.draft_n / timings.draft_n_accepted        (only when a drafter ran)

Modes (pick one):
  default            N chat completions per shape (json, code, prose), per-shape stats
  --concurrency K    K requests at once, each behind a distinct long prefix, so K
                     slots are live; aggregate tok/s (sum predicted_n / wall) + per-stream
  --prefill-tokens T one slot filled with a ~T-token synthetic prompt (numbered
                     paragraphs, calibrated through POST /tokenize), then decode
                     tok/s at that depth
  --health / --wait-health S   probe GET /health once / until ok

Sampling: nothing is sent unless --send-sampling is given, so the server's
request defaults (--temp / --repeat-penalty from the launch) are what is
measured. `--send-sampling temp=0.7,rp=1.15,top_p=0.95` overrides them from the
client side - the README's "the request wins" experiment.

--loop-check flags degenerate repetition: a 12-gram (whitespace tokens) that
occurs 4+ times in reasoning + content.

Last two stdout lines are machine-readable for the shell runner:
    KV<TAB>tok_s=..<TAB>acceptance=..<TAB>loops=..<TAB>n=..[<TAB>depth=..]
    VALUE<TAB><text for the results table's value column>

  python3 measure.py --base-url http://127.0.0.1:8089 --n 3 --loop-check
  python3 measure.py --concurrency 4 --loop-check
  python3 measure.py --prefill-tokens 250000 --slot-size 262144 --n 2
  python3 measure.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_BASE = "http://127.0.0.1:8089"
NGRAM = 12
LOOP_MIN_REPEATS = 4

# Three output shapes, short and deterministic. JSON/tool call, code, prose:
# the shapes the agent emits, in the order the README's MTP rows use.
PROMPTS = {
    "json": (
        "You are a tool-calling assistant. Reply with ONLY a JSON array (no prose, no "
        "markdown fence) of exactly 12 tool calls of the form "
        '{"name": "read_file", "arguments": {"path": "<path>", "start_line": <int>, '
        '"end_line": <int>}} covering the files src/main.c, src/util.c, src/net.c and '
        "include/config.h, three calls per file, non-overlapping line ranges of 40 lines "
        "each starting at line 1."
    ),
    "code": (
        "Write a Python module with a function parse_kv(text: str) -> dict[str, str] that "
        "parses 'key=value' lines, ignores blank lines and '#' comments, strips whitespace, "
        "raises ValueError on a line without '=', and a small unittest.TestCase with four "
        "tests. Code only, with docstrings, no explanation."
    ),
    "prose": (
        "In about 300 words of plain prose, explain why a GPU on a PCIe Gen2 x4 link "
        "loads a 16 GB model slowly but can still decode quickly once the weights are "
        "resident, and what that implies for restarting a model server often."
    ),
    # The agent edit loop, and the only shape where an n-gram drafter can show
    # what it is for: the answer is mostly a verbatim replay of text already in
    # the context. json/code/prose all generate text that never appeared in the
    # prompt, so ngram-mod must MISS on them - which is what makes them the
    # control. Not in DEFAULT_SHAPES: it would change every existing row.
    "rewrite": (
        "Here is a module.\n\n```python\n"
        "def parse_kv(text: str) -> dict[str, str]:\n"
        '    """Parse \'key=value\' lines into a dict."""\n'
        "    out: dict[str, str] = {}\n"
        "    for line in text.splitlines():\n"
        "        line = line.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        if '=' not in line:\n"
        "            raise ValueError(f'no = in line: {line!r}')\n"
        "        key, _, value = line.partition('=')\n"
        "        out[key.strip()] = value.strip()\n"
        "    return out\n"
        "```\n\n"
        "Re-emit the module exactly as given, changing only the ValueError message to "
        "read 'malformed line: {line!r}'. Code only, no explanation, no diff."
    ),
}
SHAPES = tuple(PROMPTS)
DEFAULT_SHAPES = ("json", "code", "prose")   # the three the README's rows were measured on

_ADJ = ("quiet", "red", "narrow", "old", "bright", "cold", "long", "small", "heavy", "late")
_NOUN = ("harbour", "ledger", "engine", "valley", "kettle", "bridge", "orchard", "signal",
         "window", "furnace")
_VERB = ("measures", "records", "carries", "counts", "shelters", "repeats", "divides",
         "follows", "stores", "checks")


def paragraph(i: int, tag: str = "") -> str:
    """Deterministic numbered paragraph i; `tag` makes prefixes distinct per stream."""
    a, n, v = _ADJ[i % 10], _NOUN[(i // 10) % 10], _VERB[(i // 100) % 10]
    return (f"Paragraph {i}{tag}. The {a} {n} {v} item {i * 7 % 1000} in section "
            f"{i % 37} of the register. Each entry names the {n}, its {a} state, and the "
            f"count {i % 13} that the clerk {v} before the ledger closes for the day.\n\n")


def find_loops(text: str, n: int = NGRAM, min_repeats: int = LOOP_MIN_REPEATS):
    """(count, ngram) of the most repeated n-gram if it occurs >= min_repeats times, else None."""
    words = text.split()
    if len(words) < n:
        return None
    counts = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    top, c = counts.most_common(1)[0]
    return (c, top) if c >= min_repeats else None


def parse_sampling(spec: str | None) -> dict:
    """'temp=0.7,rp=1.15,top_p=0.95,top_k=20' -> request fields; '' or None -> {}."""
    keys = {"temp": "temperature", "temperature": "temperature", "rp": "repeat_penalty",
            "repeat_penalty": "repeat_penalty", "top_p": "top_p", "top_k": "top_k",
            "min_p": "min_p"}
    out: dict = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in keys or not v:
            raise SystemExit(f"bad --send-sampling item {part!r}; want e.g. temp=0.7,rp=1.15")
        out[keys[k]] = int(v) if keys[k] == "top_k" else float(v)
    return out


def http_json(base_url: str, path: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data,
                                 method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def health_ok(base_url: str, timeout: float = 5.0) -> bool:
    try:
        return http_json(base_url, "/health", None, timeout).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def count_tokens(base_url: str, text: str, timeout: float) -> int | None:
    """POST /tokenize; None when the server has no tokenizer route (then ~4 chars/token)."""
    try:
        return len(http_json(base_url, "/tokenize", {"content": text}, timeout)["tokens"])
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def synthetic_prompt(base_url: str, target_tokens: int, tag: str, timeout: float) -> tuple[str, int]:
    """Numbered paragraphs sized to ~target_tokens by calibrating 20 of them via /tokenize."""
    sample = "".join(paragraph(i, tag) for i in range(20))
    n = count_tokens(base_url, sample, timeout)
    per_par = (n / 20.0) if n else (len(sample) / 20.0 / 4.0)
    count = max(1, int(target_tokens / per_par))
    text = "".join(paragraph(i, tag) for i in range(count))
    return text, int(count * per_par)


def chat(base_url: str, model: str, content: str, *, max_tokens: int, sampling: dict,
         timeout: float) -> dict:
    """One non-streaming chat completion; returns {text, timings, wall_s, finish}."""
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "stream": False}
    body.update(sampling)   # only when --send-sampling was given
    t0 = time.monotonic()
    data = http_json(base_url, "/v1/chat/completions", body, timeout)
    wall = time.monotonic() - t0
    choice = data["choices"][0]
    msg = choice.get("message", {})
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    return {"text": text, "timings": data.get("timings") or {}, "wall_s": wall,
            "finish": choice.get("finish_reason")}


def summarize(t: dict) -> dict:
    """Pull the fields we report out of a timings object; acceptance None without a drafter."""
    dn, da = t.get("draft_n"), t.get("draft_n_accepted")
    return {
        "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
        "tok_s": t.get("predicted_per_second"),
        "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
        "prompt_tok_s": t.get("prompt_per_second"), "cache_n": t.get("cache_n"),
        "draft_n": dn, "draft_n_accepted": da,
        "acceptance": (da / dn) if dn else None,
    }


def fmt_acc(vals) -> str:
    vals = [v for v in vals if v is not None]
    return f"{sum(vals) / len(vals):.3f}" if vals else "-"


def mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def describe(tag: str, s: dict, wall: float, loop) -> str:
    acc = f"draft {s['draft_n_accepted']}/{s['draft_n']} acc {s['acceptance']:.3f}" if s["draft_n"] else "no draft"
    lp = f"LOOP x{loop[0]}" if loop else "loop no"
    tok_s = s["tok_s"] if s["tok_s"] is not None else float("nan")
    return (f"[{tag}] predicted {s['predicted_n']} tok = {tok_s:.1f} tok/s (wall {wall:.1f} s), "
            f"prompt {s['prompt_n']} tok @ {s['prompt_tok_s'] or 0:.0f} tok/s, {acc}, {lp}")


def emit(rows: list[dict], value: str, extra: dict | None = None) -> None:
    kv = {"tok_s": f"{mean([r['tok_s'] for r in rows]) or 0:.2f}",
          "acceptance": fmt_acc(r["acceptance"] for r in rows),
          "loops": str(sum(1 for r in rows if r["loop"])), "n": str(len(rows))}
    kv.update(extra or {})
    print("KV\t" + "\t".join(f"{k}={v}" for k, v in kv.items()))
    print("VALUE\t" + value)
    sys.stdout.flush()


def run_shapes(a: argparse.Namespace, sampling: dict) -> list[dict]:
    rows, parts = [], []
    for shape in a.shapes:
        srows = []
        for i in range(a.n):
            r = chat(a.base_url, a.model, PROMPTS[shape], max_tokens=a.max_tokens,
                     sampling=sampling, timeout=a.timeout)
            s = summarize(r["timings"])
            s["loop"] = find_loops(r["text"]) if a.loop_check else None
            s["shape"], s["wall_s"] = shape, r["wall_s"]
            print(describe(f"{shape} #{i + 1}", s, r["wall_s"], s["loop"]))
            if s["loop"]:
                print(f"    repeated {NGRAM}-gram: {s['loop'][1][:100]!r}")
            srows.append(s)
        rows += srows
        loops = sum(1 for s in srows if s["loop"])
        parts.append(f"{shape} {mean([s['tok_s'] for s in srows]) or 0:.1f} tok/s "
                     f"(acc {fmt_acc(s['acceptance'] for s in srows)}, loops {loops}/{len(srows)})")
    parts.append(f"mean {mean([s['tok_s'] for s in rows]) or 0:.1f}")
    emit(rows, "; ".join(parts))
    return rows


def run_concurrency(a: argparse.Namespace, sampling: dict) -> list[dict]:
    k = a.concurrency
    prefixes = [synthetic_prompt(a.base_url, a.prefix_tokens, f"-s{j}", a.timeout)[0] for j in range(k)]
    shapes = [a.shapes[j % len(a.shapes)] for j in range(k)]
    contents = [prefixes[j] + "Now, unrelated to the register above:\n\n" + PROMPTS[shapes[j]]
                for j in range(k)]

    def one(j):
        return chat(a.base_url, a.model, contents[j], max_tokens=a.max_tokens,
                    sampling=sampling, timeout=a.timeout)

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=k) as pool:
        results = list(pool.map(one, range(k)))
    wall = time.monotonic() - t0
    rows = []
    for j, r in enumerate(results):
        s = summarize(r["timings"])
        s["loop"] = find_loops(r["text"]) if a.loop_check else None
        s["shape"], s["wall_s"] = shapes[j], r["wall_s"]
        print(describe(f"stream {j} {shapes[j]}", s, r["wall_s"], s["loop"]))
        rows.append(s)
    total = sum(s["predicted_n"] or 0 for s in rows)
    # aggregate = tokens over the whole burst; the decode-only variant excludes each
    # stream's own prompt time and is what the README's `-np N` rows report
    agg_wall = total / wall
    dec_ms = [s["predicted_ms"] for s in rows if s["predicted_ms"]]
    agg_decode = total / (max(dec_ms) / 1e3) if dec_ms else 0.0
    per = "/".join(f"{s['tok_s'] or 0:.1f}" for s in rows)
    value = (f"aggregate {agg_decode:.1f} tok/s decode-only ({agg_wall:.1f} incl. prefill), "
             f"{k} streams {per} per stream, acc {fmt_acc(s['acceptance'] for s in rows)}, "
             f"loops {sum(1 for s in rows if s['loop'])}/{k}")
    print(f"aggregate: {total} tokens in {wall:.1f} s wall")
    emit(rows, value, {"aggregate_tok_s": f"{agg_decode:.2f}", "concurrency": str(k)})
    return rows


def run_prefill(a: argparse.Namespace, sampling: dict) -> list[dict]:
    budget = a.slot_size - a.max_tokens - 1024      # template + reasoning headroom
    target = min(a.prefill_tokens, budget)
    if target < a.prefill_tokens:
        print(f"note: prefill clamped {a.prefill_tokens} -> {target} to fit slot {a.slot_size}")
    text, est = synthetic_prompt(a.base_url, target, "", a.timeout)
    content = text + ("Now, unrelated to the register above:\n\n" + PROMPTS["prose"])
    print(f"synthetic prompt ~{est} tokens ({len(text)} chars); sending {a.n} request(s)")
    rows = []
    for i in range(a.n):
        r = chat(a.base_url, a.model, content, max_tokens=a.max_tokens, sampling=sampling,
                 timeout=a.timeout)
        s = summarize(r["timings"])
        s["loop"] = find_loops(r["text"]) if a.loop_check else None
        s["shape"], s["wall_s"] = "deep-prose", r["wall_s"]
        print(describe(f"depth #{i + 1}", s, r["wall_s"], s["loop"]))
        rows.append(s)
    depth = (rows[0]["prompt_n"] or 0) + (rows[0]["cache_n"] or 0)
    first = rows[0]
    value = (f"decode {mean([s['tok_s'] for s in rows]) or 0:.1f} tok/s at depth {depth:,} tok "
             f"(prefill {first['prompt_tok_s'] or 0:.0f} tok/s, {(first['prompt_ms'] or 0) / 1e3:.0f} s; "
             f"acc {fmt_acc(s['acceptance'] for s in rows)})")
    emit(rows, value, {"depth": str(depth), "prefill_tok_s": f"{first['prompt_tok_s'] or 0:.1f}"})
    return rows


def run(a: argparse.Namespace) -> int:
    sampling = parse_sampling(a.send_sampling)
    if sampling:
        print(f"client sampling override: {sampling}")
    else:
        print("no sampling fields sent (server request defaults apply)")
    if a.concurrency:
        rows = run_concurrency(a, sampling)
    elif a.prefill_tokens:
        rows = run_prefill(a, sampling)
    else:
        rows = run_shapes(a, sampling)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"args": vars(a), "rows": rows}, f, indent=1, default=str)
    return 0


# ---------------------------------------------------------------- self-test

class _Stub(BaseHTTPRequestHandler):
    """llama-server stand-in: /health, /tokenize, /v1/chat/completions with timings."""
    seen: list = []
    loop_on = "LOOPME"

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self._send({"status": "ok"} if self.path == "/health" else {"error": self.path})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        if self.path == "/tokenize":
            self._send({"tokens": list(range(len(body["content"]) // 4))})   # 4 chars/token
            return
        _Stub.seen.append(body)
        user = body["messages"][-1]["content"]
        prompt_n = len(user) // 4
        content = ("the same twelve words repeated again and again to form a loop here "
                   * 6) if self.loop_on in user else "def parse_kv(text):\n    return {}\n"
        self._send({"choices": [{"index": 0, "finish_reason": "length", "message": {
                        "role": "assistant", "reasoning_content": "brief thought", "content": content}}],
                    "model": "stub", "object": "chat.completion",
                    "timings": {"cache_n": 0, "prompt_n": prompt_n, "prompt_ms": prompt_n / 0.8,
                                "prompt_per_token_ms": 1.25, "prompt_per_second": 800.0,
                                "predicted_n": 400, "predicted_ms": 400 / 0.0466,
                                "predicted_per_token_ms": 21.46, "predicted_per_second": 46.6,
                                "draft_n": 380, "draft_n_accepted": 335}})

    def log_message(self, *a):
        pass


def _ns(**kw) -> argparse.Namespace:
    base = dict(base_url=DEFAULT_BASE, model="stub", n=1, shapes=list(DEFAULT_SHAPES), max_tokens=64,
                send_sampling=None, concurrency=0, prefix_tokens=200, prefill_tokens=0,
                slot_size=262144, loop_check=True, timeout=10.0, json_out=None)
    base.update(kw)
    return argparse.Namespace(**base)


def self_test() -> int:
    import contextlib
    import io
    import os
    import tempfile

    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    assert health_ok(base)
    assert not health_ok("http://127.0.0.1:1")

    # loop detector on its own
    assert find_loops("a b c " * 3) is None
    lp = find_loops(" ".join(str(i % 12) for i in range(12 * 5)))
    assert lp and lp[0] >= 4, lp
    assert parse_sampling("temp=0.7,rp=1.15") == {"temperature": 0.7, "repeat_penalty": 1.15}
    assert parse_sampling("") == {}

    def capture(ns):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert run(ns) == 0
        return buf.getvalue()

    # shapes, no sampling sent
    _Stub.seen.clear()
    out = capture(_ns(base_url=base, n=2))
    assert len(_Stub.seen) == 6
    assert all(not ({"temperature", "repeat_penalty", "top_p"} & b.keys()) for b in _Stub.seen), "sampling leaked"
    assert all(b["stream"] is False and b["max_tokens"] == 64 for b in _Stub.seen)
    kv = [ln for ln in out.splitlines() if ln.startswith("KV\t")][-1]
    assert "tok_s=46.60" in kv and "acceptance=0.882" in kv and "loops=0" in kv and "n=6" in kv, kv
    assert out.splitlines()[-1].startswith("VALUE\tjson 46.6 tok/s (acc 0.882, loops 0/2); code")

    # client override reaches the request
    _Stub.seen.clear()
    capture(_ns(base_url=base, send_sampling="temp=0.7,rp=1.15"))
    assert all(b["temperature"] == 0.7 and b["repeat_penalty"] == 1.15 for b in _Stub.seen)

    # concurrency: distinct prefixes, aggregate line
    _Stub.seen.clear()
    out = capture(_ns(base_url=base, concurrency=4))
    assert len(_Stub.seen) == 4
    assert len({b["messages"][0]["content"][:400] for b in _Stub.seen}) == 4, "prefixes not distinct"
    assert "aggregate" in out.splitlines()[-1] and "4 streams" in out.splitlines()[-1]
    assert any("concurrency=4" in ln for ln in out.splitlines())

    # prefill: calibrated size, depth reported, clamped to the slot
    _Stub.seen.clear()
    out = capture(_ns(base_url=base, prefill_tokens=5000, slot_size=4000, max_tokens=64))
    assert "clamped" in out
    assert 2000 <= len(_Stub.seen[0]["messages"][0]["content"]) // 4 <= 3200, len(_Stub.seen[0]["messages"][0]["content"])
    assert "decode 46.6 tok/s at depth" in out.splitlines()[-1]

    # loop check fires on a looping completion, and json_out is written
    with tempfile.TemporaryDirectory() as d:
        PROMPTS["loopy"] = "please " + _Stub.loop_on
        try:
            out = capture(_ns(base_url=base, shapes=["loopy"], json_out=os.path.join(d, "r.json")))
        finally:
            del PROMPTS["loopy"]
        assert "loops=1" in out and "LOOP x" in out, out
        assert json.load(open(os.path.join(d, "r.json")))["rows"][0]["loop"][0] >= 4

    srv.shutdown()
    print("self-test OK: health, loop detector, sampling parse, per-shape run with no sampling "
          "leak, client override, concurrency prefixes + aggregate, calibrated prefill, loop flag")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default="tenselerate", help="the server's --alias")
    ap.add_argument("--n", type=int, default=3, help="requests per shape (or per depth run)")
    ap.add_argument("--shapes", default=",".join(DEFAULT_SHAPES),
                    help=f"comma list of {','.join(SHAPES)} (default: {','.join(DEFAULT_SHAPES)})")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--send-sampling", default=None,
                    help="client-side override, e.g. temp=0.7,rp=1.15[,top_p=..,top_k=..,min_p=..]; "
                         "default sends none, so the server's request defaults apply")
    ap.add_argument("--concurrency", type=int, default=0, help="fire K requests at once on K slots")
    ap.add_argument("--prefix-tokens", type=int, default=1500,
                    help="--concurrency: distinct synthetic prefix per stream, in tokens")
    ap.add_argument("--prefill-tokens", type=int, default=0,
                    help="deep-slot mode: synthetic prompt of ~T tokens, then decode at depth")
    ap.add_argument("--slot-size", type=int, default=262144, help="--prefill-tokens: slot ctx")
    ap.add_argument("--loop-check", action="store_true",
                    help=f"flag a {NGRAM}-gram repeated {LOOP_MIN_REPEATS}+ times in the output")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-request seconds")
    ap.add_argument("--json-out", default=None, help="write every row as JSON here")
    ap.add_argument("--health", action="store_true", help="probe GET /health once; exit 0 if ok")
    ap.add_argument("--wait-health", type=float, default=0.0,
                    help="poll GET /health up to S seconds; exit 0 when ok")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.health:
        return 0 if health_ok(a.base_url) else 1
    if a.wait_health:
        deadline = time.monotonic() + a.wait_health
        while time.monotonic() < deadline:
            if health_ok(a.base_url):
                return 0
            time.sleep(5)
        return 1
    a.shapes = [s.strip() for s in a.shapes.split(",") if s.strip()]
    for s in a.shapes:
        if s not in PROMPTS:
            ap.error(f"unknown shape {s!r}; have {SHAPES}")
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
