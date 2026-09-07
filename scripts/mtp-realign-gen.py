#!/usr/bin/env python3
"""
mtp-realign-gen - regenerate completions with the served model, for training a
draft head that agrees with THIS trunk (docs/mtp-realign-davidau.md).

Reads prompts as JSONL - either {"messages": [...]} (OpenAI chat form) or
{"prompt": "..."} - asks the local OpenAI-compatible server for a completion
with thinking kept, and appends {"id", "messages", "reasoning", "completion",
"reasoning_effort", "model"} to the output JSONL. Resumable: ids already in
the output file are skipped. Standard library only.

  python3 scripts/mtp-realign-gen.py prompts.jsonl -o gen.jsonl \\
      --base-url http://127.0.0.1:8080/v1 --model tenselerate \\
      --reasoning low --xhigh-every 4 --max-tokens 2048 --workers 4
  python3 scripts/mtp-realign-gen.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def load_prompts(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
            out.append({"id": row.get("id", n), "messages": msgs})
    return out


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                continue
    return ids


def request_completion(base_url: str, model: str, messages: list[dict], *,
                       reasoning: str, max_tokens: int, timeout: float) -> dict:
    """One non-streaming chat completion; thinking comes back as reasoning_content."""
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "stream": False, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": reasoning},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    return {"completion": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "model": data.get("model", model)}


def run(args: argparse.Namespace) -> int:
    prompts = load_prompts(Path(args.prompts))
    out_path = Path(args.out)
    skip = done_ids(out_path)
    todo = [p for p in prompts if p["id"] not in skip]
    print(f"{len(prompts)} prompts, {len(skip)} done, {len(todo)} to go", file=sys.stderr)
    lock = threading.Lock()
    failures = 0

    def work(i_p):
        i, p = i_p
        effort = "xhigh" if args.xhigh_every and i % args.xhigh_every == 0 else args.reasoning
        try:
            res = request_completion(args.base_url, args.model, p["messages"],
                                     reasoning=effort, max_tokens=args.max_tokens,
                                     timeout=args.timeout)
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            return p["id"], None, f"{type(e).__name__}: {e}"
        row = {"id": p["id"], "messages": p["messages"], "reasoning_effort": effort, **res}
        return p["id"], row, None

    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for pid, row, err in pool.map(work, enumerate(todo)):
            with lock:
                if err:
                    failures += 1
                    print(f"  id {pid}: {err}", file=sys.stderr)
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print(f"wrote {len(todo) - failures} rows to {out_path}; {failures} failed", file=sys.stderr)
    return 1 if failures and failures == len(todo) else 0


class _StubHandler(BaseHTTPRequestHandler):
    """A llama-server stand-in: echoes the prompt back with a fake thinking block."""
    def do_POST(self):  # noqa: N802 - http.server API
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        assert body["chat_template_kwargs"]["reasoning_effort"] in ("low", "medium", "high", "xhigh")
        user = body["messages"][-1]["content"]
        reply = {"model": "stub", "choices": [{"message": {
            "role": "assistant", "reasoning_content": "thinking about " + user,
            "content": "answer to " + user}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quiet
        pass


def self_test() -> int:
    import tempfile
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    with tempfile.TemporaryDirectory() as d:
        prompts = Path(d, "p.jsonl")
        prompts.write_text('{"id": "a", "prompt": "one"}\n'
                           '{"messages": [{"role": "user", "content": "two"}]}\n')
        out = Path(d, "o.jsonl")
        ns = argparse.Namespace(prompts=str(prompts), out=str(out), base_url=base,
                                model="stub", reasoning="low", xhigh_every=2,
                                max_tokens=16, workers=2, timeout=5.0)
        assert run(ns) == 0
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert {r["id"] for r in rows} == {"a", 1}, rows
        assert all(r["reasoning"].startswith("thinking") for r in rows)
        assert {r["reasoning_effort"] for r in rows} == {"low", "xhigh"}
        # resumable: a second run adds nothing
        assert run(ns) == 0
        assert len(out.read_text().splitlines()) == 2
    srv.shutdown()
    print("self-test OK: prompt forms, thinking kept, effort mix, resume")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="?", help="JSONL of {messages} or {prompt}")
    ap.add_argument("-o", "--out", default="mtp-realign-gen.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="tenselerate")
    ap.add_argument("--reasoning", default="low", choices=("low", "medium", "high", "xhigh"))
    ap.add_argument("--xhigh-every", type=int, default=4,
                    help="every Nth prompt at xhigh so the head sees long thinking too (0 = never)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4, help="match the server's slots")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.prompts:
        ap.error("prompts.jsonl is required (or --self-test)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
