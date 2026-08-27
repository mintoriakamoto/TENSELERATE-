"""
OpenAI-compatible HTTP server for the TENSELERATE engine.

Stdlib only — no framework — so `python -m tenselerate.server` stands up a
/v1/chat/completions and /v1/models endpoint on 127.0.0.1:8080 that Hermes can
point at, today, running the reference engine. The compute backend is swapped
under this same HTTP contract as the CUDA kernels land; the wire protocol does
not change.

At bring-up the tokenizer is byte-level (vocab 256 == the TINY config), so the
endpoint is genuinely end to end. A real tokenizer + weights load behind the
same Generator when they exist.

Bind is loopback-only and non-configurable to off-host on purpose.
"""

from __future__ import annotations

import argparse
import sys
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tenselerate.config import CONFIGS, TINY
from tenselerate.engine.generation import Generator, SamplingParams
from tenselerate.reference.model import ReferenceModel


class ByteTokenizer:
    """Trivial UTF-8 byte tokenizer for the dev server (vocab 256)."""
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8")) or [0]

    def decode(self, tokens: list[int]) -> str:
        return bytes(t & 0xFF for t in tokens).decode("utf-8", errors="replace")


class Engine:
    def __init__(self, config_name: str = TINY.name, seed: int = 0):
        self.cfg = CONFIGS[config_name]
        self.model = ReferenceModel(self.cfg, seed=seed)
        self.generator = Generator(self.model)
        self.tokenizer = ByteTokenizer()
        self.served_name = (
            "deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-"
            "Frontier-Intelligence-Injected-OBLITERATED-GGUF:Q4_K_M"
        )

    def complete(self, prompt: str, params: SamplingParams) -> tuple[str, int, int]:
        ids = self.tokenizer.encode(prompt)
        out = self.generator.generate_list(ids, params)
        return self.tokenizer.decode(out), len(ids), len(out)


def _messages_to_prompt(messages: list[dict]) -> str:
    # minimal chat template; a real jinja template loads with the real model
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    engine: Engine    # bound per-server via a subclass in build_server

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):     # noqa: A002 - match base signature
        pass  # quiet by default

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": self.engine.served_name, "object": "model",
                 "owned_by": "tenselerate"}]})
        elif self.path == "/health":
            self._send(200, {"status": "ok", "config": self.engine.cfg.name})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid JSON"}})
            return

        params = SamplingParams(
            max_tokens=int(req.get("max_tokens", 64)),
            temperature=float(req.get("temperature", 0.0)),
            repeat_penalty=float(req.get("repeat_penalty", 1.15)),
            seed=int(req.get("seed", 0)),
        )
        if path == "/v1/chat/completions":
            prompt = _messages_to_prompt(req.get("messages", []))
        else:
            prompt = req.get("prompt", "")
        if not prompt:
            self._send(400, {"error": {"message": "empty prompt"}})
            return

        t0 = time.time()
        text, n_in, n_out = self.engine.complete(prompt, params)
        created = int(t0)
        cid = "cmpl-" + uuid.uuid4().hex[:24]
        usage = {"prompt_tokens": n_in, "completion_tokens": n_out,
                 "total_tokens": n_in + n_out}
        if path == "/v1/chat/completions":
            self._send(200, {
                "id": cid, "object": "chat.completion", "created": created,
                "model": self.engine.served_name,
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": text}}],
                "usage": usage})
        else:
            self._send(200, {
                "id": cid, "object": "text_completion", "created": created,
                "model": self.engine.served_name,
                "choices": [{"index": 0, "finish_reason": "length", "text": text}],
                "usage": usage})


def build_server(host: str, port: int, config_name: str) -> ThreadingHTTPServer:
    engine = Engine(config_name=config_name)
    handler = type("BoundHandler", (Handler,), {"engine": engine})
    return ThreadingHTTPServer((host, port), handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="loopback only by design")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", default=TINY.name, choices=sorted(CONFIGS))
    args = ap.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        ap.error("refusing to bind off-host: this dev engine is loopback-only")
    srv = build_server(args.host, args.port, args.config)
    sys.stderr.write(
        f"TENSELERATE reference engine ({args.config}) on "
        f"http://{args.host}:{args.port}/v1  (dev - reference numerics, not the CUDA backend)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
