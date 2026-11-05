# -*- coding: utf-8 -*-
"""Shared helpers for the 2026-11 writing evals (WritingBench + Lech Mazur).

Only stdlib is used so the scripts run under any of the project's Python
interpreters (venv-dml / bundled runtime).
"""
import json
import os
import re
import time
import urllib.error
import urllib.request


# ---------------------------------------------------------------- JSONL I/O
def read_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- HTTP / LLM
class LlamaServer(object):
    """Minimal OpenAI-compatible client for llama-server."""

    def __init__(self, base_url, model=None, timeout=900):
        self.base_url = base_url.rstrip("/")
        self.model = model or "local"
        self.timeout = timeout

    def chat(self, messages, temperature=0.7, top_p=0.9, top_k=None,
             max_tokens=800, repeat_penalty=None, seed=None, stop=None,
             retries=3):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self.base_url + "/v1/chat/completions"
        last_err = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                return {
                    "text": (msg.get("content") or "").strip(),
                    "usage": data.get("usage", {}),
                    "timings": data.get("timings", {}),
                }
            except Exception as e:  # noqa: BLE001 - surfaced after retries
                last_err = e
                time.sleep(min(20, 2 * attempt))
        raise RuntimeError("llama-server call failed after %d tries: %s" % (retries, last_err))

    def health(self):
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_object(text):
    """Extract the first JSON object from a model answer (tolerates fences)."""
    if not text:
        return None
    candidates = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    for cand in candidates:
        s = cand.find("{")
        while s != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(s, len(cand)):
                ch = cand[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            frag = cand[s:i + 1]
                            try:
                                return json.loads(frag)
                            except json.JSONDecodeError:
                                break
            s = cand.find("{", s + 1)
    return None


def clean_story(text):
    """Return the story body: inside <story> tags if present, minus [n] counters."""
    if not text:
        return ""
    m = re.search(r"<story>(.*?)</story>", text, re.S | re.I)
    body = m.group(1) if m else text
    body = re.sub(r"\[\s*\d+\s*\]", "", body)
    return body.strip()


def word_count(text):
    """Word count in the benchmark's sense (whitespace tokens)."""
    return len(re.findall(r"\S+", text or ""))


def fmt_seconds(s):
    s = int(round(s))
    return "%d:%02d" % (s // 60, s % 60)
