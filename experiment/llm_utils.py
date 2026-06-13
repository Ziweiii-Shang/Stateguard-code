"""LLM API utilities."""
import json
import time
import urllib.request
import urllib.error
from openai import OpenAI
from config import API_BASE, API_KEY, MODELHUB_URL, MODELHUB_AK, MODELHUB_MODELS

_client = OpenAI(base_url=API_BASE, api_key=API_KEY)


def _chat_modelhub(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    """Call the optional secondary provider (OpenAI-style crawl API). Content
    must be a list of typed parts and auth is an `ak` query parameter."""
    converted = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": m["role"], "content": content})

    payload = {"stream": False, "model": model, "max_tokens": max(max_tokens, 256),
               "temperature": temperature, "messages": converted}

    for attempt in range(3):
        req = urllib.request.Request(
            MODELHUB_URL + "?ak=" + MODELHUB_AK,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Request-Id": "stateguard-exp"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read().decode("utf-8", "replace"))
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):  # typed parts in response
                content = "".join(p.get("text", "") for p in content)
            if not (content or "").strip():
                raise ValueError("Empty content in response")
            return content.strip()
        except Exception as e:
            err = str(e).lower()
            if "temperature" in err and "temperature" in payload:
                payload.pop("temperature")
                continue
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"[WARN] modelhub chat failed for {model}: {e}")
                return ""


def _fix_messages_for_claude(messages: list) -> list:
    """Fix message format for Claude models via OpenAI proxy.

    Claude doesn't support multiple system messages or system messages
    after user/assistant turns. Convert mid-conversation system messages
    to user messages with [System] prefix. Also filter empty messages
    and ensure proper alternation.
    """
    if not messages:
        return [{"role": "user", "content": "Hello"}]
    # Filter out empty content messages
    messages = [m for m in messages if m.get("content", "").strip()]
    if not messages:
        return [{"role": "user", "content": "Hello"}]
    fixed = []
    seen_non_system = False
    for msg in messages:
        if msg["role"] == "system" and seen_non_system:
            # Convert mid-conversation system message to user message
            fixed.append({"role": "user", "content": f"[System instruction]: {msg['content']}"})
        else:
            fixed.append(msg.copy())
        if msg["role"] in ("user", "assistant"):
            seen_non_system = True
    # Ensure no consecutive same-role messages (merge them)
    merged = []
    for msg in fixed:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg)
    # Claude requires at least one non-system message
    non_system = [m for m in merged if m["role"] != "system"]
    if not non_system:
        merged.append({"role": "user", "content": "Please proceed."})
    return merged


# Per-model parameter capabilities, detected at runtime and cached.
# Newer models (gpt-5/o-series) reject `max_tokens` (require `max_completion_tokens`)
# and reject non-default `temperature`.
_MODEL_CAPS = {}


# Models matching these hints burn completion budget on hidden reasoning
# (reasoning_content channel); seed them with a high token floor so short
# answers (judge scores, classifications) don't come back truncated/empty.
_REASONING_HINTS = ("kimi", "reasoner", "thinking", "-r1")


def _caps(model: str) -> dict:
    if model not in _MODEL_CAPS:
        floor = 2048 if any(h in model.lower() for h in _REASONING_HINTS) else 0
        _MODEL_CAPS[model] = {
            "use_max_completion_tokens": False,
            "no_temperature": False,
            # Raised when a model returns empty content because hidden reasoning
            # consumed the completion budget (e.g. gemini thinking models).
            "token_floor": floor,
        }
    return _MODEL_CAPS[model]


def chat(model: str, messages: list, max_tokens: int = 512, temperature: float = 0.3) -> str:
    """Send a chat completion request. Retries on failure and auto-adapts
    request parameters for models that reject max_tokens/temperature.
    Models listed in config.MODELHUB_MODELS are routed to the secondary provider."""
    if model in MODELHUB_MODELS:
        return _chat_modelhub(model, messages, max_tokens, temperature)

    # Fix messages for Claude models
    if "claude" in model.lower():
        messages = _fix_messages_for_claude(messages)

    caps = _caps(model)
    attempt = 0
    max_attempts = 3
    while attempt < max_attempts:
        budget = max(max_tokens, caps["token_floor"])
        kwargs = {"model": model, "messages": messages}
        if caps["use_max_completion_tokens"]:
            # Reasoning models consume completion budget on hidden reasoning;
            # keep enough headroom so short answers don't come back empty.
            kwargs["max_completion_tokens"] = max(budget, 256)
        else:
            kwargs["max_tokens"] = budget
        if not caps["no_temperature"]:
            kwargs["temperature"] = temperature

        try:
            resp = _client.chat.completions.create(**kwargs)
            if not resp.choices:
                raise ValueError("Empty choices in response")
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                raise ValueError("Empty content in response")
            return content.strip()
        except Exception as e:
            err = str(e).lower()
            # Balance exhaustion: abort the whole run immediately. Retrying
            # only floods checkpoints with empty responses / fallback scores
            # that later have to be scrubbed and re-paid.
            if "insufficient_balance" in err or ("402" in err and "balance" in err):
                import os
                print(f"\n[FATAL] API balance exhausted — aborting run to protect "
                      f"checkpoints. Recharge and restart; checkpoints will resume.\n({e})")
                os._exit(42)
            # Hidden-reasoning models (e.g. gemini thinking) silently return
            # empty content when reasoning eats the completion budget:
            # raise the per-model token floor once and retry.
            if "empty content" in err and caps["token_floor"] < 1024:
                caps["token_floor"] = 1024
                print(f"[INFO] {model}: empty content with small budget, raising completion budget to 1024")
                continue
            # Parameter-compatibility errors: adapt and retry without burning an attempt.
            if not caps["use_max_completion_tokens"] and "max_tokens" in err and (
                    "unsupported" in err or "max_completion_tokens" in err or "not supported" in err):
                caps["use_max_completion_tokens"] = True
                print(f"[INFO] {model}: switching to max_completion_tokens")
                continue
            if not caps["no_temperature"] and "temperature" in err and (
                    "unsupported" in err or "not supported" in err
                    or "does not support" in err or "deprecated" in err):
                caps["no_temperature"] = True
                print(f"[INFO] {model}: dropping temperature parameter")
                continue
            attempt += 1
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                print(f"[WARN] chat failed for {model}: {e}")
                return ""


def verify_model(model: str) -> bool:
    """Preflight check: verify a model responds via the proxy. Fail fast
    instead of letting judge calls silently degrade mid-run."""
    resp = chat(model, [{"role": "user", "content": "Reply with the single word: OK"}],
                max_tokens=10, temperature=0.0)
    ok = bool(resp)
    status = "OK" if ok else "FAILED"
    print(f"[Preflight] {model}: {status}" + (f" (reply: {resp[:40]!r})" if ok else ""))
    return ok


def chat_json(model: str, messages: list, max_tokens: int = 1024, temperature: float = 0.1) -> dict:
    """Chat expecting JSON output. Parses response."""
    raw = chat(model, messages, max_tokens, temperature)
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return {}
