import sys, time, uuid
import runpod, torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- startup diagnostics: fail fast with a readable message ---
print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
      f"transformers {transformers.__version__} | CUDA {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
if not transformers.__version__.startswith("4."):
    raise RuntimeError(f"AyurParam remote code needs transformers 4.x, found {transformers.__version__}")

MODEL = "/models/AyurParam"
MODEL_ID = "bharatgenai/AyurParam"
STOP_IDS = [256005, 3, 256001]   # </actual_response>, </s>, <user>
MAX_CTX = 2048

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()


def generate(prompt, max_tokens=300, temperature=0.6, top_p=0.95):
    enc = tok(prompt, return_tensors="pt").to("cuda")
    n_in = enc["input_ids"].shape[1]
    if n_in >= MAX_CTX - 16:
        raise ValueError(f"prompt too long: {n_in} tokens (limit ~{MAX_CTX - 16})")
    max_tokens = max(1, min(int(max_tokens), MAX_CTX - n_in, 1024))
    temperature = float(temperature)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_k=50,
            top_p=float(top_p),
            use_cache=True,
            eos_token_id=STOP_IDS,
            pad_token_id=3,
        )
    ids = out[0][n_in:]
    finish = "stop" if len(ids) and ids[-1].item() in STOP_IDS else "length"
    return tok.decode(ids, skip_special_tokens=True).strip(), n_in, len(ids), finish


def messages_to_prompt(messages):
    def text(c):  # OpenAI content can be a string or a list of parts
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c or ""

    system = " ".join(text(m["content"]) for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] in ("user", "assistant")][-9:]  # ~5 turns
    parts = [f"<system_prompt> {system}"] if system else []
    for m in convo:
        tag = "<user> " if m["role"] == "user" else "<assistant> "
        parts.append(tag + text(m["content"]))
    parts.append("<assistant>")
    return " ".join(parts)


def openai_error(msg, typ="invalid_request_error"):
    return {"error": {"message": msg, "type": typ}}


def openai_handler(route, body):
    if route.endswith("/models"):
        return {"object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "bharatgenai"}]}
    if body.get("stream"):
        return openai_error("stream=true is not supported; send stream=false")

    params = dict(
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 300,
        temperature=body.get("temperature", 0.6),
        top_p=body.get("top_p", 0.95),
    )
    now, rid = int(time.time()), uuid.uuid4().hex[:24]

    try:
        if route.endswith("/chat/completions"):
            if not body.get("messages"):
                return openai_error("messages is required")
            text, n_in, n_out, finish = generate(messages_to_prompt(body["messages"]), **params)
            return {
                "id": f"chatcmpl-{rid}", "object": "chat.completion", "created": now, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                          "total_tokens": n_in + n_out},
            }
        if route.endswith("/completions"):
            if not body.get("prompt"):
                return openai_error("prompt is required")
            text, n_in, n_out, finish = generate(body["prompt"], **params)
            return {
                "id": f"cmpl-{rid}", "object": "text_completion", "created": now, "model": MODEL_ID,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                          "total_tokens": n_in + n_out},
            }
    except ValueError as e:
        return openai_error(str(e))

    return openai_error(f"unsupported route {route}")


def handler(job):
    inp = job["input"]

    # OpenAI-compatible route: /v2/<id>/openai/v1/...
    if "openai_route" in inp:
        return openai_handler(inp["openai_route"], inp.get("openai_input", {}))

    # Native RunPod format: {"input": {"prompt": ..., "context": ..., "history": [[u, a], ...]}}
    if not inp.get("prompt"):
        return {"error": "prompt is required", "received_keys": list(inp.keys())}
    ctx = f"<context> {inp['context']} " if inp.get("context") else ""
    hist = "".join(f"<user> {u} <assistant> {a} " for u, a in inp.get("history", [])[-4:])
    try:
        text, _, n_out, _ = generate(
            f"{ctx}{hist}<user> {inp['prompt']} <assistant>",
            inp.get("max_new_tokens", 300),
            inp.get("temperature", 0.6),
        )
    except ValueError as e:
        return {"error": str(e)}
    return {"output": text, "tokens": n_out}


runpod.serverless.start({"handler": handler})
