# --------------------------------------------------
# Load configuration, tokenizer, and model
# --------------------------------------------------

try:
    print(f"Loading configuration for {MODEL_ID}...", flush=True)

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print("Config class:", type(config).__name__, flush=True)

    original_rope_scaling = getattr(config, "rope_scaling", None)
    print("Original RoPE scaling:", original_rope_scaling, flush=True)

    # AyurParam's custom model code accesses rope_scaling["type"].
    # Newer Transformers configurations may provide "rope_type" instead.
    if original_rope_scaling is None:
        rope_scaling = {
            "type": "default",
            "rope_type": "default",
        }
    else:
        rope_scaling = dict(original_rope_scaling)
        if "type" not in rope_scaling:
            rope_scaling["type"] = rope_scaling.get("rope_type", "default")

    config.rope_scaling = rope_scaling
    print("Effective RoPE scaling:", config.rope_scaling, flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32

    print("Selected dtype:", dtype, flush=True)
    print("Loading model weights and initializing model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )

    model.eval()
    print("=== AyurParam loaded successfully ===", flush=True)

except Exception:
    print("=== AYURPARAM STARTUP FAILED ===", flush=True)
    traceback.print_exc()
    raise


# --------------------------------------------------
# RunPod Serverless handler
# --------------------------------------------------

def handler(event):
    data = event.get("input", {})
    prompt_text = data.get("prompt", "").strip()

    if not prompt_text:
        raise ValueError("input.prompt is required")

    max_tokens = int(
        data.get("max_new_tokens", data.get("max_tokens", 128))
    )
    max_tokens = max(1, min(max_tokens, 512))

    prompt = f"<user> {prompt_text} <assistant>"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return {
        "model": MODEL_ID,
        "answer": answer,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
