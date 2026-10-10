import os
import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "bharatgenai/AyurParam"

os.environ.setdefault(
    "HF_HOME", "/workspace/.cache/huggingface"
)

print(f"Loading {MODEL_ID}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

if torch.cuda.is_available():
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
else:
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto"
)
model.eval()

print("AyurParam loaded successfully.", flush=True)


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
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return {
        "model": MODEL_ID,
        "answer": answer
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})