"""A/B harness: ERNIE-4.5-VL weight loading, new loader vs old loader.

Runs a fixed set of greedy prompts (text-only + image) and dumps the outputs to
JSON so the two code revisions can be diffed byte-for-byte. The image prompt is
what exercises the vision-expert re-indexing in `Ernie4_5_VLMoeModel._preprocess`.
"""

import base64
import io
import json
import sys

from PIL import Image, ImageDraw

from vllm import LLM, SamplingParams

MODEL = "baidu/ERNIE-4.5-VL-28B-A3B-PT"


def make_image() -> str:
    """Deterministic test image: red square + blue circle on white."""
    img = Image.new("RGB", (448, 448), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 200, 200], fill=(220, 30, 30))
    d.ellipse([250, 250, 390, 390], fill=(30, 60, 210))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


TEXT_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Explain what a mixture-of-experts model is, in one sentence.",
    "List the first five prime numbers.",
]

IMAGE_QUESTIONS = [
    "What shapes and colors do you see in this image?",
    "How many distinct shapes are in the image?",
]


def main(out_path: str) -> None:
    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=64, seed=0)

    results = {}

    text_msgs = [[{"role": "user", "content": p}] for p in TEXT_PROMPTS]
    for prompt, out in zip(TEXT_PROMPTS, llm.chat(text_msgs, sampling)):
        results[f"text::{prompt}"] = out.outputs[0].text

    b64 = make_image()
    img_msgs = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": q},
                ],
            }
        ]
        for q in IMAGE_QUESTIONS
    ]
    for question, out in zip(IMAGE_QUESTIONS, llm.chat(img_msgs, sampling)):
        results[f"image::{question}"] = out.outputs[0].text

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"===== RESULTS ({out_path}) =====", flush=True)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
