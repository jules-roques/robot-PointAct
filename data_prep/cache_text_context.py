"""Precompute the frozen VLM's text-only hidden states for every instruction in a dataset.

Background: PointAct feeds the Qwen2.5-VL `last_hidden_state` into the point-action expert as
a ragged context tensor that the action tokens cross-attend to. With `--ptv3_apply_point_ca
False` (every RoboCasa run) the point *features* never see it, so the VLM's entire
contribution is that context. Running a 3B forward each step to produce it is wasteful when
the instruction set is tiny -- OpenDrawer has exactly two strings.

This script runs the text-only forward once per unique instruction and stores the result. The
tensor has the same shape and meaning as before, so `ctx_proj` and every cross-attention block
downstream are untouched; only the source changes. Point it at a LeRobot dataset directory and
set `text_context_file` in the data config plus `--context_source text_cache` on the train run.

What this deliberately is NOT: the text hidden states of a forward that also saw images. The
image tokens are gone, so the text positions attend over text alone. That difference is the
ablation -- language conditioning is kept, visual conditioning is removed.

Example:
    python data_prep/cache_text_context.py \
        --dataset-dir $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
        --vlm-path $SCRATCH/models/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def read_instructions(dataset_dir: Path) -> list[str]:
    """Every distinct instruction string in the dataset, from meta/tasks.jsonl."""
    tasks_file = dataset_dir / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"no meta/tasks.jsonl under {dataset_dir}")

    instructions = []
    with tasks_file.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line).get("task")
            if task and task not in instructions:
                instructions.append(task)

    if not instructions:
        raise ValueError(f"meta/tasks.jsonl in {dataset_dir} lists no tasks")
    return instructions


def build_cache(
    instructions: list[str],
    vlm_path: str,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    processor = AutoProcessor.from_pretrained(vlm_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path, dtype=dtype, attn_implementation="sdpa"
    )
    model.to(device).eval()

    cache = {}
    for instruction in instructions:
        # Match the prompt the training path builds for a robot sample, minus the image and
        # state placeholder tokens: those correspond to modalities this variant does not have.
        text = processor.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        # Stored on CPU in fp16: a handful of instructions x ~30 tokens x 2048 is well under
        # a megabyte, so the whole cache is loaded into memory by every dataloader worker.
        cache[instruction] = outputs.last_hidden_state[0].to("cpu", torch.float16)
        print(f"  {instruction!r} -> {tuple(cache[instruction].shape)}")

    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="LeRobot dataset directory (the one holding meta/tasks.jsonl).",
    )
    parser.add_argument("--vlm-path", required=True, help="Local Qwen2.5-VL checkpoint.")
    parser.add_argument(
        "--out",
        default="text_context/qwen2.5-vl-3b.pt",
        help="Output path, relative to --dataset-dir. Mirror this in the data config's "
        "`text_context_file`.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    instructions = read_instructions(args.dataset_dir)
    print(f"{len(instructions)} unique instruction(s) in {args.dataset_dir.name}")

    cache = build_cache(instructions, args.vlm_path, args.device, torch.bfloat16)

    out_path = args.dataset_dir / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
