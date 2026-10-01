"""Precompute the frozen VLM's hidden states for every frame of a dataset: images + instruction.

The per-frame counterpart of cache_text_context.py. PointAct's action expert reads the
Qwen2.5-VL `last_hidden_state` only through cross-attention, and with the VLM frozen that
tensor is a fixed function of (camera images, instruction). So it can be computed once per
frame instead of once per training step, and a `context_source=frame_cache` run trains with
image conditioning at the cost of a VLM-free run. Keys are "{episode}-{frame}", the same
converted-episode indexing as the point LMDB; values are fp16 (L, hidden_size) arrays.

What makes the cache valid is that it holds exactly what inference computes live, because at
eval there is no cache (the simulator renders new frames): `run_server.py` re-attaches the
same frozen VLM (`attach_frozen_vlm`). So this script does not reimplement the VLM input. It
calls the eval processor's own `vlm_messages` / `vlm_inputs` and the model's own
`encode_vlm_context`, which is what `select_action` -> `sample_actions` run. The only
difference left is where the pixels come from: mp4-decoded here, rendered at eval.

Recipe choices that are part of the arm, recorded in <out>/meta.json:
  - chat template scripts/chat_template.json and the pipeline's image min/max pixels, i.e.
    what training writes into the checkpoint's processor, and therefore what eval uses;
  - no image augmentation: one fixed context per frame is what a cache means.

Example (GPU node; `module load ffmpeg` first on Jean Zay for torchcodec):
    python data_prep/cache_frame_context.py \
        --dataset-dir $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
        --vlm-path $POINTACT_VLM_PATH
    # then spot-check the stored entries against a fresh batch-1 forward:
    python data_prep/cache_frame_context.py ... --verify 64
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

msgpack_numpy.patch()

DEFAULT_VIDEO_KEYS = (
    "observation.images.left_image",
    "observation.images.right_image",
    "observation.images.wrist_image",
)


def load_processor(vlm_path: str, chat_template: Path, min_pixels: int, max_pixels: int):
    """The processor a training run saves into its checkpoint -- and eval therefore loads.

    Mirrors scripts/train.py: load_processor (padding_side="right"),
    smart_tokenizer_and_embedding_resize (the added special tokens), and the two parts of
    configure_processor that touch the VLM input: the chat template and min/max pixels.
    """
    from pointact.model.vla_pointact.processing_vla_pointact import VLAEncDec3DProcessor
    from pointact.train.train_utils import smart_tokenizer_and_embedding_resize

    processor = VLAEncDec3DProcessor.from_pretrained(vlm_path, padding_side="right")
    smart_tokenizer_and_embedding_resize(processor)
    template = json.loads(chat_template.read_text())["chat_template"]
    processor.chat_template = processor.tokenizer.chat_template = template
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    return processor


def read_episodes(dataset_dir: Path) -> list[dict]:
    episodes = []
    with (dataset_dir / "meta" / "episodes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                episodes.append(json.loads(line))
    for ep in episodes:
        # One instruction per episode, with no "<br>" alternatives, is what makes a per-frame
        # cache exact: select_task_text would otherwise pick a string at random per sample.
        if len(ep["tasks"]) != 1 or "<br>" in ep["tasks"][0]:
            raise ValueError(
                f"episode {ep['episode_index']} has instructions {ep['tasks']!r}; a per-frame "
                "context cache needs exactly one fixed instruction per episode"
            )
    return episodes


class EpisodeFrames(Dataset):
    """One item per episode: its frame indices, instruction, and every frame of every view.

    Decoded at the parquet's own timestamps, as LeRobot does, so a frame here is the frame
    the dataloader would have produced for that index.
    """

    def __init__(self, dataset_dir: Path, episodes: list[dict], video_keys: list[str]):
        self.dataset_dir = dataset_dir
        self.episodes = episodes
        self.video_keys = video_keys
        self.info = json.loads((dataset_dir / "meta" / "info.json").read_text())

    def __len__(self):
        return len(self.episodes)

    def _path(self, template_key: str, ep_idx: int, **kwargs) -> Path:
        chunk = ep_idx // self.info["chunks_size"]
        return self.dataset_dir / self.info[template_key].format(
            episode_chunk=chunk, episode_index=ep_idx, **kwargs
        )

    def __getitem__(self, i):
        from torchcodec.decoders import VideoDecoder

        ep = self.episodes[i]
        ep_idx = ep["episode_index"]
        table = pd.read_parquet(self._path("data_path", ep_idx), columns=["frame_index", "timestamp"])
        frame_indices = table["frame_index"].to_numpy()
        timestamps = table["timestamp"].to_numpy().tolist()

        views = {}
        for key in self.video_keys:
            decoder = VideoDecoder(str(self._path("video_path", ep_idx, video_key=key)))
            frames = decoder.get_frames_played_at(seconds=timestamps).data  # (N, C, H, W) uint8
            views[key] = frames.permute(0, 2, 3, 1).numpy()
        return ep_idx, ep["tasks"][0], frame_indices, views


def frame_messages(processor, views: dict, video_keys: list[str], task: str, rows) -> list:
    # np.asarray: through a DataLoader the arrays arrive as torch tensors (default_convert).
    return [
        processor.vlm_messages(
            [Image.fromarray(np.asarray(views[key][r])) for key in video_keys], task
        )
        for r in rows
    ]


@torch.no_grad()
def context_batch(model, processor, batch_messages: list, device) -> list[np.ndarray]:
    """Per-frame (L_i, hidden) fp16 context, via the same calls select_action makes.

    Everything up to the LM runs ONE FRAME AT A TIME, exactly eval's batch-1 calls: the
    preprocessing (`vlm_inputs`) and the vision tower (`embed_vlm_inputs`). Only the LM
    forward is batched. Measured 2026-09-30 on OpenDrawer: a batched vision tower moves
    image features enough that ~16% of context tokens fall below cosine 0.99 against the
    batch-1 context, while a batched LM over batch-1 embeddings matches it (mean 0.9996).
    Frames of equal token length are stacked for the LM; any other length runs alone.
    """
    from pointact.model.vla_pointact.modeling_vla_pointact import (
        embed_vlm_inputs,
        encode_vlm_context,
    )

    per_frame = [processor.vlm_inputs([messages], device=device) for messages in batch_messages]
    embeds = [
        embed_vlm_inputs(model, p["input_ids"], p["pixel_values"], p["image_grid_thw"])
        for p in per_frame
    ]
    groups: dict[int, list[int]] = {}
    for i, inputs in enumerate(per_frame):
        groups.setdefault(inputs["input_ids"].shape[1], []).append(i)

    out: list[np.ndarray | None] = [None] * len(per_frame)
    for idx in groups.values():
        inputs = {
            key: torch.cat([per_frame[i][key] for i in idx], dim=0)
            for key in ("input_ids", "attention_mask", "image_grid_thw")
        }
        outputs = encode_vlm_context(
            model,
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"].bool(),  # select_action passes it as bool
            image_grid_thw=inputs["image_grid_thw"],  # for the 3D position ids
            inputs_embeds=torch.cat([embeds[i] for i in idx], dim=0),
            use_cache=False,
        )
        hidden = outputs.last_hidden_state.to(torch.float16).cpu().numpy()
        for row, i in enumerate(idx):
            out[i] = hidden[row]
    return out


def load_vlm(vlm_path: str, attn: str, device):
    from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_path, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval().requires_grad_(False)


def build(args, processor, model, episodes, video_keys, out_dir: Path) -> dict:
    env = lmdb.open(str(out_dir), map_size=args.map_size_gb << 30, readahead=False, meminit=False)

    with env.begin() as txn:
        todo = [ep for ep in episodes
                if txn.get(f"{ep['episode_index']}-{ep['length'] - 1}".encode()) is None]
    print(f"{len(episodes) - len(todo)} episode(s) already cached, {len(todo)} to build")

    loader = DataLoader(
        EpisodeFrames(args.dataset_dir, todo, video_keys),
        batch_size=None, num_workers=args.num_workers, prefetch_factor=2 if args.num_workers else None,
    )
    lengths, n_frames, t0 = [], 0, time.time()
    for k, (ep_idx, task, frame_indices, views) in enumerate(loader):
        n = len(frame_indices)
        with env.begin(write=True) as txn:
            for start in range(0, n, args.batch_size):
                rows = range(start, min(start + args.batch_size, n))
                contexts = context_batch(
                    model, processor, frame_messages(processor, views, video_keys, task, rows),
                    args.device,
                )
                for r, ctx in zip(rows, contexts):
                    txn.put(f"{ep_idx}-{int(frame_indices[r])}".encode(), msgpack.packb(ctx))
                    lengths.append(len(ctx))
        n_frames += n
        if k % 10 == 0 or k == len(todo) - 1:
            rate = n_frames / (time.time() - t0)
            print(f"  [{k + 1}/{len(todo)}] episode {ep_idx}: {n} frames, {rate:.1f} frames/s, "
                  f"tokens/frame {min(lengths)}-{max(lengths)}", flush=True)
    env.close()
    return {"frames_built": n_frames, "tokens_per_frame": sorted(set(lengths))}


def verify(args, processor, model, episodes, video_keys, out_dir: Path) -> None:
    """Recompute random stored frames one at a time (eval's batch size) and compare."""
    rng = random.Random(0)
    env = lmdb.open(str(out_dir), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        built = [ep for ep in episodes
                 if txn.get(f"{ep['episode_index']}-{ep['length'] - 1}".encode()) is not None]
    if not built:
        raise SystemExit(f"verify: nothing built yet in {out_dir}")
    picked = rng.sample(built, min(args.verify, len(built)))
    frames = EpisodeFrames(args.dataset_dir, picked, video_keys)
    per_episode = -(-args.verify // len(picked))  # N frames in total, spread over the episodes
    samples = []
    for i in range(len(picked)):
        ep_idx, task, frame_indices, views = frames[i]
        for r in rng.sample(range(len(frame_indices)), min(per_episode, len(frame_indices))):
            samples.append((ep_idx, int(frame_indices[r]),
                            frame_messages(processor, views, video_keys, task, [r])))

    frame_means, frame_bad = [], []
    with env.begin() as txn:
        for ep_idx, frame, messages in samples:
            stored = txn.get(f"{ep_idx}-{frame}".encode())
            if stored is None:
                raise KeyError(f"{ep_idx}-{frame} missing from {out_dir}")
            stored = msgpack.unpackb(stored).astype(np.float32)
            fresh = context_batch(model, processor, messages, args.device)[0].astype(np.float32)
            if fresh.shape != stored.shape:
                raise ValueError(f"shape {fresh.shape} != stored {stored.shape} at {ep_idx}-{frame}")
            num = (fresh * stored).sum(-1)
            den = np.linalg.norm(fresh, axis=-1) * np.linalg.norm(stored, axis=-1) + 1e-6
            cos = num / den
            frame_means.append(float(cos.mean()))
            frame_bad.append(float((cos < 0.99).mean()))
    env.close()
    # Judged per frame, not by the single worst token: bf16 alone puts the odd token near 0.91
    # between two correct forwards (batch 2 vs 1), while a frame given another frame's image
    # features shows up as a mean ~0.98 with 10-30% of its tokens below 0.99. The token-share
    # bar is 3%, not 1%: the full OpenDrawer build's correct cache measured a worst frame of
    # 1.1% over 64 frames (mean 0.9994) -- bf16 noise in the batched LM, an order of magnitude
    # below the failure it exists to catch.
    print(f"verify: {len(frame_means)} frames, worst frame mean cosine {min(frame_means):.5f}, "
          f"worst share of tokens < 0.99: {max(frame_bad):.3%}")
    if min(frame_means) < 0.999 or max(frame_bad) > 0.03:
        raise SystemExit("verify FAILED: stored context does not match a fresh batch-1 forward")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--vlm-path", required=True, help="Local base Qwen2.5-VL checkpoint.")
    parser.add_argument("--out", default="frame_context/qwen2.5-vl-3b-3views",
                        help="LMDB directory, relative to --dataset-dir. Mirror it in the data "
                             "config's `frame_context_lmdb`.")
    parser.add_argument("--video-keys", nargs="+", default=list(DEFAULT_VIDEO_KEYS),
                        help="Views, in prompt order. Must equal the run's "
                             "select_video_keys_for_vlm, which is what eval sends.")
    parser.add_argument("--chat-template", type=Path, default=Path("scripts/chat_template.json"))
    # pipeline_config.py defaults, which configure_processor writes into the processor.
    parser.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=350 * 28 * 28)
    parser.add_argument("--attn", default="flash_attention_2",
                        help="Must match the training config's attn_implementation: eval attaches "
                             "the VLM with the checkpoint's.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--map-size-gb", type=int, default=400)
    parser.add_argument("--verify", type=int, default=0,
                        help="Only spot-check N random stored frames against a fresh forward.")
    parser.add_argument("--processor-dir", default="",
                        help="With --verify: load the processor a trained checkpoint saved (what "
                             "eval uses) instead of rebuilding it, closing the last gap between "
                             "the cache and the live path.")
    parser.add_argument("--limit-episodes", type=int, default=0,
                        help="Smoke test: only the first N episodes (resume completes the rest).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = args.dataset_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes(args.dataset_dir)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    if args.processor_dir:
        from pointact.model.vla_pointact.processing_vla_pointact import VLAEncDec3DProcessor

        processor = VLAEncDec3DProcessor.from_pretrained(args.processor_dir)
        print(f"processor from {args.processor_dir}")
    else:
        processor = load_processor(args.vlm_path, args.chat_template, args.min_pixels, args.max_pixels)
    model = load_vlm(args.vlm_path, args.attn, args.device)

    if args.verify:
        verify(args, processor, model, episodes, args.video_keys, out_dir)
        return

    stats = build(args, processor, model, episodes, args.video_keys, out_dir)
    meta = {
        "vlm_path": args.vlm_path,
        "video_keys": args.video_keys,
        "chat_template_sha256": hashlib.sha256(args.chat_template.read_bytes()).hexdigest(),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "attn_implementation": args.attn,
        "dtype": "float16",
        "image_augmentation": False,
        "episodes": len(episodes),
        **stats,
    }
    meta_path = out_dir.parent / f"{out_dir.name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir} and {meta_path}")


if __name__ == "__main__":
    main()
