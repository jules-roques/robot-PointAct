"""Serve MolmoPoint-8B over the same ZeroMQ transport the policy server uses.

Why a third process. Evaluating a molmo-anchored checkpoint needs three things running at
once that cannot share one Python environment: the simulator (``envs/robocasa365``, MuJoCo +
EGL), the policy (root ``.venv``, transformers 5.x), and MolmoPoint (``envs/molmo``,
transformers 4.57.1, pinned because the checkpoint's remote code targets it). The pointer is
therefore a server the sim client calls, exactly as it already calls the policy server.

    uv run --project envs/molmo --no-sync python scripts/run_molmo_server.py \
        --args.model_dir $SCRATCH/models/MolmoPoint-8B --args.port 5556

Endpoint ``point``: {"images": [HxWx3 uint8, ...], "prompt": str} -> a list of
{"image_num", "object_id", "x", "y"} dicts, one per detection, in pixel coordinates of the
image each landed in. All views ride in one request, so a frame costs one forward per query
rather than one per (query, camera) -- see MolmoPointer.point for why requests themselves
cannot be batched on this checkpoint.

Memory: ~18 GB in bf16, sharing the eval GPU with the policy (~10 GB) and MuJoCo. Comfortable
on the 80 GB A100 the evals already request.
"""

import dataclasses

import numpy as np
import tyro

from pointact.roi_sampling.molmo_pointer import MolmoPointer
from pointact.utils.server_client import PolicyServer


@dataclasses.dataclass
class MolmoServerArgs:
    model_dir: str = ""
    host: str = "127.0.0.1"
    port: int = 5556
    max_new_tokens: int = 128
    # Probe the checkpoint's point-tuple field order at startup. Cheap (one forward) against
    # the cost of getting it wrong: [object_id, image_num, ...] and [image_num, object_id, ...]
    # both parse, and picking wrong attributes a detection to a camera it did not come from,
    # which lifts to a plausible but wrong 3D anchor.
    verify_point_order: bool = True


class MolmoService:
    def __init__(self, args: MolmoServerArgs):
        if not args.model_dir:
            raise ValueError("--args.model_dir is required")
        self.pointer = MolmoPointer(args.model_dir, max_new_tokens=args.max_new_tokens)
        if args.verify_point_order:
            print(f"[molmo] point field order: {self.pointer.verify_point_order()}",
                  flush=True)
        self.n_requests = 0
        print(f"[molmo] ready on {args.host}:{args.port}", flush=True)

    def point(self, images, prompt):
        """One pointing request. Errors return an empty list, never take the eval down.

        A pointer that fails is a frame that falls back to uniform sampling -- the same thing
        that happens when the model declines to point, and recorded as such by the caller's
        hit-rate counter. A pointer that raises would end a 100-trial rollout at trial 3.
        """
        self.n_requests += 1
        try:
            ims = [np.asarray(im, dtype=np.uint8) for im in images]
            dets = self.pointer.point([ims], [prompt])[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[molmo] request {self.n_requests} failed: {type(exc).__name__}: {exc}",
                  flush=True)
            return []
        return [{"image_num": d.image_num, "object_id": d.object_id,
                 "x": float(d.x), "y": float(d.y)} for d in dets]

    # The transport registers these two unconditionally (PolicyServer.__init__), so they have
    # to exist even though pointing is stateless.
    def get_action(self, batch, options=None):
        raise NotImplementedError("this is a pointing server, not a policy server")

    def reset(self, options=None):
        return {"status": "ok"}


def main(args: MolmoServerArgs) -> None:
    service = MolmoService(args)
    server = PolicyServer(service, host=args.host, port=args.port)
    # PolicyServer.run expands the request's "data" dict into keyword arguments, so the
    # handler's signature is the wire format: {"images": [...], "prompt": "..."}.
    server.register_endpoint("point", service.point)
    server.run()


if __name__ == "__main__":
    tyro.cli(main)
