"""MolmoMotion-4B-H3-F30 wrapper: a gripper query point in, its forecast trajectory out.

Where :mod:`pointact.roi_sampling.molmo_pointer` asks a *detector* "where is the object the
instruction names", this asks a *forecaster* "where is this point about to go". The query
point is the end-effector, which we already know exactly from proprio -- so unlike the
MolmoPoint arm there is no detection step that can point at the wrong object.

Three properties of the checkpoint drive the design here, all confirmed against
``src/molmo_motion/processor.py`` rather than the model card:

* **It is single-camera.** ``history_frames`` is documented as "list of PIL.Image, length
  must equal self.config.history_size. Ordered earliest -> query (t_0)" -- that list axis is
  *time*, not view, and no argument anywhere accepts several views at one timestep. This is
  the opposite of MolmoPoint, where left/right/wrist rode in a single forward and each
  returned point named its image. Multi-view here costs one forward per view.
* **Everything is camera-frame-at-t0.** The 3D history goes in as camera-frame metres and
  ``future_3d`` comes back as absolute camera-frame metres, so both ends convert through
  :func:`pointact.roi_sampling.geometry.base_to_camera` / ``camera_to_base``.
* **The query point count is adjustable, despite looking fixed.** ``config.num_points``
  defaults to 8 and the processor hard-validates every input shape against it, which reads
  like an architectural constraint. It is not: ``num_points`` never sizes a weight -- it only
  validates inputs, fills the "{P} points" slot in the prompt, and shapes the parsed output
  (the package's own ``eval/full_rollout.py`` exposes it as ``--num_points``). Since this is
  a decode-bound model, asking for 1 point instead of 8 cuts the generated tokens and the
  wall clock with them. Pass ``num_points=1`` to spend the budget on the one query we have.
  Replicating a single query into 8 slots also works and is what ``reduce`` collapses, but it
  pays ~8x for seven copies of the same answer.
* **A parse failure returns zeros, not an error.** If the model emits no valid ``<tracks>``
  block, ``predict_trajectory`` returns ``zeros((P, F, 3))`` and skips the step that adds the
  anchor back -- so the "prediction" is the camera's optical centre, which lifts to a
  perfectly plausible 3D point somewhere in the room. It has to be detected explicitly, and
  it is not hypothetical: the token budget is ``160 * future_horizon`` (4,800 at F=30) while
  ``max_sequence_length`` is 2,560, so a long horizon at P=8 can be truncated mid-block.
* **The base tokenizer is a separate hub repo.** ``config.yaml`` names it as
  ``llm.tokenizer.identifier: Qwen/Qwen3-4B-Instruct-2507`` and the processor resolves it
  through ``AutoTokenizer`` *by id*, so the checkpoint shipping its own ``tokenizer.json``
  does not make it loadable offline. It must be in the hub cache before any compute-node job
  starts (``download_molmo_motion.slurm`` does this and proves it with an offline load).
* **History frames carry no timestamps.** The processor builds them as
  ``np.arange(H) * 1.0``, so the model has no way to be told our 20 fps differs from the
  15 fps it was trained at. The self-consistent reading is that output step ``f`` lands one
  history-spacing after step ``f - 1``: feed history at stride ``s`` frames and the forecast
  covers ``F * s`` frames. That is an assumption, not a documented guarantee, which is why
  the gate sweeps ``s`` and measures which value actually tracks ground truth instead of
  picking one here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: The checkpoint's native history length and horizon. H3-F30 is the AR variant recommended
#: for video; the H1-F32 sibling sees a single frame and cannot infer velocity from images.
HISTORY_SIZE = 3
FUTURE_HORIZON = 30


class MolmoMotionForecaster:
    """Frozen MolmoMotion, one forecast per forward.

    The model is detached from PointAct exactly as MolmoPoint was: it only decides *where*
    the point budget goes, and nothing it emits reaches the policy as an input.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        history_size: int = HISTORY_SIZE,
        future_horizon: int = FUTURE_HORIZON,
        num_points: int | None = None,
    ):
        import torch
        from molmo_motion import MolmoMotion, MolmoMotionProcessor

        self.torch = torch
        self.device = device
        self.history_size = int(history_size)
        self.future_horizon = int(future_horizon)

        # A local directory, never a hub id: compute nodes have no internet and the failure
        # mode without local weights is a hang, not an error (docs/clusters/jean-zay.md).
        self.processor = MolmoMotionProcessor.from_pretrained(model_dir)
        self.model = MolmoMotion.from_pretrained(model_dir)
        self.model._internal = self.model._internal.to(torch.bfloat16).to(device)

        cfg = self.model.config
        cfg_h = int(getattr(cfg, "history_size", self.history_size))
        if cfg_h != self.history_size:
            # The processor hard-validates this, so a mismatch is a crash later rather than a
            # silently truncated history. Fail here where the message names the cause.
            raise ValueError(
                f"checkpoint expects history_size={cfg_h}, got {self.history_size} -- "
                f"H1 and H3 checkpoints are not interchangeable")
        # `num_points` is a *config* field, not a weight shape: it is used only to validate
        # the input shapes, to fill the "{P} points" slot in the prompt, and to size the
        # parsed output array (the package's own `eval/full_rollout.py` exposes it as a CLI
        # flag). Lowering it to 1 therefore asks the model for one track instead of eight,
        # which cuts the generated tokens -- and this is a decode-bound model, so it cuts the
        # wall clock nearly in proportion. The checkpoint was trained at P=8, so P=1 is out of
        # distribution for the prompt; whether accuracy survives is an empirical question the
        # gate answers, not something to assume in either direction.
        self.num_points = int(num_points if num_points is not None
                              else getattr(cfg, "num_points", 8))
        if num_points is not None:
            # Both objects consult their own config: the processor to validate and build the
            # prompt, the model to shape the parsed trajectory. Setting one and not the other
            # gives a shape mismatch at parse time rather than a clean error.
            self.processor.config.num_points = self.num_points
            self.model.config.num_points = self.num_points
        self.max_sequence_length = int(getattr(cfg, "max_sequence_length", 2560))

    def forecast_point(
        self,
        history_frames: list[np.ndarray],
        point_2d_at_t0: np.ndarray,
        point_3d_history_cam: np.ndarray,
        action: str,
        future_horizon: int | None = None,
        reduce: str = "median",
    ) -> np.ndarray | None:
        """Forecast where **one** query point goes, in the camera frame at t0.

        The checkpoint's ``num_points`` is fixed (8 here) and the processor hard-validates
        the shapes against it, so a single query point cannot simply be passed as P=1. The
        point is replicated across all P slots instead: the model is still told about
        exactly one distinct location -- the anchor it predicts deltas from is point 0 at
        the last history frame, i.e. the gripper either way -- and the P outputs become P
        samples of the same question. ``reduce="median"`` collapses them, which costs
        nothing and takes the decoding noise out; ``reduce="first"`` gives the raw point-0
        track. Their disagreement is itself a useful diagnostic, since identical inputs
        should ideally give identical tracks.

        Args:
            history_frames: ``history_size`` RGB uint8 HxWx3 arrays, **earliest first**, the
                last being the query frame t0. All from one camera.
            point_2d_at_t0: (2,) pixel coords of the query point in the t0 frame.
            point_3d_history_cam: (H, 3) camera-frame-at-t0 metres, same time order as
                ``history_frames``.
            action: the episode's own instruction, verbatim.
            future_horizon: defaults to the checkpoint's native 30.
            reduce: ``"median"`` or ``"first"`` across the replicated slots.

        Returns:
            (F, 3) absolute camera-frame-at-t0 XYZ in metres, or **None** when the model
            emitted no parseable ``<tracks>`` block. That case must not be confused with a
            prediction: ``predict_trajectory`` returns exact zeros for it *without* adding
            the anchor back, which would otherwise read as a confident forecast at the
            camera's optical centre -- a plausible-looking 3D point metres from anything.
        """
        from PIL import Image

        torch = self.torch
        h = int(future_horizon or self.future_horizon)
        p = self.num_points

        if len(history_frames) != self.history_size:
            raise ValueError(
                f"need exactly {self.history_size} history frames, got {len(history_frames)}")
        uv = np.asarray(point_2d_at_t0, dtype=np.float32).reshape(2)
        xyz = np.asarray(point_3d_history_cam, dtype=np.float32).reshape(self.history_size, 3)

        pts2d = np.tile(uv[None, :], (p, 1))                      # (P, 2)
        pts3d = np.tile(xyz[:, None, :], (1, p, 1))               # (H, P, 3)

        frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in history_frames]
        inputs = self.processor(
            history_frames=frames,
            points_2d_at_t0=torch.from_numpy(pts2d),
            points_3d_history=torch.from_numpy(pts3d),
            action=action,
            future_horizon=h,
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        # The processor already emits `future_horizon` in its dict, so passing it again as a
        # keyword is a TypeError. Set it rather than trusting it through: `predict_trajectory`
        # uses this value for the token budget *and* to truncate the parsed tensor, while the
        # processor only writes it into the prompt string, and the two disagreeing would show
        # up as a quietly short trajectory rather than an error.
        inputs["future_horizon"] = h
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            out = self.model.predict_trajectory(**inputs)

        #: Raw ``<tracks>`` text of the most recent call. Kept for profiling and for
        #: debugging a parse failure, which is otherwise indistinguishable from a
        #: prediction of zeros.
        self.last_future_text = getattr(out, "future_text", "")

        traj = np.asarray(out.future_3d.float().cpu().numpy(), dtype=np.float64)  # (P, F, 3)
        if traj.size == 0 or not np.any(traj):
            logger.warning("MolmoMotion emitted no parseable tracks (text=%.120r)",
                           getattr(out, "future_text", ""))
            return None
        # F comes back as the shorter of the request and what was emitted before EOS, so the
        # caller cannot assume it got `future_horizon` steps.
        return np.median(traj, axis=0) if reduce == "median" else traj[0]


def history_frame_indices(t0: int, history_size: int, stride: int) -> list[int] | None:
    """Frame indices for a history window ending at ``t0``, earliest first.

    Returns ``None`` when the window runs off the start of the episode. Clamping instead
    would feed the model a stationary run-up and make the first ~0.5 s of every episode look
    like a stopped gripper, which is precisely where the forecast matters most.

    The spacing is uniform by construction. 20 fps cannot represent the model's native
    15 fps exactly (1/15 s is 1.333 frames), and rounding to ``[t0-3, t0-1, t0]`` would give
    a 2-then-1 frame spacing -- non-uniform input to a model that assumes uniform. Integer
    strides keep the assumption intact and let the gate measure which one transfers.
    """
    idx = [t0 - k * stride for k in range(history_size - 1, -1, -1)]
    return idx if idx[0] >= 0 else None


if __name__ == "__main__":
    # The model is not needed to check the indexing, and the indexing is where an off-by-one
    # would quietly misalign every forecast against its ground truth.
    assert history_frame_indices(10, 3, 1) == [8, 9, 10], "stride 1 window"
    assert history_frame_indices(10, 3, 2) == [6, 8, 10], "stride 2 window"
    assert history_frame_indices(10, 3, 4) == [2, 6, 10], "stride 4 window"

    # Ordering: earliest first, ending exactly at t0. The processor documents
    # "ordered earliest -> query (t_0)" and reversing it would feed the motion backwards.
    for s in (1, 2, 3):
        idx = history_frame_indices(9, 3, s)
        assert idx[-1] == 9, "the window must end at t0"
        assert idx == sorted(idx), "earliest first"
        assert len({b - a for a, b in zip(idx, idx[1:])}) == 1, "spacing must be uniform"

    # Off the start of the episode -> None, never a clamped/stationary run-up.
    assert history_frame_indices(1, 3, 1) is None, "t0=1 cannot fill a 3-frame stride-1 window"
    assert history_frame_indices(2, 3, 1) == [0, 1, 2], "t0=2 exactly fills it"
    assert history_frame_indices(3, 3, 2) is None, "t0=3 cannot fill a 3-frame stride-2 window"
    assert history_frame_indices(4, 3, 2) == [0, 2, 4], "t0=4 exactly fills it"

    # A longer history costs more run-up, which is what makes the first replans unscorable.
    assert history_frame_indices(4, 5, 2) is None
    assert history_frame_indices(8, 5, 2) == [0, 2, 4, 6, 8]
    print("molmo_motion history-window self-test OK")
