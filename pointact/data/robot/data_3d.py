import os
import random
from collections.abc import Callable
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from lerobot.constants import ACTION, OBS_STATE
from pointact.constants import OBS_POINTS

from pointact.data.robot.base import LeRobotDatasetMixin
from pointact.data.robot.registry import register_robot_dataset
from pointact.data.transforms.pointcloud import (
    augment_point_cloud_color,
    random_rotate_point_around_z,
    random_rotate_quat_around_z,
    random_rotate_delta_quat_around_z,
)
from pointact.roi_sampling.geometry import eef_density_weights, halo_weights
from pointact.roi_sampling.sampling import density_weighted_indices, roi_guided_indices, soft_guided_indices

msgpack_numpy.patch()


@register_robot_dataset("LeRobotPointCloudDataset")
class LeRobotPointCloudDataset(LeRobotDatasetMixin):
    """LeRobot dataset variant backed by precomputed point clouds in LMDB.

    Point cloud LMDB entries are xyzrgb with RGB in [0, 1]. 
    State/action tensors are treated as world-frame values before optional rotation augmentation and point-cloud centering.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        select_video_keys: list[str] | None = None,
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
        train_subtask: str | None = None, # cumulate, mixture:0.5
        is_delta_action: bool = False,
        is_action_eef: bool = True,
        weight: float | None = None,
        image_size: int | None = None,
        converted_rot_type: str | None = None, # quat (default),euler, rot6d
        state_action_norm_file: str | None = None,
        # point related
        video_key_ids_for_vlm: list[int] | None = None,
        points_workspace: dict | None = None,
        max_npoints: int = 4096,
        augment_pc_rot: int = 0,
        point_cloud_dirname: str | None = None,
        # ROI-guided sampling (optional). When roi_point_cloud_dirname is set, a
        # per-point ROI flag LMDB (same keys/order as the point LMDB) is loaded and
        # the uniform subsample is replaced by a guarded ROI/background split of the
        # same total size. Missing/empty ROI falls back to uniform sampling.
        roi_point_cloud_dirname: str | None = None,
        roi_ratio: float = 0.7,
        # Halo radius multiplier applied on top of the cached radius, and hard/soft
        # selection. Both are runtime knobs: the cache stores the halo, not a baked mask.
        roi_radius_scale: float = 1.0,
        roi_mode: str = "hard",     # "hard" (ball) or "soft" (Gaussian falloff)
        roi_softness: float = 1.0,  # soft mode: sigma as a multiple of radius
        # EEF-density sampling (optional, mutually exclusive with roi_point_cloud_dirname).
        # No cache needed: the anchor is the frame's own end-effector position (state[:3],
        # already in the point-cloud/base frame before centering). Replaces the uniform
        # subsample with weight-proportional sampling under a Gaussian-with-floor density,
        # so points near the eef dominate the budget while every point stays reachable.
        eef_sampling: bool = False,
        eef_sampling_sigma: float = 0.08,   # Gaussian bandwidth, meters
        eef_sampling_floor: float = 0.05,   # minimum weight at infinite distance
        # Oracle sampling (optional): the upper bound on what any learned sampler could buy.
        # Uses the SAME Gaussian-with-floor density as eef_sampling above, so the two arms
        # differ only in where the bump is centred: the gripper (eef_sampling) vs. the handle
        # it is reaching for (here). The anchor is the centroid of the points the simulator
        # labels as the handle, read from a sibling label LMDB written by convert.py
        # --point-labels. Privileged information: preprocessing only, never a policy input.
        oracle_sampling: bool = False,
        oracle_label_dirname: str | None = None,
        # Labels whose centroid anchors the Gaussian. Default (4,) = the handle alone.
        oracle_anchor_labels: tuple[int, ...] = (4,),
        # Used when no anchor-labelled point is visible this frame (the handle can be occluded
        # by the gripper or face away): default (3,) = the drawer front panel it sits on.
        oracle_anchor_fallback_labels: tuple[int, ...] = (3,),
        oracle_sampling_sigma: float = 0.08,   # Gaussian bandwidth, meters
        oracle_sampling_floor: float = 0.05,   # minimum weight at infinite distance
        **kwargs,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            select_action_keys=select_action_keys,
            select_state_keys=select_state_keys,
            select_video_keys=select_video_keys,
            video_key_ids_for_vlm=video_key_ids_for_vlm,
            train_subtask=train_subtask,
            is_delta_action=is_delta_action,
            is_action_eef=is_action_eef,
            weight=weight,
            image_size=image_size,
            converted_rot_type=converted_rot_type,
            state_action_norm_file=state_action_norm_file,
        )

        self.points_workspace = points_workspace
        self.max_npoints = max_npoints
        self.augment_pc_rot = augment_pc_rot

        assert point_cloud_dirname is not None
        self.point_cloud_dir = os.path.join(self.root, point_cloud_dirname)
        self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_txn = None
        self._point_cloud_lmdb_pid = None

        self.roi_ratio = roi_ratio
        self.roi_radius_scale = roi_radius_scale
        self.roi_mode = roi_mode
        self.roi_softness = roi_softness
        self.roi_point_cloud_dir = (
            os.path.join(self.root, roi_point_cloud_dirname)
            if roi_point_cloud_dirname is not None
            else None
        )
        self._roi_lmdb_env = None
        self._roi_lmdb_txn = None
        self._roi_lmdb_pid = None

        self.eef_sampling = eef_sampling
        self.eef_sampling_sigma = eef_sampling_sigma
        self.eef_sampling_floor = eef_sampling_floor

        self.oracle_sampling = oracle_sampling
        self.oracle_anchor_labels = tuple(int(label) for label in oracle_anchor_labels)
        self.oracle_anchor_fallback_labels = tuple(
            int(label) for label in oracle_anchor_fallback_labels
        )
        self.oracle_sampling_sigma = oracle_sampling_sigma
        self.oracle_sampling_floor = oracle_sampling_floor
        if oracle_sampling and oracle_label_dirname is None:
            raise ValueError("oracle_sampling requires oracle_label_dirname")
        self.oracle_label_dir = (
            os.path.join(self.root, oracle_label_dirname)
            if oracle_label_dirname is not None
            else None
        )
        self._oracle_lmdb_env = None
        self._oracle_lmdb_txn = None
        self._oracle_lmdb_pid = None

    def __del__(self):
        if getattr(self, "_point_cloud_lmdb_txn", None) is not None:
            self._point_cloud_lmdb_txn.abort()
            self._point_cloud_lmdb_txn = None
        if getattr(self, "_point_cloud_lmdb_env", None) is not None:
            self._point_cloud_lmdb_env.close()
            self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_pid = None
        if getattr(self, "_roi_lmdb_txn", None) is not None:
            self._roi_lmdb_txn.abort()
            self._roi_lmdb_txn = None
        if getattr(self, "_roi_lmdb_env", None) is not None:
            self._roi_lmdb_env.close()
            self._roi_lmdb_env = None
        self._roi_lmdb_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_lmdb_env"] = None
        state["_point_cloud_lmdb_txn"] = None
        state["_point_cloud_lmdb_pid"] = None
        state["_roi_lmdb_env"] = None
        state["_roi_lmdb_txn"] = None
        state["_roi_lmdb_pid"] = None
        return state

    def set_feature_keys(
        self, video_keys=None, state_keys=None, action_keys=None,
        video_key_ids_for_vlm=None, **kwargs
    ):
        # the point cloud can be constructed using all video keys, while the VLM only utilizes a subset of the video keys
        self.select_video_keys = self.meta.video_keys if video_keys is None else video_keys
        if video_key_ids_for_vlm is None:
            self.select_video_keys_for_vlm = self.select_video_keys
        else:
            self.select_video_keys_for_vlm = [self.select_video_keys[i] for i in video_key_ids_for_vlm]

        self.select_state_keys = (
            [key for key in self.meta.features if key.startswith(OBS_STATE)]
            if state_keys is None
            else state_keys
        )
        self.select_action_keys = (
            [key for key in self.meta.features if key.startswith(ACTION)]
            if action_keys is None
            else action_keys
        )

        self.select_feature_keys = self.select_video_keys_for_vlm + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{key}_is_pad" for key in self.select_action_keys]

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        delta_indices = delta_indices or self.delta_indices

        if self.weight is not None:
            idx = random.randint(0, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        item, query_indices = self.query_action_chunk(item, idx, ep_idx, delta_indices)
        item = self.add_video_frames(item, ep_idx, query_indices)
        self.apply_image_transforms(item, self.select_video_keys_for_vlm)

        point_cloud = self.load_point_cloud(ep_idx, frame_idx)
        # Frame-level halo (anchor, radius): unaffected by the workspace crop below.
        roi_halo = self.load_roi_halo(ep_idx, frame_idx)
        # Per-point ground-truth labels must be cropped alongside the cloud to stay aligned.
        point_labels = self.load_point_labels(ep_idx, frame_idx) if self.oracle_sampling else None
        point_cloud, point_labels = self.filter_point_cloud_by_workspace(point_cloud, point_labels)
        point_cloud = self.augment_point_cloud(point_cloud, item, roi_halo, point_labels)
        point_cloud = self.center_point_cloud(point_cloud, item)
        item[OBS_POINTS] = torch.from_numpy(point_cloud)

        self.convert_eef_rotation(item)
        self.normalize_state_action(item)
        self.select_task_text(item, ep_idx, idx)

        return self.post_process(item)

    def load_point_cloud(self, ep_idx: int, frame_idx: int):
        txn = self.get_point_cloud_lmdb_txn()
        point_key = f"{ep_idx}-{frame_idx}"
        point_cloud = txn.get(point_key.encode("ascii"))
        if point_cloud is None:
            raise KeyError(f"Point cloud '{point_key}' not found in {self.point_cloud_dir}")
        return msgpack.unpackb(point_cloud).copy().astype(np.float32)

    def filter_point_cloud_by_workspace(self, point_cloud: np.ndarray, point_labels=None):
        """Crop to the workspace box, keeping any per-point labels in lockstep."""
        if self.points_workspace is None:
            return point_cloud, point_labels

        workspace = self.points_workspace
        point_mask = (
            (point_cloud[:, 0] > workspace["X_BBOX"][0])
            & (point_cloud[:, 0] < workspace["X_BBOX"][1])
            & (point_cloud[:, 1] > workspace["Y_BBOX"][0])
            & (point_cloud[:, 1] < workspace["Y_BBOX"][1])
            & (point_cloud[:, 2] > workspace["Z_BBOX"][0])
            & (point_cloud[:, 2] < workspace["Z_BBOX"][1])
        )
        if point_labels is not None:
            point_labels = point_labels[point_mask]
        return point_cloud[point_mask], point_labels

    def get_point_cloud_lmdb_txn(self):
        current_pid = os.getpid()
        if self._point_cloud_lmdb_pid != current_pid:
            self._point_cloud_lmdb_env = None
            self._point_cloud_lmdb_txn = None
            self._point_cloud_lmdb_pid = current_pid

        if self._point_cloud_lmdb_env is None:
            self._point_cloud_lmdb_env = lmdb.open(
                self.point_cloud_dir,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._point_cloud_lmdb_txn = self._point_cloud_lmdb_env.begin(buffers=True)

        return self._point_cloud_lmdb_txn

    def get_roi_lmdb_txn(self):
        current_pid = os.getpid()
        if self._roi_lmdb_pid != current_pid:
            self._roi_lmdb_env = None
            self._roi_lmdb_txn = None
            self._roi_lmdb_pid = current_pid

        if self._roi_lmdb_env is None:
            self._roi_lmdb_env = lmdb.open(
                self.roi_point_cloud_dir,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._roi_lmdb_txn = self._roi_lmdb_env.begin(buffers=True)

        return self._roi_lmdb_txn

    def get_oracle_lmdb_txn(self):
        current_pid = os.getpid()
        if self._oracle_lmdb_pid != current_pid:
            self._oracle_lmdb_env = None
            self._oracle_lmdb_txn = None
            self._oracle_lmdb_pid = current_pid

        if self._oracle_lmdb_env is None:
            self._oracle_lmdb_env = lmdb.open(
                self.oracle_label_dir,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._oracle_lmdb_txn = self._oracle_lmdb_env.begin(buffers=True)

        return self._oracle_lmdb_txn

    def load_point_labels(self, ep_idx: int, frame_idx: int):
        """Ground-truth label per point, aligned with load_point_cloud's output order."""
        if self.oracle_label_dir is None:
            return None
        txn = self.get_oracle_lmdb_txn()
        point_key = f"{ep_idx}-{frame_idx}"
        buf = txn.get(point_key.encode("ascii"))
        if buf is None:
            raise KeyError(f"Point labels '{point_key}' not found in {self.oracle_label_dir}")
        return msgpack.unpackb(buf).copy().astype(np.uint8)

    def load_roi_halo(self, ep_idx: int, frame_idx: int):
        """Return (anchor_xyz, radius) for this frame, or None.

        The cache stores the halo itself — anchor(3), radius(1), n_in_box(1) as float32 —
        rather than a baked per-point mask, so radius scaling and hard/soft selection stay
        runtime knobs. None (or a zero radius, meaning no reliable detection) makes the
        caller fall back to uniform sampling.
        """
        if self.roi_point_cloud_dir is None:
            return None
        txn = self.get_roi_lmdb_txn()
        buf = txn.get(f"{ep_idx}-{frame_idx}".encode("ascii"))
        if buf is None:
            return None
        rec = np.frombuffer(bytes(buf), dtype=np.float32)
        if len(rec) < 4 or not np.isfinite(rec[:4]).all() or rec[3] <= 0:
            return None
        return rec[:3].astype(np.float64), float(rec[3])

    def oracle_anchor(self, point_cloud: np.ndarray, point_labels: np.ndarray):
        """Centroid of the ground-truth handle points, or None if nothing usable is visible.

        The handle is a small compact cluster (a few cm across), so its centroid is a good
        Gaussian centre. It can be occluded by the gripper or face away from every camera, in
        which case we fall back to the drawer panel it sits on rather than to no guidance at
        all; only when neither is visible does the caller revert to a uniform draw.
        """
        for labels in (self.oracle_anchor_labels, self.oracle_anchor_fallback_labels):
            if not labels:
                continue
            mask = np.isin(point_labels, labels)
            if mask.any():
                return point_cloud[mask, :3].mean(axis=0).astype(np.float64)
        return None

    def augment_point_cloud(
        self,
        point_cloud: np.ndarray,
        item: dict,
        roi_halo: tuple | None = None,
        point_labels: np.ndarray | None = None,
    ):
        # Baseline count rule is preserved exactly; only the *selection* changes when a
        # valid halo is present.
        max_npoints = min(int(len(point_cloud) * np.random.uniform(0.8, 1.0)), self.max_npoints)
        if len(point_cloud) > max_npoints:
            ridxs = None
            if self.oracle_sampling and point_labels is not None:
                # Same Gaussian-with-floor density as the eef arm, centred on the handle the
                # gripper is reaching for instead of on the gripper itself. Frames with no
                # visible anchor fall through to the uniform draw below.
                anchor = self.oracle_anchor(point_cloud, point_labels)
                if anchor is not None:
                    rng = np.random.default_rng(np.random.randint(2**31 - 1))
                    w = eef_density_weights(
                        point_cloud[:, :3], anchor,
                        self.oracle_sampling_sigma, self.oracle_sampling_floor,
                    )
                    ridxs = density_weighted_indices(len(point_cloud), max_npoints, w, rng)
            elif self.eef_sampling:
                rng = np.random.default_rng(np.random.randint(2**31 - 1))
                eef_pos = item[OBS_STATE][:3].numpy()
                w = eef_density_weights(
                    point_cloud[:, :3], eef_pos, self.eef_sampling_sigma, self.eef_sampling_floor
                )
                ridxs = density_weighted_indices(len(point_cloud), max_npoints, w, rng)
            elif roi_halo is not None:
                anchor, radius = roi_halo
                w = halo_weights(
                    point_cloud[:, :3], anchor, radius * self.roi_radius_scale,
                    mode=self.roi_mode, softness=self.roi_softness,
                )
                rng = np.random.default_rng(np.random.randint(2**31 - 1))
                if self.roi_mode == "soft":
                    if np.any(w >= 0.5) and not np.all(w >= 0.5):
                        ridxs = soft_guided_indices(
                            len(point_cloud), max_npoints, w, self.roi_ratio, rng
                        )
                else:
                    mask = w > 0
                    if mask.any() and not mask.all():
                        ridxs = roi_guided_indices(
                            len(point_cloud), max_npoints, mask, self.roi_ratio, rng
                        )
            if ridxs is None:  # no/unusable halo -> baseline uniform draw
                ridxs = np.random.choice(len(point_cloud), max_npoints, replace=False)
            point_cloud = point_cloud[ridxs]

        point_cloud_color = augment_point_cloud_color(
            point_cloud[:, 3:6],
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            jitter_std=0.02,
        )
        point_cloud[:, 3:6] = point_cloud_color * 2 - 1

        if self.augment_pc_rot != 0:
            angle = np.random.uniform(-1, 1) * np.deg2rad(self.augment_pc_rot)
            point_cloud[:, :3] = random_rotate_point_around_z(point_cloud[:, :3], angle=angle)
            if self.is_action_eef:
                item[OBS_STATE][:3] = random_rotate_point_around_z(
                    item[OBS_STATE][:3].unsqueeze(0), angle=angle
                )[0]
                item[OBS_STATE][3:7] = random_rotate_quat_around_z(item[OBS_STATE][3:7], angle)
                item[ACTION][:, :3] = random_rotate_point_around_z(item[ACTION][:, :3], angle=angle)
                if self.is_delta_action:
                    item[ACTION][:, 3:7] = random_rotate_delta_quat_around_z(item[ACTION][:, 3:7], angle)
                else:
                    item[ACTION][:, 3:7] = random_rotate_quat_around_z(item[ACTION][:, 3:7], angle)

        return point_cloud

    def center_point_cloud(self, point_cloud: np.ndarray, item: dict):
        point_center = point_cloud[:, :3].mean(0)
        point_cloud[:, :3] = point_cloud[:, :3] - point_center
        point_center = torch.from_numpy(point_center)
        if self.is_action_eef:
            item[OBS_STATE][:3] = item[OBS_STATE][:3] - point_center
            if not self.is_delta_action:
                item[ACTION][:, :3] = item[ACTION][:, :3] - point_center[None, :]
        item[f"{OBS_POINTS}.center"] = point_center
        return point_cloud

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ordered_keys = self.select_feature_keys + [OBS_POINTS, "task", f"{OBS_POINTS}.center"] + self.select_action_is_pad_keys
        item = {key: item[key] for key in ordered_keys if key in item}
        return item
