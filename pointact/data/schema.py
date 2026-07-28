# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Schema for the dataset configuration."""

from dataclasses import dataclass, field

import yaml


@dataclass
class MMDatasetConfig:
    json_path: str
    sampling_strategy: str = "all"
    vision_base_path: str | None = None


@dataclass
class LerobotConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    is_delta_action: bool = False
    is_action_eef: bool = True

    state_action_norm_file: str | None = None

    converted_rot_type: str | None = "euler"  # None, euler, rot6d

    train_subtask: str | bool | None = False  # Optional[true, false, mix:0.5, cumulate]
    select_video_keys: list[str] = None
    video_key_ids_for_vlm: list[int] | None = None
    select_depth_keys: list[str] = None
    select_camera_keys: list[str] = None
    select_action_keys: list[str] = None
    select_state_keys: list[str] = None
    advantage_key: str | None = None
    effector_indices: list[int] = None
    weight: float | None = None
    
    points_workspace: dict | None = None
    max_npoints: int = 4096
    augment_pc_rot: int = 0 # 0: no rotation augmentation on z-axis, [-rot, rot], unit: degrees
    point_cloud_dirname: str | None = None

    # ROI-guided point sampling (optional). Directory (under `root`) of the per-point
    # ROI-flag LMDB produced by data_prep/roi_sampling; None disables it (uniform).
    roi_point_cloud_dirname: str | None = None
    roi_ratio: float = 0.7
    roi_radius_scale: float = 1.0
    roi_mode: str = "hard"      # "hard" (ball) or "soft" (Gaussian falloff)
    roi_softness: float = 1.0

    # EEF-density point sampling (optional, mutually exclusive with the ROI fields above).
    # No cache needed: the anchor is the frame's own end-effector position. See
    # pointact.roi_sampling.geometry.eef_density_weights.
    eef_sampling: bool = False
    eef_sampling_sigma: float = 0.08
    eef_sampling_floor: float = 0.05

    # Oracle point sampling (optional): the upper bound on what any learned sampler could buy.
    # Same Gaussian-with-floor density as eef_sampling above, but centred on the handle the
    # gripper is reaching for rather than on the gripper, so the two arms differ only in the
    # anchor. `oracle_label_dirname` is the per-point label LMDB (under `root`) written by
    # convert.py --point-labels.
    # Labels: 0 background, 1 robot, 2 target fixture, 3 target door/drawer panel, 4 handle.
    oracle_sampling: bool = False
    oracle_label_dirname: str | None = None
    oracle_anchor_labels: tuple[int, ...] = (4,)
    oracle_anchor_fallback_labels: tuple[int, ...] = (3,)
    oracle_sampling_sigma: float = 0.08
    oracle_sampling_floor: float = 0.05

    image_size: int | None = None

    class_name: str = 'LeRobotDataset'


@dataclass
class DataConfig:
    mm_datasets: list[MMDatasetConfig] = field(default_factory=list)
    lerobot_datasets: list[LerobotConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            mm_datasets=[MMDatasetConfig(**d) for d in raw.get("mm_datasets") or []],
            lerobot_datasets=[LerobotConfig(**d) for d in raw.get("lerobot_datasets") or []],
        )
