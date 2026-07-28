# ROI-guided point sampling (RoboCASA OpenDrawer PoC)

Concentrates the fixed 4096-point budget on the drawer/handle instead of sampling
uniformly. A lightweight open-vocab detector (YOLO-World) localizes the object in the
left/right frames; the stored base-frame cloud is reprojected to find the epicenter; a
3D "halo" around it flags ROI points; the dataloader then draws a count-preserving
guarded ROI/background split. The trained policy is unchanged — only which points it
sees. Design: `docs/superpowers/specs/2026-07-26-roi-guided-point-sampling-design.md`.

## Pipeline

```
Stage 0  setup_roi_env.sh                 # login node (internet): build .venv-roi, bake YOLO-World prompt
Stage A  roi_calib.slurm                  # robocasa365 env, V100: dump left/right camera calibration
Stage B  roi_build.slurm                  # .venv-roi, V100: build points_3views_roi flag LMDB
Gate     viz_roi.py                        # interactive HTML — eyeball before training
Train    roi/train_roi.slurm              # 2 H100: ROI variant, epoch=5
```

### Stage 0 — environment (login node)
```bash
bash data_prep/roi_sampling/setup_roi_env.sh
```
Builds `.venv-roi` (isolated from the training env) and bakes `"drawer handle"` into
`$SCRATCH/models/YOLO-World/yoloworld_s_drawerhandle.pt` so compute nodes need no network.

### Stage A — camera calibration
```bash
sbatch experiments/13_robocasa365/roi/roi_calib.slurm
```
Writes `OpenDrawer/roi_meta/camera_calib.npz`. Verifies base→cam is constant across
episodes (robot-mounted cameras); warns if not.

### Stage B — ROI-flag cache
```bash
# dev subset first:
sbatch --export=ALL,EPISODES="0 1 2" --qos=qos_gpu-dev --time=00:30:00 \
       experiments/13_robocasa365/roi/roi_build.slurm
# then the full dataset:
sbatch experiments/13_robocasa365/roi/roi_build.slurm
```
Writes `OpenDrawer/points_3views_roi` (bit-packed per-point flags, same keys/order as
`points_3views`) and `roi_meta/build_summary.json` (detect rate, ROI fraction).

### Gate — visualization
```bash
.venv-roi/bin/python data_prep/roi_sampling/viz_roi.py \
  --dataset-dir $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
  --out /tmp/roi_viz.html
```
Open the HTML locally; dropdown switches frames, legend toggles background / ROI /
post-sample.

### Gate — per-episode animation, one file per strategy
```bash
uv run --no-sync python -m data_prep.roi_sampling.viz_sampling_episode \
  --dataset-dir $SCRATCH/datasets/robot_data/robocasa365/lerobot_point_lmdb/OpenDrawer \
  --episode 0 --method all --out-dir /scratch/$USER/viz
```
Writes `sampling_ep0000_{uniform,eef,roi}.html`: the *same* episode and clouds under each
strategy, animated over the episode (play/pause + frame slider) so you can watch where the
4096-point budget goes as the arm approaches and pulls. Needs only the training env and
`points_3views` — the ROI halo is recomputed on the fly from proprioception (see
`build_roi_cache_proprio.py`), so no ROI cache is required. Point count is deterministic
here (no 0.8-1.0 jitter) so frames and strategies compare frame-for-frame.

### Train
```bash
module load arch/h100
sbatch --constraint=h100 experiments/13_robocasa365/roi/train_roi.slurm
```
Uses `data-robocasa365-opendrawer-point-roi.yaml` (identical to the baseline config
except `roi_point_cloud_dirname` + `roi_ratio`).
```
