# IDD-PeD ego–human joint reconstruction

Activate the existing environment from the repository root:

```bash
source /home/infres/zwang-24/ENTER/etc/profile.d/conda.sh
conda activate lgm2motion
cd /home/infres/zwang-24/LGM2Motion
```

Run the prioritized five-clip workflow (01 and 03 first):

```bash
python scripts/idd_ped/run_joint_pipeline.py \
  --clips 01 03 02 04 05 \
  --scale_mode relative \
  --resume
```

If the original IDD-PeD tree is available, pass its root. The runner then reads
the manifest source video, adds 90 frames of temporal context at each end, runs
SLAM on that full-frame interval, and trims the trajectory back to the clip:

```bash
python scripts/idd_ped/run_joint_pipeline.py \
  --source_root /path/to/IDD-PeD \
  --clips 01 03 02 04 05 \
  --dynamic_mask \
  --scale_mode camera_height \
  --camera_height 1.5 \
  --resume
```

`--dynamic_mask` inpaints COCO person/car/motorcycle/bus/truck detections before
DROID, while retaining the original full image geometry. Supply calibrated
`fx fy cx cy` with `--calib path/to/calib.txt` when available.

`relative` normalizes the 95th percentile camera displacement to one unit.
`camera_height` fits a road plane from DROID keyframe depths and scales its
camera height to the requested metres. An unstable road plane is explicitly
reported and falls back to fitting camera scale against GENMO global-root
velocity with a root-smoothness term.

The GENMO demo accepts a trajectory directly:

```bash
python scripts/demo/demo_smpl_hpe.py \
  --video /path/to/full_frame_clip.mp4 \
  --camera_traj /path/to/camera_trajectory.npz \
  --ckpt_path inputs/pretrained/gem_smpl.ckpt \
  --hmr2_ckpt inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt \
  --output_root outputs/manual_run \
  --reprojection_only
```

Moving-camera inference requires `--camera_traj`. A static camera is permitted
only when explicitly requested with `--static_cam`; the pipeline never silently
falls back to it.
