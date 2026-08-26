# MegaSAM per-clip cache policy

The patched `run_megasam_targetlocked_pipeline.py` keeps its historical behavior by default (`--cache_policy keep`). New batch jobs should use `--cache_policy metric_tmp`.

Under `metric_tmp`, the work root must resolve inside the Slurm-provided `$SLURM_TMPDIR`; the script refuses to run this policy outside a Slurm temporary directory. No persistent symlink is created.

For every clip, cleanup is gated by validation of both:

- the complete `camera_trajectory.npz`;
- the annotation-locked `camera_trajectory_target.npz`, including the expected target-frame count.

Only after both checks pass, the script deletes the exact clip-scoped paths `frames/`, `mono_depth/`, the work copy `megasam_raw.npz`, and MegaSAM's official raw-output copy. `metric_depth/` remains in `$SLURM_TMPDIR` for the rest of that job and is removed automatically with the Slurm temporary directory. The canonical and target camera NPZ files remain persistent.

Every cleanup writes a JSON line to `megasam_cleanup_log.jsonl`, including validated files, exact removed paths, and removed byte counts. A failed validation prevents cleanup and stops the clip.

`--cache_policy delete_all` is available when even the temporary metric depth should be deleted immediately; it uses the same validation gate. Existing jobs that do not opt in continue to retain all caches.

