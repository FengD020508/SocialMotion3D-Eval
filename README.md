# SocialMotion3D-Eval

IDD-PeD 上 E1（MotionBERT/GEM）、E2a（固定人体、受控 ego-motion）和 E3（DROID/MegaSAM 对 OBD 速度）的可复现实验代码。

当前 pilot 的原则：不重新运行 GEM；E3 先独立校准相机尺度，再将该尺度用于 E2a；准确性采用共同有效区间，鲁棒性单独报告。完整定义见 `docs/protocol.md`。

## 运行

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
python scripts/run_e1.py --config configs/private_e1_batch23.json --render-blind
python scripts/run_e3.py --config configs/private_e3_pilot.json
python scripts/run_e2a.py --config configs/private_e2a_pilot.json
```

集群 CPU 作业：

```bash
mkdir -p ops_private/slurm
sbatch -o "ops_private/slurm/%x_%j.out" \
  --export=ALL,E3_CONFIG="$PWD/configs/private_e3_pilot.json",E2A_CONFIG="$PWD/configs/private_e2a_pilot.json" \
  slurm/run_pilot.sbatch
```

真实输入、结果和操作日志均由 `.gitignore` 排除。
