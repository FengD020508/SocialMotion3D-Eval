# E1 / E2a / E3 pilot protocol

## E1：MotionBERT 与 GEM 的人体运动比较

E1 分为两层，避免把两类方法不共同定义的能力硬凑成一个“谁更准”的分数。

第一层是共同关节空间的技术诊断。23 段视频使用同一目标行人的连续标注区间；MotionBERT 使用目标中心裁剪、AlphaPose Halpe-26、OneEuro 和固定 checkpoint，GEM 使用统一的 DROID+GEM 输出。GEM 的 SMPL-X 先回归 COCO-17，再构造 H36M-like common-17；两者按原视频 local frame 取交集。所有自动指标在 pelvis-relative、按骨架总骨长归一化后计算，包括有效帧比例、骨长 CV、关节速度/加速度/jerk 的 P95 和稳健时序离群比例。另报告刚体对齐后的方法间 normalized MPJPE，它只表示两种重建的分歧，不是真值误差。

第二层是 social-cue fidelity 和系统能力。HG、LOS、FTT、crossing initiation、static 和 occlusion 片段使用相同骨架样式、匿名 A/B 顺序进行人工评分，记录技术成功、cue fidelity、onset fidelity 和偏好。能力表单独记录 native SMPL、native root trajectory 以及额外 fitting/轨迹标注步骤；不能因 MotionBERT 原生不输出 world trajectory 而给它虚构一个轨迹误差，也不能把 GEM 的相对/global translation 直接写成已获得绝对米制 world trajectory。

解释边界：没有 3D ground truth 时，较低 jerk 或骨长漂移只能支持“更稳定”，不能支持“姿态更准确”；social cue 保真度必须结合匿名人工评分。E1 允许出现 MotionBERT 在局部动作稳定性更好、但 GEM 因原生 SMPL 与动作—root trajectory 联合输出更适合 scene-ready MotionPool 的结论。

E1c 将动作—轨迹耦合单独评估。主条件为 GEM 原生 global common-17、经过一次整段旋转/尺度适配后放置在相同 GEM root trajectory 上的 MotionBERT，以及只将 GEM root-relative articulation 错开 15 帧的负对照。负对照不循环、不补帧，只裁剪共同区间；另以 ±8、±15、±30 帧检查剂量响应。人工材料必须使用相同参考视频、骨架、地面、相机和轨迹视图。MotionBERT shared-root 是外部轨迹组装控制，不得写成 MotionBERT 原生轨迹输出；common-17 脚接触指标是诊断，不是真值物理接触误差。

## E3：DROID 与 MegaSAM 的 ego-motion 评估

目标不是证明“使用 ego-motion 一定改善人体姿态”，而是检验两种相机轨迹恢复方法能否重建与车辆 OBD 一致的自车运动，并为 E2a 提供独立尺度。

每段视频使用相同的 OBD `OBD_speed`（原单位 km/h，读取后换算为 m/s）。相机位置的一阶差分转为无尺度视觉速度；OBD 和视觉速度使用相同的时间对齐、平滑窗口及差分规则。

尺度只在时间上靠前的 40% 区间拟合：对 OBD≥0.5 m/s 且视觉位移非零的共同有效区间，取逐帧 `OBD速度 / 视觉速度` 的中位数。该估计对连续少量视觉速度尖峰有 50% breakdown point；同时报告普通过原点最小二乘尺度作为敏感性诊断，但不用于后 60% 的主结果。MAE、RMSE、Pearson、WRDE 和尺度稳定性只在后 60% 区间计算，避免用被评价区间反推尺度。

准确性指标只在 DROID 与 MegaSAM 共同有效、且 OBD 有效的区间上计算，以保证两者面对相同样本。有效性由统一规则重新计算：数值有限、旋转矩阵接近 SO(3)、相邻时间为正、相机位移不属于稳健离群跳变。方法自己输出的失败标记只作为诊断，不参与共同样本筛选。

报告内容：

- Accuracy：MAE、RMSE、Pearson correlation coefficient。若任一速度序列近似常量，Pearson 记为 `null`，不强行解释。
- Scale stability：先用前 40% 拟合全局尺度，再在后 60% 的 1、2、3 秒滑窗内计算 `OBD路程 / 已标定预测路程` 的局部尺度修正比（理想值 1），报告 median、IQR、CV。真实路程或预测路程低于 0.5 m 的窗口均不参与。
- WRDE：在 1、2、3 秒滑窗内比较预测路程和 OBD 路程；OBD 路程小于 0.5 m 的窗口不参与，避免近零分母夸大误差。
- Robustness：各方法独立的 frame/interval Valid Ratio；另报告方法自带 failed ratio。
- Rotation stability diagnostic：报告相邻帧旋转角、角速度、角加速度、相对首帧倾斜，以及解缠后的 yaw 净变化/范围。车载视频允许真实转弯、颠簸和安装姿态变化，因此这些量只诊断抖动与漂移，不作为有真值的 rotation accuracy。

5 段 pilot 只做描述性比较；扩展到 18 段后仍报告逐片结果、宏平均和 MegaSAM−DROID 的配对差值，并按昼夜、动作和难度分层查看，样本量不足时不声称统计显著性。

边界：OBD 速度是标量，因此只能验证尺度和速度大小，不能单独判定 world 坐标轴的正负方向。方向验收依赖 `T_c2w/T_w2c` 互逆、SO(3)、既有 y-up 转换以及重投影/定性检查；论文中不能把标量速度相关性表述成“方向也已由 OBD 验证”。

## E2a：固定人体、只改变 ego-motion

使用 `02_gp_set_0003_vid_0005_gp_6979_HG`。主实验固定 MegaSAM 流程产生的 GEM `body_params_incam`，分别做：

1. No ego：直接把逐帧 incam root translation 当作固定坐标轨迹；
2. DROID ego：`R_c2w @ p_incam + s_droid * camera_center`；
3. MegaSAM ego：`R_c2w @ p_incam + s_megasam * camera_center`。

其中相机尺度来自该片 E3 的独立 OBD 校准。对称性检查固定 DROID 流程产生的同一份 GEM incam，再重复上述三种 grounding。整个 E2a 不重新运行 GEM。

定量结果使用刚体坐标变换不变量：端点位移、共同有效区间上的累计路程、root speed 的中位数和 P95，并绘制 root speed 与累计路程。为避免把 GEM 的深度跳变误写成 ego 效果，同一份固定人体下的三种 grounding 取共同有效区间：数值有限、通过稳健位移跳变检测且 root speed 不超过 8 m/s；同时报告各条件独立 Valid Ratio、共同 Valid Ratio 和未过滤 raw path。不同后端的 world 坐标轴未对齐，因此 top-down 图只作各自轨迹的定性诊断，不用作跨方法的数值误差。

本片是机制验证和论文示意图，不代表总体结论；后续用新增静止/低速行人片段进行正式统计。

## 输出与复现边界

- `results/`：数值报告、逐帧序列和图，不进入代码仓库。
- 每段 E3 另输出 OBD 标定后的 `camera_metric.npz`（米制 `camera_center_m/T_c2w/T_w2c`），供 E2a 和后续 world grounding 直接复用；这一步是后处理，不需要重新运行 GEM。
- `data_private/`：OBD 和其他私有输入，不进入代码仓库。
- `ops_private/`：集群命令、作业号、清理记录和提交版本，不进入代码仓库。
- 代码、配置模板、协议和测试进入 GitHub。真实绝对路径配置以 `configs/private_*.json` 保存并忽略。
