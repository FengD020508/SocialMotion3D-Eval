# E2a / E3 pilot protocol

## E3：DROID 与 MegaSAM 的 ego-motion 评估

目标不是证明“使用 ego-motion 一定改善人体姿态”，而是检验两种相机轨迹恢复方法能否重建与车辆 OBD 一致的自车运动，并为 E2a 提供独立尺度。

每段视频使用相同的 OBD `OBD_speed`（原单位 km/h，读取后换算为 m/s）。相机位置的一阶差分转为无尺度视觉速度；OBD 和视觉速度使用相同的时间对齐、平滑窗口及差分规则。

尺度只在时间上靠前的 40% 区间拟合，采用过原点非负最小二乘；MAE、RMSE、Pearson、WRDE 和尺度稳定性只在后 60% 区间计算。这样不会用被评价区间反推尺度。

准确性指标只在 DROID 与 MegaSAM 共同有效、且 OBD 有效的区间上计算，以保证两者面对相同样本。有效性由统一规则重新计算：数值有限、旋转矩阵接近 SO(3)、相邻时间为正、相机位移不属于稳健离群跳变。方法自己输出的失败标记只作为诊断，不参与共同样本筛选。

报告内容：

- Accuracy：MAE、RMSE、Pearson correlation coefficient。若任一速度序列近似常量，Pearson 记为 `null`，不强行解释。
- Scale stability：在 1、2、3 秒滑窗内重新估计局部尺度，报告 median、IQR、CV。
- WRDE：在 1、2、3 秒滑窗内比较预测路程和 OBD 路程；OBD 路程小于 0.5 m 的窗口不参与，避免近零分母夸大误差。
- Robustness：各方法独立的 frame/interval Valid Ratio；另报告方法自带 failed ratio。

5 段 pilot 只做描述性比较，报告逐片结果、宏平均和 MegaSAM−DROID 的配对差值，不据此声称统计显著性。

边界：OBD 速度是标量，因此只能验证尺度和速度大小，不能单独判定 world 坐标轴的正负方向。方向验收依赖 `T_c2w/T_w2c` 互逆、SO(3)、既有 y-up 转换以及重投影/定性检查；论文中不能把标量速度相关性表述成“方向也已由 OBD 验证”。

## E2a：固定人体、只改变 ego-motion

使用 `02_gp_set_0003_vid_0005_gp_6979_HG`。主实验固定 MegaSAM 流程产生的 GEM `body_params_incam`，分别做：

1. No ego：直接把逐帧 incam root translation 当作固定坐标轨迹；
2. DROID ego：`R_c2w @ p_incam + s_droid * camera_center`；
3. MegaSAM ego：`R_c2w @ p_incam + s_megasam * camera_center`。

其中相机尺度来自该片 E3 的独立 OBD 校准。对称性检查固定 DROID 流程产生的同一份 GEM incam，再重复上述三种 grounding。整个 E2a 不重新运行 GEM。

定量结果使用刚体坐标变换不变量：净位移、路径长度、root speed 的中位数和 P95，并绘制 root speed 与累计路程。不同后端的 world 坐标轴未对齐，因此 top-down 图只作各自轨迹的定性诊断，不用作跨方法的数值误差。

本片是机制验证和论文示意图，不代表总体结论；后续用新增静止/低速行人片段进行正式统计。

## 输出与复现边界

- `results/`：数值报告、逐帧序列和图，不进入代码仓库。
- `data_private/`：OBD 和其他私有输入，不进入代码仓库。
- `ops_private/`：集群命令、作业号、清理记录和提交版本，不进入代码仓库。
- 代码、配置模板、协议和测试进入 GitHub。真实绝对路径配置以 `configs/private_*.json` 保存并忽略。
