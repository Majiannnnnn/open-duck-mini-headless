# duck

无头 MiniDuck 的强化学习与仿真项目。用 MuJoCo MJX 训练十个关节的步态策略，通过 Brax PPO 采样和更新，从保存点导出 ONNX，再用原生 MuJoCo 运行策略并录制视频。

项目自行实现环境、观测、动作控制和奖励。默认任务参考 Rose 的 `directional_v4`。项目自带的冻结策略用于对照仿真和校验，训练从随机参数开始。已完成 CPU 冒烟测试和正式 GPU 训练，自训策略通过原生 MuJoCo 的六方向测试。

## Simulation Results

使用本项目从随机参数训练得到的第 655,360,000 步策略，在原生 MuJoCo 的标称平地上测试。低速和常规两档分别执行六个方向，每次站立预热 1 秒，再执行 6 秒固定指令。自训策略 12 次均按指令方向运动，全程未跌倒。

[观看自训策略前进录像（MP4，6 秒）](docs/media/mujoco-forward-655360000.mp4)

| 常规档 | 指令 | 自训策略 | Rose V4 对照 |
| --- | --- | ---: | ---: |
| 前进 | `vx = +0.06 m/s` | 45.5 cm | 44.1 cm |
| 后退 | `vx = −0.06 m/s` | 43.5 cm | 43.5 cm |
| 左移 | `vy = +0.08 m/s` | 56.8 cm | 58.7 cm |
| 右移 | `vy = −0.08 m/s` | 55.1 cm | 54.2 cm |
| 左转 | `yaw_rate = +0.35 rad/s` | 111.7° | 111.3° |
| 右转 | `yaw_rate = −0.35 rad/s` | 119.3° | 127.9° |

两份策略使用相同的仿真和指令条件。表中是沿指令方向测得的位移或转角。两档测试中，自训策略的平移平均速度比指令高约 14%–35%。本节记录的是仿真结果，实机表现待验证。

## 安装与自检

以下命令在 WSL 或 Linux 的项目根目录运行。本机项目位于 `/home/milky/projects/duck`。使用 Python 3.12 和 [requirements.txt](requirements.txt) 中的版本。本地自检可用 CPU。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python

.venv/bin/python -m miniduck.cli validate
JAX_PLATFORMS=cpu .venv/bin/python -m miniduck.cli validate-alignment
JAX_PLATFORMS=cpu .venv/bin/python -m miniduck.cli validate-curriculum
```

| 命令 | 检查内容 |
| --- | --- |
| `validate` | 资产哈希、MJCF、关节和控制时钟、观测协议、参考步态与扰动库 |
| `validate-alignment` | JAX 与 NumPy 观测、0/1/2 帧目标延迟、起停和换向时序、发送失败后的状态、冻结策略的 ONNX 与 NumPy 输出 |
| `validate-curriculum` | 课程进度、状态重置、物理与传感器扰动、奖励分项与汇总 |

`validate-alignment --policy-stem 路径` 可检查另一套兼容策略。传入 ONNX、NPZ 和 JSON 共用的文件名前缀，省略扩展名。CPU 自检可能打印 Warp 导入和 ONNX Runtime GPU 探测提示。以命令的退出状态和末尾检查结果为准。

## 用 V4 策略看前进效果

以下命令使用随项目提供的冻结权重，按 `vx=0.06 m/s` 向前运行 6 秒。每次会在 `runs/` 新建一个输出目录，保存 MP4 和仿真报告。录像用 EGL 离屏渲染，需要 WSL 或 Linux 中可用的 EGL。编码使用系统 FFmpeg 或依赖中附带的 FFmpeg。

```bash
demo_dir=$(mktemp -d runs/v4-demo-XXXXXX)
MUJOCO_GL=egl EGL_PLATFORM=surfaceless OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m miniduck.cli record \
  --policy fixtures/directional_v4/directional_v4_000524288000.onnx \
  --command 0.06 0 0 --duration 6 \
  --video "$demo_dir/forward.mp4" --report "$demo_dir/forward.json"
```

`--command` 的顺序是 `vx vy yaw_rate`，测试时最多一个轴非零。三个轴的幅度上限分别是 0.08 m/s、0.10 m/s 和 0.45 rad/s。仿真先站立 1 秒，MP4 从指令生效时开始录制。报告包含起步航向坐标系下的前进与侧向位移、转角、跌倒状态、推理次数和控制步数。

## 从零跑一次训练、导出和录像

先查看正式训练计划。`--plan` 只打印解析后的配置。

```bash
.venv/bin/python -m miniduck.cli train --plan --output runs/v4-plan
```

再用 CPU 运行 160 条环境转移的冒烟测试。`run_dir` 必须是尚未创建的目录，重复执行时换个名字。

```bash
run_dir=runs/ppo-smoke-local
JAX_PLATFORMS=cpu .venv/bin/python -m miniduck.cli train \
  --smoke --output "$run_dir"

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 .venv/bin/python -m miniduck.cli export \
  "$run_dir/checkpoints/000000000160" "$run_dir/export/actor.onnx"

MUJOCO_GL=egl EGL_PLATFORM=surfaceless OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m miniduck.cli record \
  --policy "$run_dir/export/actor.onnx" --command 0.06 0 0 --duration 6 \
  --video "$run_dir/export/forward.mp4" --report "$run_dir/export/forward.json"
```

导出会生成同名的 ONNX、NPZ 和 JSON，并用 514 条原始观测校验 Brax、ONNX 和 NumPy 的输出。ONNX 内含归一化与确定性 actor，输入是原始观测，输出是 10 维动作。录像读取完整的三件套。导出和录像要求输出文件尚不存在，重复执行时使用新的输出文件名。

已保存的 CPU 冒烟记录显示，actor、critic 和归一化统计都发生了更新，保存点重载通过。用同一条 6 秒前进指令，160 步策略前向位移约 −0.23 mm，V4 冻结策略约 +44 cm。冒烟结果用于验证训练与推理链路。

### 训练配置和产物

| 配置 | V4 正式计划 | CPU 冒烟 |
| --- | ---: | ---: |
| 训练环境 / 评估环境 | 8,192 / 192 | 4 / 2 |
| 回合长度 / 起步等待 | 650 / 50 步 | 24 / 4 步 |
| 每段轨迹 | 40 步 | 40 步 |
| 每批环境转移 | 327,680 | 160 |
| 同一批数据的更新遍数 | 4 | 1 |
| 总环境转移 | 655,360,000 | 160 |

正式配置面向 GPU。训练从随机 actor、critic 和空的观测归一化统计开始。Actor 接收 75 维观测，critic 接收 175 维观测，两个网络均使用 `512 → 256 → 128` 的 SiLU 隐层，评估使用确定性动作。可用 `train --config 路径` 传入另一份 PPO 配置，用 `--seed` 覆盖种子。

| 运行目录中的文件 | 内容 |
| --- | --- |
| `run_config.json` | 实际 PPO、任务、课程配置、计算后端和依赖版本 |
| `policy_contract.json` | 观测、控制和机器人模型协议 |
| `metrics.jsonl` | 按环境步数记录的训练、评估和校验指标 |
| `progress.json` | 运行状态、采样步数和已通过重载校验的保存点 |
| `checkpoints/<12 位环境步数>/` | Orbax 保存点、网络配置和训练步数 |
| `export/` | 按上述命令生成的策略、视频和仿真报告 |

状态 `budget_complete_unvalidated` 表示已完成采样预算并保存末次参数，步态表现留给仿真评估。保存点可重载模型参数和归一化统计。精确断点续训需要保存优化器、随机状态和并行环境状态，目前尚未实现。

## 读代码和修改参数

| 路径 | 职责 |
| --- | --- |
| [configs/robot.json](configs/robot.json) 和 [miniduck/contract.py](miniduck/contract.py) | 共享关节顺序、初始姿态、限位、控制时钟和观测切片 |
| [configs/task.json](configs/task.json) | 回合时序、六方向命令、重置、奖励权重、参考步态和地面摩擦 |
| [configs/curriculum.json](configs/curriculum.json) | 可选的课程、动力学扰动、传感器误差和延迟 |
| [configs/ppo.json](configs/ppo.json) | 网络、并行环境数、采样批量和训练预算 |
| [miniduck/env/](miniduck/env/) | 加载模型，执行 MJX 的 `reset` 和 `step`，拼接观测、计算关节目标和奖励 |
| [miniduck/learning/](miniduck/learning/) | 并行包装、Brax PPO 接入、日志和保存点 |
| [miniduck/inference/](miniduck/inference/) | 策略对照、运行时状态、ONNX 导出和原生 MuJoCo 仿真 |
| [miniduck/cli.py](miniduck/cli.py) | 自检、训练、导出和录像的命令入口 |

直接使用单环境时，可以 JIT 编译 `reset` 和 `step`。

```python
import jax
import jax.numpy as jnp

from miniduck.env.env import DuckEnv

env = DuckEnv()
reset = jax.jit(env.reset)
step = jax.jit(env.step)
state = reset(jax.random.key(0))
state = step(state, jnp.zeros(env.contract.action_size))
print(state.obs["state"].shape, state.obs["privileged_state"].shape)
```

- **物理时钟**。每 20 ms 执行一次控制，推进 10 个 2 ms 的物理子步。一个步态周期为 40 个控制步。
- **固定条件**。可在 `reset` 时指定 `command`、`start_phase`、`floor_friction` 和 `randomize_pose=False`。固定命令在起步等待后生效，可用于比较策略。正式 V4 训练会抽样单轴命令与起步相位，并允许回合中换向或停步。
- **终止和重置**。`step` 返回本次转移。`step_and_reset` 返回 `(transition, next_state)`，终止帧和指标保存在 `transition`，下一次控制读取 `next_state`。跌倒、非有限状态和时间截断各有标记，重置后保留课龄和该环境的摩擦样本。
- **观测与控制**。Critic 前 75 维与 actor 相同。参考步态库形状为 `180 × 40 × 40`，`32:34` 是左、右脚接触切片。Actor 使用当前控制相位，活动参考帧查表使用 +15 帧偏移。观测的目标历史记录限速限位后的 `sent`，MJX 使用延迟队列输出的 `applied`。NumPy 运行时按 `prepare → infer → propose → commit` 顺序工作，目标成功发送后提交历史和相位。
- **奖励排查**。V4 奖励在 [rewards.py](miniduck/env/rewards.py)。`reward_raw/<name>` 记录分项原值，`reward_contribution/<name>` 记录乘权重和控制步长后的带符号贡献。贡献求和后裁剪到 `[-10000, 10000]` 得到该步奖励。
- **可选课程**。用 `DuckEnv(curriculum=True)` 或 `train --curriculum` 开启，也可传入配置路径。课龄按每个环境的控制步递增，跨回合保留。默认课程的命令幅度从 70% 增至全幅，扰动按课龄加强，姿态引导权重为零。物理参数可现采样，也可从扰动库按整组抽样。纯 V4 基准使用干净传感器、零动作延迟和 `0.5–1.1` 的地面摩擦。

## 资产和版本管理

- [assets/](assets/) 包含 MJCF、STL、参考步态和动力学扰动库。[资产清单](assets/manifest.json) 记录了课堂包中的来源和 SHA-256。
- [fixtures/directional_v4/](fixtures/directional_v4/) 是从 `rose5720-from-ta-dev_full.zip` 提取的冻结策略和校验清单。默认的运行时对齐检查依赖这组文件。
- 源码、配置、资产和冻结对照均属于项目文件。训练和仿真生成的 checkpoint、日志、策略、视频与报告统一放在 `runs/`，通过 [.gitignore](.gitignore) 排除。需要长期保留的运行产物请单独备份。受哈希校验的文本资产通过 [.gitattributes](.gitattributes) 保留原始字节。
