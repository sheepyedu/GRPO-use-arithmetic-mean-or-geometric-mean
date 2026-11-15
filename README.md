# GRPO 序列概率比的算术平均与几何平均

本项目研究序列概率比的算术平均（AM）与几何平均（GM）在强化学习中的性能差异与机制。主模型固定为官方 `Qwen/Qwen3-8B`，默认统一启用 thinking 模式。

当前进入实现与试运行阶段。Node01已复用官方Qwen3-8B，固定三套公开数据及双环境6000题划分，9项测试通过。首个seed17按纯数学AM/GM、混合AM/GM顺序推进。正式完成0/20，当前试运行结果待核。

## 实验设计

正式实验包含 2 个训练环境 × 2 个目标 × 5 个配对 seed，共 20 个运行。

| 训练环境 | 训练数据 | AM | GM |
|---|---|---:|---:|
| 纯数学 | 6,000 道 GSM8K | 5 个 seed | 5 个 seed |
| 混合 | 3,000 道 GSM8K、1,500 道 ARC-Easy、1,500 道 BoolQ | 5 个 seed | 5 个 seed |

配对 seed 为 `[17, 29, 43, 71, 101]`。训练 temperature=1.0、top-p=1.0，评估 temperature=0、每题一次贪心生成。每个运行计划进行 200 轮 rollout，每轮 64 个 prompt，每题 8 个回答，共 102,400 条训练回答，20 个运行合计 2,048,000 条。

两组共享优势计算、序列级裁切、KL、数据顺序与训练预算，仅改变序列概率比聚合算子。AM 组不是标准 token 级 GRPO。两个环境均测试 GSM8K、ARC-Easy 和 BoolQ，按环境报告五个 seed 的配对统计。

完整协议包含数据隔离、奖励解析、目标公式、共同配置、统计与机制诊断要求。见[实验方案](docs/experiment_plan.md)。

## 运行

实现采用 Transformers/PEFT 训练与 vLLM 单卡 rollout，每卡独立进程。后端决策与共同配置见 [实施决策](docs/method_decision.md)，测试见 [TEST.md](TEST.md)。服务器工作目录为 `/home/remoteadmin/grpo-sequence-ratio`，模型与数据路径在 `configs/seed17.json`。

```sh
export PYTHONPATH="$PWD"
export USE_TF=0 USE_FLAX=0
python -m unittest discover -s tests -v
python -m src.data --output artifacts/data-v1
python scripts/run_seed.py --config configs/pilot-seed17.json --output artifacts/pilot-seed17-v1 --phase pilot
# 四个条件试运行校准完成并冻结共同配置后执行
python scripts/run_seed.py --config configs/seed17.json --output artifacts/formal-seed17-v1 --phase formal
```

`pilot-seed17.json` 由seed17共同配置复制，将轮数改为10；试运行产物与正式run独立。已存在的数据清单和run目录需核实复用，入口拒绝覆盖。当前正式训练尚未启动。

每个run保存配置、源码和环境清单、原始回答/token、冻结行为/参考分数、逐轮指标、LoRA与AdamW/scheduler/RNG检查点、最终测试预测及终态回执。异常保留原日志。完整checkpoint和原始轨迹位于服务器 `artifacts/`，GitHub保存可复现源码及结果摘要。

## 项目进度

[Notion根项目](https://app.notion.com/p/4195e704b6a88148bbfcd9b3c8240945) · [首seed四run](https://app.notion.com/p/4195e704b6a881308564c2afff182ec4)
