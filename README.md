# GRPO 序列概率比的算术平均与几何平均

本项目研究序列概率比的算术平均（AM）与几何平均（GM）在强化学习中的性能差异与机制。主模型固定为官方 `Qwen/Qwen3-8B`，默认统一启用 thinking 模式。

当前按用户要求在阿里云PAI H20重新训练。GPU0–3同时运行seed17的纯数学AM/GM与混合AM/GM，随后推进其他seed。Node01试运行已停止并释放本作业显存，已验证的两轮证据保留。正式完成0/20，PAI环境、重新下载的固定数据及19项测试已通过，11月16日03:43实测四run第10轮验证已闭合，数学GSM8K两组均92.8%，混合加权91.7417%/91.7789%；04:45实测数学已完成14轮、当前第15轮，混合已完成16轮、当前第17轮，累计AdamW更新224/224/264/264次。首轮四份完整检查点CPU核验通过。正式run仍在途。

## 实验设计

正式实验包含 2 个训练环境 × 2 个目标 × 5 个配对 seed，共 20 个运行。

| 训练环境 | 训练数据 | AM | GM |
|---|---|---:|---:|
| 纯数学 | 6,000 道 GSM8K | 5 个 seed | 5 个 seed |
| 混合 | 3,000 道 GSM8K、1,500 道 ARC-Easy、1,500 道 BoolQ | 5 个 seed | 5 个 seed |

配对 seed 为 `[17, 29, 43, 71, 101]`。训练 temperature=1.0、top-p=1.0，评估 temperature=0、每题一次贪心生成。每个运行最多进行 200 轮 rollout，采用共同早停规则，每轮 64 个 prompt，每题 8 个回答，共 102,400 条训练回答，20 个运行合计 2,048,000 条。

两组共享优势计算、序列级裁切、KL、数据顺序与训练预算，仅改变序列概率比聚合算子。AM 组不是标准 token 级 GRPO。两个环境均测试 GSM8K、ARC-Easy 和 BoolQ，按环境报告五个 seed 的配对统计。

完整协议包含数据隔离、奖励解析、目标公式、共同配置、统计与机制诊断要求。见[实验方案](docs/experiment_plan.md)。

## 运行

实现采用Transformers/PEFT训练及vLLM单卡rollout。PAI模型为 `/mount-data/Models/Qwen3-8B`，配置为 `configs/pai-seed17.json`，工作根 `/mount-data/GRPO-Sequence-Ratio`。软件及共同配置见[实施决策](docs/method_decision.md)，测试见[TEST.md](TEST.md)。

```sh
export PYTHONPATH="$PWD"
export USE_TF=0 USE_FLAX=0
python -m unittest discover -s tests -v
HF_ENDPOINT=https://hf-mirror.com python -m src.data --output /mount-data/Datasets/GRPO-Sequence-Ratio/data-v1 --source-manifest reports/data-manifest.json
python scripts/run_seed.py --config configs/pai-seed17.json --output /mount-data/GRPO-Sequence-Ratio/runs/seed17 --phase formal --parallel-environments
```

## 早停

每10轮验证、至少20轮。AM和GM都连续两次未提升0.5个百分点时，在同一完整轮次停止。每个环境另设12小时软上限，可优先于最小轮数，在完整checkpoint提交后触发。最终验证和锁定测试另计。200轮为最大上限。保存共同停止轮次的模型，报告真实轮数及原因。停止判据只使用验证集。

原始轨迹、LoRA、AdamW/scheduler/RNG及检查点保存在服务器 `runs/`，GitHub保存源码、配置和小摘要。异常保留原日志和退出码。seed17四run全部终态、最终测试及结果汇总完成后，再进入下一seed。

## 项目进度

已启用本聊天每30分钟一次的定时监督，检查进程、轮次/更新、验证/测试、早停、资源及故障，并同步Notion。四条件第10轮验证提升后，按原规则平台早停最早第30轮，首seed训练及测试整体暂估11月16日11:00–14:00结束；若达到12小时软预算再完成评测，约13:00–15:00，后续按实测速率修正。

[Notion根项目](https://app.notion.com/p/4195e704b6a88148bbfcd9b3c8240945) · [首seed四run](https://app.notion.com/p/4195e704b6a881308564c2afff182ec4)
