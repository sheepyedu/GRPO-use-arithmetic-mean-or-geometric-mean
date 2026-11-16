# GRPO 序列概率比的算术平均与几何平均

本项目研究序列概率比的算术平均（AM）与几何平均（GM）在强化学习中的性能差异与机制。主模型固定为官方 `Qwen/Qwen3-8B`，默认统一启用 thinking 模式。

首个seed17的四组正式run已于2026-11-16北京时间13:03:58完成，正式进度4/20。PAI GPU0–3均已释放。纯数学AM/GM各36轮，GSM8K最终测试93.86%/93.33%；混合AM/GM各40轮，加权测试93.46%/92.75%。四组均达到原12小时软预算，最终测试、完整训练状态及文件哈希核验通过。训练生成44192411 token，含验证和测试总输出77399019 token。详见[首seed结果](reports/seed17-results.md)及[机器可读摘要](reports/seed17-summary.json)。

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

首seed四run执行及汇总已完成，固定审核提交按项目流程办理。后续seed29、43、71、101共16run仍待执行，五seed统计和根项目审核待完成。30分钟监督使用共享时间门，保留原始失败和科学预算。

[Notion根项目](https://app.notion.com/p/4195e704b6a88148bbfcd9b3c8240945) · [首seed四run](https://app.notion.com/p/4195e704b6a881308564c2afff182ec4)

## 首seed详细汇报

[黑体PDF报告（12页）](output/pdf/GRPO首seed实验详报_黑体_20261116.pdf)包含完整训练曲线、资源时间、token及归因分析。新增逐轮核查发现数学AM第9轮KL有限尖峰258.71179，具体token成因待定位。报告按实际GRPO目标撰写，DPO未实施。

## 效率优化与后续seed

后续seed使用source-v4，开启vLLM CUDA Graph与64请求槽，1536token以内关闭梯度检查点，延迟同步诊断。生成短测约4.46倍，短序列反向耗时下降约31%。21项测试通过，首完整训练周期已实测约1.34–1.39倍，稳态速度继续积累。详见[效率报告](docs/efficiency-v4.md)。不同后端版本在跨seed统计中披露。

## 通用推理训练提速Skill

已沉淀[推理与训练提速Skill](skills/inference-training-acceleration/SKILL.md)，包含耗时分解、候选选择、正确性与显存核验、正式入口复测，以及事件分析脚本和H20实测参考。本机已安装到`~/.codex/skills/inference-training-acceleration`，可用`$inference-training-acceleration`调用。
