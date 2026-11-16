# Qwen3-8B / H20 推理训练提速案例

2026-11-16，Qwen3-8B BF16、LoRA rank16/alpha32，四张H20各跑一个GRPO条件。torch2.10.0、transformers4.57.6、PEFT0.17.1、vLLM0.17.1，每轮64题×8回答、microbatch1、两epoch、16次AdamW。vLLM和HF在同卡交替，采样时HF驻CPU，优化时vLLM sleep。

## 耗时归属

首seed四run约53.04 GPU小时，事件区间为优化及相邻保存/切换35.20%、rollout31.29%、验证测试20.02%、冻结评分及首微批12.39%、配对等待1.01%、初始化0.09%。旧事件在optimizer step结束时打点，包含首微批与保存的归属偏差。新版本增加sleep_and_offload、optimization、checkpoint和wake_and_adapter事件。

## 采纳方法与实际测量

| 方法 | 测量结果 | 解释范围 |
|---|---|---|
| CUDA Graph和请求槽32→64联合启用 | 48个固定prompt、greedy、256token上限，两轮合并吞吐约4.46倍 | 联合设置收益，短测输出数12181–12215存在变化 |
| 正式初始验证 | 四组2570题，约776–808秒降至302–305秒，约2.54–2.67倍 | 验证集与基座相同，seed与后端版本不同 |
| 短序列关闭梯度检查点 | 8条保存序列，热测反向3.421→2.379秒，约1.44倍 | 分数和梯度一致，峰值分配约19.3→32.3GiB |
| 1536总token边界 | 合成输入128＋生成1408token，峰值35.49GiB，反向0.987→0.651秒 | 一条合成序列，实测分数/梯度一致 |
| 减少重复统计与同步 | 冻结评分跳过未使用熵，复用logsumexp，逐轮集中汇总detach标量 | 对应数值测试通过，各自独立提速量未测 |

1536阈值和64槽是该H20工作负载的选择。长序列沿用梯度检查点。图捕获初始化从22.56秒增至71.66秒，需由任务持续时间摊销。LoRA加载与sleep/wake组合实测通过。候选21项测试通过。

## 撤回的候选和真实失败

- 变长序列padding batch2/4增加无效计算，比逐条慢，梯度相对L2差异约9.9%。恢复microbatch1。
- 词表统计torch.compile在充分预热后收益较小，梯度相对差异约1.06%。最终采用eager词表统计。
- 首次生成测速脚本缺少spawn的main guard，初始化失败。保留exit1和原日志，修正版本取得两配置exit0。
- 1536边界测试的原RESULT沿用通用scope文字，文字声称8条保存样本，实际rows为1条合成序列。报告依据真实fixture和rows更正解释，保留原件。

## 公平性与预算

新旧后端会影响浮点计算和生成调度，即使模型、题目与解码参数相同，完整轨迹也可能变化。各seed内AM/GM使用同一后端，跨seed统计披露版本。每环境12小时软预算下，提速可增加完成轮数，整run墙钟还取决于早停、评测与初始化。

当时完整训练轮的提速尚待保存后的测量。生成4.46倍、反向1.44倍和正式验证约2.6倍分别具有各自范围，不能据此宣称整个实验4.46倍。

## 原始证据入口

仓库 https://github.com/sheepyedu/GRPO-use-arithmetic-mean-or-geometric-mean ，效率实现提交8df4118，正式初始验证记录提交408238d。本机仓库及远端路径由当前项目记录确定。

- docs/efficiency-v4.md
- reports/efficiency/seed17-events.json 与 seed17-stage-timing.json
- reports/efficiency/benchmarks.json 与 benchmark_sources/
- reports/efficiency/candidate-tests.txt
- reports/efficiency/formal-validation-timing.json
- reports/pai-seed29/latest-status.json

这些文件包含历史实际结果，当前作业状态需读取最新任务记录。此案例自身不触发服务器访问或训练派发。
