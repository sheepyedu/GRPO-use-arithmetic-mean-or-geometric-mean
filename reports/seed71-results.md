# seed71 四组 GRPO 结果

四个正式run于2026-11-18北京时间02:44:11结束，数学与混合均60轮、各960次更新，按原paired_plateau规则停止。两parent均[0,0]、wrapper0，原四driver、controller与wrapper均gone。03:36:50实测GPU0–3显存与利用率均0，03:39:42必要续接资源核查确认compute-app空。03:38:59独立CPU完整状态和逐token验收exit0，未恢复或执行优化器。

| 条件 | 轮数/更新 | GSM8K | ARC-Easy | BoolQ | 主指标 | 训练输出token | 评测输出token | 输入token按回答计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math AM | 60/960 | 93.7074% | 74.6212% | 85.1070% | 93.7074% | 19,410,800 | 10,426,194 | 6,017,831 |
| math GM | 60/960 | 94.5413% | 79.6296% | 85.5352% | 94.5413% | 19,192,323 | 10,166,480 | 6,017,831 |
| mixed AM | 60/960 | 93.2525% | 98.2744% | 88.1040% | 93.2208% | 15,887,091 | 10,791,166 | 6,396,807 |
| mixed GM | 60/960 | 93.3283% | 97.8535% | 89.8777% | 93.5969% | 15,948,762 | 10,804,562 | 6,396,807 |

数学GM−AM为+0.8340个百分点，混合为+0.3761个百分点。数学主指标GSM8K，混合为0.5×GSM8K＋0.25×ARC-Easy＋0.25×BoolQ。各run最终测试6965题，分母1319/2376/3270。五seed统计仍待seed101完成。

训练输出70,438,976、验证输出31,691,788、最终测试输出10,496,614，总输出112,627,378 token。按回答计输入24,829,276，输入加输出137,456,654。训练prompt仅计一次的总输入口径13,668,784。逐读取240份训练rollout、28份验证预测及4份测试预测，输出含thinking。物理缓存/prefill及HF重复前反向token处理未测。

四run累计分配41.752 GPU小时，含最终测试。数学停止预算时间39029.4秒、混合34616.0秒，双方共同最后checkpoint-060用于锁定测试。原停止计数数学4/2，混合3/2。

四份最终LoRA各504张量、AdamW各1512张量均实际CPU读取有限。scheduler960、完整Python/NumPy/Torch/CUDA RNG及数据调度存在，COMMITTED全部文件SHA、配对停止、验证序列及60轮连续metrics由原生汇总器检查通过。

## 数值与实现局限

训练metrics均有限，数学GM出现较大的KL尖峰。完整状态通过及最终成绩应结合该训练异常解释。

| run | 轮次 | 最大KL | 同轮loss |
|---|---:|---:|---:|
| formal-math-am | 29 | 0.017348087 | 0.000132991 |
| formal-math-gm | 32 | 7359.608041868 | 73.596033285 |
| formal-mixed-am | 52 | 0.941915405 | 0.009210250 |
| formal-mixed-gm | 43 | 0.111035244 | 0.001123918 |

数学GM第32轮KL7359.608、loss73.596为原metrics实值，具体token和机制原因尚未定位。后续五seed分析需保留该异常，当前未重跑、删除或改写原科学运行。

数学第40轮AM从原停止参考0.937到0.942，名义提升0.5个百分点，但原Python比较`score >= previous + delta`计算0.937＋0.005为0.9420000000000001，实际0.942未达到阈值，bad_checks保持2。GM提升0.6个百分点、计数重置0。随后数学第50轮3/1、第60轮4/2共同停止。原生decision保留，边界证据`reports/pai-seed71/math040-threshold-boundary.json`，作为冻结实现的浮点边界局限披露。

source-v4冻结18科学源码文件实际SHA一致，配置仅seed改变。既有21项测试和H20工程核验复用。首seed17使用source-v2，后续29/43/71/101使用source-v4，跨seed统计需披露后端与同时间预算训练暴露差异。正式完整验收现16/20，seed101及五seed汇总待完成。

完整资产保留CPFS `/mount-data/GRPO-Sequence-Ratio/runs/seed71`。四组LoRA与完整状态位于各自 `checkpoint-060`，配原始 `/mount-data/Models/Qwen3-8B` 使用。

远端CPU验收 `acceptance-v1/RESULT.json` SHA256 `4cce712d72ea239e09816ffb54b579f4976619f7e1a32d9492ba2613ef923e98`。本机 `reports/pai-seed71/final-verification.json`、`final-verification-receipt.json`、`latest-status.json`、`math-terminal.json`、`mixed-terminal.json`和`terminal-source-resource-check.json`保存小证据。执行方自查完成，固定审核按notion-subtask-review另行提交，根任务由杨洋审核。
