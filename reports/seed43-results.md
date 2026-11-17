# seed43 四组 GRPO 结果

四个正式run于2026-11-17北京时间15:22:11结束，数学60轮、混合70轮，均按原paired_plateau规则成对停止。原四driver、共享controller和wrapper均gone，两parent均[0,0]、wrapper0。15:30:12实测GPU0–3显存与利用率均0，15:36:59必要续接资源检查再次确认四卡compute-app为空，GPU4–7其他项目保留。15:33:32独立CPU验收exit0，每run最终锁定测试6965题。

| 条件 | 轮数/更新 | GSM8K | ARC-Easy | BoolQ | 主指标 | 训练输出token | 评测输出token | 输入token按回答计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math AM | 60/960 | 94.8446% | 71.1700% | 84.2202% | 94.8446% | 19,099,197 | 10,278,595 | 6,013,679 |
| math GM | 60/960 | 94.5413% | 85.9428% | 86.3914% | 94.5413% | 19,202,518 | 10,170,416 | 6,013,679 |
| mixed AM | 70/1120 | 93.9348% | 98.2744% | 90.0000% | 94.0360% | 18,006,110 | 11,768,642 | 7,265,251 |
| mixed GM | 70/1120 | 93.1008% | 98.2323% | 89.6330% | 93.5168% | 18,316,971 | 12,197,737 | 7,265,251 |

数学GM−AM为-0.3033个百分点，混合为-0.5192个百分点。数学主指标为GSM8K，混合为0.5×GSM8K＋0.25×ARC-Easy＋0.25×BoolQ，测试分母1319/2376/3270。仅本seed配对描述，五seed统计待71/101完成。

训练输出74,624,796，验证输出34,010,995，最终测试输出10,404,395，总输出119,040,186 token。按回答重复输入26,557,860，输入加输出145,598,046。训练prompt仅计一次的总输入口径为14,420,980。逐读取260份训练rollout、30份验证预测和4份最终测试预测。输出含thinking，物理缓存/prefill及HF重复前反向token处理量未测。

四run累计分配47.307 GPU小时，含最终测试。数学成对停止预算时间40873.8秒，混合42626.4秒。数学bad_checks2/2，混合3/2。提升阈值0.5个百分点使用原停止参考，末轮最大得分与停止参考分别记录。最终评测使用最后共同checkpoint。

四份最终LoRA各504张量与AdamW各1512张量均实际CPU读取有限，scheduler960/1120与updates一致，Python/NumPy/Torch/CUDA RNG及数据调度存在。原生汇总器检查最终COMMITTED全部文件SHA、配对停止、验证序列和连续训练轮次通过。未执行optimizer恢复或重放。

## 数值与限制

训练metrics全有限。四组最大逐轮KL如下，完整状态验收不替代优化稳定性判断。

| run | 轮次 | 最大KL | 同轮loss |
|---|---:|---:|---:|
| formal-math-am | 51 | 0.007370540 | 0.000038917 |
| formal-math-gm | 51 | 0.005618455 | 0.000005419 |
| formal-mixed-am | 58 | 0.014161750 | -0.000425318 |
| formal-mixed-gm | 2 | 0.028012303 | -0.000190504 |

source-v4冻结18科学源码文件复核一致，配置与seed29只改变seed字段，21项测试和既有H20工程核验沿用。首seed17使用source-v2旧后端，后续seed使用source-v4，跨seed比较需披露后端及相同时间预算下训练暴露差异。当前完整验收12/20，剩余seed71/101和五seed汇总待完成。

此前启动metadata读取器因初始化source_manifest尚未生成而失败，R2修正缺席字段处理，原失败保留，科学派发仅一次。本次终态KL辅助读取的glob意外匹配formal-math-control，管理命令exit1后改为四个明确run目录取得结果，无科学动作。

完整资产留在CPFS `/mount-data/GRPO-Sequence-Ratio/runs/seed43`。数学最终LoRA位于两侧`checkpoint-060`，混合位于两侧`checkpoint-070`，配原始`/mount-data/Models/Qwen3-8B`使用。

远端CPU验收 `acceptance-v1/RESULT.json` SHA256 `4d45d7daddb16b58f92e72ca1b206c5ae607295556887beb8b02ac62dd86ccd5`。本机 `reports/pai-seed43/final-verification.json`、`final-verification-receipt.json`、`latest-status.json`、`mixed-terminal.json`与`reports/seed43-summary.json`提供小证据。执行方自查已完成，固定审核单另附，根任务由杨洋审核。
