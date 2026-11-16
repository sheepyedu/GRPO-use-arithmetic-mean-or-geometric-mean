# 首seed耗时与框架提速

2026-11-16按用户要求分析seed17并优化后续seed。科学代码原版source-v2及seed17产物保留，后续采用版本化source-v4，从原始Qwen3-8B开始。GPU0–3用于四条件并行。

## 时间实际花费

按events.jsonl相邻事件时间区间加总，四run共约53.04 GPU小时。

| 原日志区间 | GPU小时 | 占比 |
|---|---:|---:|
| optimizer及相邻保存/切换 | 18.67 | 35.20% |
| rollout采样 | 16.59 | 31.29% |
| evaluation验证与测试 | 10.62 | 20.02% |
| frozen_scores及首个微批 | 6.57 | 12.39% |
| 配对等待 | 0.54 | 1.01% |
| 初始化 | 0.05 | 0.09% |

旧事件在每个optimizer step后写入，首微批反向计入前一个frozen_scores区间，最后一个step之后的检查点与唤醒归入optimizer。该表反映区间归属，精确纯CUDA kernel耗时未采集。新版本新增sleep_and_offload、optimization、checkpoint、wake_and_adapter事件，便于后续定位。

## 已采纳的修改

1. vLLM开启CUDA Graph，请求槽32改64，批token预算仍4096。48个固定保存prompt、greedy、上限256token的两次工程短测，原1073/1122 token/s，新4672/5112 token/s，合并约4.46倍。初始加载和编译从22.56秒增加到71.66秒。已有LoRA加载、sleep/wake及其前后输出一致性通过，两个测试进程真实退出0。
2. 总序列长度不超过1536token时关闭HF梯度检查点，更长序列启用原重计算模式。8条保存序列第二组热测反向3.421→2.379秒，约快1.44倍、耗时降低30.5%，峰值分配约19.3→32.3GiB。分数和梯度实测完全一致。1536token合成边界检查，峰值35.49GiB，反向0.987→0.651秒，分数和梯度一致。该边界夹具为128个prompt token与1408个生成token，原脚本RESULT的通用scope文字不准确，rows中sequences=1与tokens=1408为实际数量。
3. 冻结行为/参考分数时跳过未使用的熵，训练时复用logsumexp结果。策略分数与梯度数学表达式保持一致，新增回归测试核对。
4. 诊断标量先detach后留在GPU，每轮集中汇总，减少每条回答多次GPU到CPU同步。optimizer step数、minibatch64、microbatch1和两epoch保持原配置。
5. 训练与生成缓存固定在CPFS，实际run配置与源码manifest分别存档。

## 未采纳的候选

变长序列padding批2/4的工程测试比逐条计算慢，梯度相对L2差异约9.9%，恢复逐条microbatch1。词表统计torch.compile热测收益较小，梯度相对差异约1.06%，最终框架采用eager词表统计。首次生成测试脚本缺少spawn main guard而失败，原日志和exit1保留，r2修正后两个配置真实退出0。以上均为工程核验，未执行optimizer step，也未计入科学run结果。

## 验证与解释范围

原框架19项测试通过。最终候选21项测试通过，包含分数、熵、梯度一致与短长序列检查点切换。工程短测与完整正式训练的长度、并发和LoRA阶段不同，4.46倍表示生成短测联合设置收益，无法分别归因于CUDA Graph和槽数。生成结果token总量在12181–12215间变化，表明调度与浮点数值会影响轨迹，后续跨seed报告须披露后端版本。优化后的四条件在同一版本下配对。

按旧区间占比和短测速率估算，整体计算吞吐有望约1.7–2.0倍，实际首完整轮耗时尚待验证。每环境12小时软预算继续使用，更快框架可能在同一预算内完成更多轮。整体run结束时间还取决于共同早停和最终评测。

所有模型、任务、奖励、优势、AM/GM目标、KL、学习率、每轮64×8、16次optimizer、解码长度和最终锁定测试保持原科学设置。后续seed29/43/71/101依次四条件运行，终态后按固定审核流程提交。

原始数据为reports/efficiency/seed17-events.json、seed17-stage-timing.json、benchmarks.json、candidate-tests.txt、source-v4-manifest.json和seed29-preflight.json。远端工程日志保留在项目efficiency-*目录，source-v3为候选试验材料，source-v4为已验证正式源。

## seed29正式入口实测

2026-11-16 14:45四个run各2570题初始验证完成，进入第1轮rollout。原seed17对应775.94/789.00/807.82/780.59秒，新seed29对应305.05/304.20/303.00/302.42秒，约2.54–2.67倍。同一固定验证集与基座，seed与后端版本不同，实际生成长度可以变化。完整训练轮速度仍待首轮保存后核对。

## 首完整训练周期实测 · 2026-11-16 15:25

已完成数学3轮、混合4轮，当前数学第4轮、混合第5轮优化，身份与提交匹配，无FAILURE。首完整rollout到下一轮rollout周期包含采样、概率评分、两epoch反向与16次更新、完整checkpoint、唤醒及配对等待。数学旧936.53–936.96秒，新676.16–676.81秒，约1.38–1.39倍；混合旧797.30–797.62秒，新595.84–597.18秒，约1.34倍。不同seed与生成长度构成实际工作量差异。此前1.7–2.0倍为粗略外推，首周期尚未达到该范围，当前报告采用实测1.34–1.39倍，稳态整轮速度继续积累。

15:26首轮四份checkpoint-001文件哈希与COMMITTED一致，LoRA各504张量、AdamW各1512张量有限，scheduler16、Python/NumPy/Torch/CUDA RNG及调度状态完整，CPU原生读取验证通过。证据reports/pai-seed29/first-state-verification.json。
