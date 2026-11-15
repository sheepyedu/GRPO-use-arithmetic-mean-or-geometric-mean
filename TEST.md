# 实现检查

在项目根目录使用训练环境执行。

```sh
USE_TF=0 USE_FLAX=0 python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
git diff --check
```

2026-11-15 Node01实际19项通过。覆盖数字规范化、重复排除、配对顺序及混合32/16/16比例、三种答案解析、AM≥GM、全一比值的梯度一致、正负优势裁切、极端log ratio、变长mask及零奖励方差。小型随机初始化Qwen3加LoRA执行两次真实反向/AdamW更新，并验证行为概率冻结。该小模型检查在CPU运行。结果汇总另用临时合成文件验证四run配对差值、学习曲线面积、缺失run拒绝及检查点篡改拒绝，合成文件随测试临时目录删除。

第一次目标不等式检查发现长度log项采用float32导致float64测试的精度误差，已统一到输入dtype，重测通过。原失败日志保留。

Qwen3-8B正在进行四条件各10轮独立试运行，检查显存切换、LoRA同步、梯度、输出格式、长度及验证集零优势比例。共同配置冻结后再开始seed17的四个200轮正式run。正式运行通过最终checkpoint、三个锁定测试集和COMPLETE才计完成。

数据manifest保存来源revision、许可证卡片、排除数、划分内容哈希。数学题的数字及符号被保留，近重复要求数字列表相同且词bigram Jaccard≥0.9。验证与测试在下载后即固定，禁止以测试表现选题或调参。

AM/GM共享单一配置、训练入口和prompt调度，只有operator及输出标识不同。GPU与软件版本、原始失败、解析退化和backend logprob差异均需如实报告。

## 早停与PAI并行入口

新增成对平台期、min_delta、最少轮数、12小时预算、非有限验证值、ABORT及两真实CPU子进程同轮停止测试。结果汇总兼容完整200轮与有共同停止决定的提前终态，拒绝两侧轮数不一致。2026-11-15 PAI隔离环境实际19项全部通过，原始日志见 `reports/pai-seed17/setup-tests.txt`。CPU夹具只验证控制逻辑，正式结果按PAI实际运行及终态回执认定。

四卡入口为 `scripts/run_seed.py --config configs/pai-seed17.json --phase formal --parallel-environments`，输出根通过 `--output` 指定。每个环境有独立成对控制目录、checkpoint屏障及共同STOP_DECISION。Node01原试运行保留原 source 和停止回执。
