# 首 seed 实施决策

正式矩阵沿用 seed 17、29、43、71、101，双环境 AM/GM，共20run。用户最新要求在PAI GPU0–3同时完成seed17四run，再推进其余seed。每卡一个独立进程和优化器，从同一官方BF16模型初始化LoRA。

## 后端

已检查 verl 的 LoRA 文档与当前依赖。Node01现有环境未安装verl，当前主分支要求transformers 5.5.3以上，现有可用栈为torch2.10.0、transformers4.57.6、peft0.17.1、vLLM0.17.1。为复用现有环境，本实现采用Transformers/PEFT训练及vLLM单卡rollout，sleep期间切换显存。未修改既有环境的依赖。实际导入路径、版本和GPU记录在run环境清单中。

[verl LoRA文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) · [verl当前依赖](https://github.com/volcengine/verl/blob/main/requirements.txt)

## 共同配置

配置初始值见 `configs/seed17.json`，rank16/alpha32、BF16、七类attention/MLP线性模块、dropout0。每轮64题×8回答，两个优化epoch，序列微批1、优化minibatch64、每epoch8次更新。训练率1e-5、epsilon0.2、beta0.01、优势delta1e-6、梯度范数裁切1。两组只改变AM/GM算子。

每轮先用更新前HF策略计算并冻结行为分数，用同一模型禁用adapter计算初始参考策略。vLLM采样原始logprob保留，记录与HF行为分数的数值差异。该后端差异是实现局限，试运行需要检查其量级。初始模型无量化，不进行SFT。

## 完成与失败

正式run达到200轮上限或产生已验证的成对早停终态后，完成该终态checkpoint的三个锁定测试集评价及完整结果保存，才记完成。实际轮数、停止原因与最大预算分别报告。验证、试运行、代码发布和启动均单列。首seed完成后保留0/20至4/20的真实进度，不提前启动下一seed。

退出、OOM、非有限损失和解析退化保存原日志及FAILURE，后续恢复使用版本化新attempt，不能覆盖旧失败。正式运行的自动重试尚未启用。检查点保留LoRA、AdamW和RNG；当前实现遇失败由管理Agent核查后处理。

## 2026-11-15 用户新增早停及H20重新训练

用户授权停止Node01并释放本作业显存，在阿里云PAI共享原始模型 `/mount-data/Models/Qwen3-8B` 重新训练。GPU0/1分别为纯数学AM/GM，GPU2/3分别为混合AM/GM。Node01已提交的两轮试运行检查点与中断记录留作历史证据。PAI重新下载同revision数据并验证相同划分内容哈希。

每对AM/GM同步到每轮完整checkpoint后作出共同决定。每10轮验证，至少20轮；两侧都连续两次未比各自最佳值提高0.005时，成对停止。纯数学使用GSM8K验证正确率，混合使用0.5/0.25/0.25固定权重验证正确率。另设每对环境12小时软上限，包含加载、训练及周期验证，在共同完整轮次边界触发，可优先于20轮下限；最终验证和锁定测试另计。200轮保留为最大上限。

早停最终使用共同停止轮次的checkpoint，不根据测试分数挑选。两侧轮数及停止决定必须一致。父控制器或伙伴故障写ABORT及原真实失败，不能作为成功早停。结果清单保留实际/计划轮数、停止原因、共同决定、生成token和GPU小时。不同环境允许停止于不同轮数，跨环境按实际预算解释。

PAI入口配置 `configs/pai-seed17.json`，独立CPFS环境复用现有torch2.10/transformers4.57.6/vLLM0.17.1并增加PEFT0.17.1、datasets4.6.1。源与大产物位于 `/mount-data/GRPO-Sequence-Ratio`，数据位于 `/mount-data/Datasets/GRPO-Sequence-Ratio/data-v1`。四卡入口增加 `--parallel-environments`。Node01已验证资源与反向/检查点流程，按用户最新直接训练要求，PAI从正式入口执行。
