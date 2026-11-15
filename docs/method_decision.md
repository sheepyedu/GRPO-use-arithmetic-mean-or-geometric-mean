# 首 seed 实施决策

正式矩阵沿用 seed 17、29、43、71、101，双环境 AM/GM，共20run。当前授权完成seed17四run，先纯数学配对，再混合配对。每卡一个独立进程和优化器，从同一官方BF16模型初始化LoRA。

## 后端

已检查 verl 的 LoRA 文档与当前依赖。Node01现有环境未安装verl，当前主分支要求transformers 5.5.3以上，现有可用栈为torch2.10.0、transformers4.57.6、peft0.17.1、vLLM0.17.1。为复用现有环境，本实现采用Transformers/PEFT训练及vLLM单卡rollout，sleep期间切换显存。未修改既有环境的依赖。实际导入路径、版本和GPU记录在run环境清单中。

[verl LoRA文档](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) · [verl当前依赖](https://github.com/volcengine/verl/blob/main/requirements.txt)

## 共同配置

配置初始值见 `configs/seed17.json`，rank16/alpha32、BF16、七类attention/MLP线性模块、dropout0。每轮64题×8回答，两个优化epoch，序列微批1、优化minibatch64、每epoch8次更新。训练率1e-5、epsilon0.2、beta0.01、优势delta1e-6、梯度范数裁切1。两组只改变AM/GM算子。

每轮先用更新前HF策略计算并冻结行为分数，用同一模型禁用adapter计算初始参考策略。vLLM采样原始logprob保留，记录与HF行为分数的数值差异。该后端差异是实现局限，试运行需要检查其量级。初始模型无量化，不进行SFT。

## 完成与失败

只有200轮训练、最终指定checkpoint、三个锁定测试集评价及完整结果保存后，正式run才记完成。验证、试运行、代码发布和启动均单列。首seed完成后保留0/20至4/20的真实进度，不提前启动下一seed。

退出、OOM、非有限损失和解析退化保存原日志及FAILURE，后续恢复使用版本化新attempt，不能覆盖旧失败。正式运行的自动重试尚未启用。检查点保留LoRA、AdamW和RNG；当前实现遇失败由管理Agent核查后处理。
