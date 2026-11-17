# GRPO-use-arithmetic-mean-or-geometric-mean — 项目指南

> **重要：Claude 必须自主维护本文件。** 架构或约定变化时更新，保持简洁。

## Git 信息

- Remote: git@github.com:sheepyedu/GRPO-use-arithmetic-mean-or-geometric-mean.git
- 默认分支: main

## 任务生命周期

你收到任务后，按以下 9 步流程自主完成：

1. **领取任务** — 你已被分配任务，阅读本文件和项目代码理解上下文
2. **创建工作区**:
   - `git fetch origin`（如有 remote）
   - `git worktree add -b task-<简短描述> .claude-manager/worktrees/task-<简短描述> origin/main`
   - 进入 worktree 目录工作（后续所有操作在 worktree 中）
   - 如果 worktree 创建失败，直接在当前分支工作
3. **实现功能** — 编写代码，确保可运行
4. **提交代码** — `git add` + `git commit`，commit message 简洁描述改动
5. **Merge + 测试**:
   - `git fetch origin && git merge origin/main`（集成最新代码，如有 remote）
   - 运行测试（如有测试命令）
6. **自动合并到 main**（如有 remote）:
   - `git fetch origin main`
   - `git rebase origin/main`，如果冲突则自行 resolve
   - 如果成功：`git checkout main && git merge <task-branch> && git push origin main`
   - 如果这一步有任何失败，退回到步骤 5 重试
   - （纯本地项目跳过本步）
7. **标记完成** — 更新文档（必须在清理之前，防止进程被杀时状态丢失）
8. **清理** — 回到项目根目录:
   - `git worktree remove .claude-manager/worktrees/<worktree名>`
   - `git branch -D <task-branch>`
   - 如有 remote: `git push origin --delete <task-branch>`
9. **经验沉淀** — 在 PROGRESS.md 记录经验教训（可选）

### 冲突处理

rebase 发生冲突时：
1. 查看冲突文件: `git diff --name-only --diff-filter=U`
2. 逐个解决冲突
3. `git add <resolved-files> && git rebase --continue`
4. 如果无法解决: `git rebase --abort`，退回步骤 5

### 状态判断

- 通过 `git remote -v` 判断是否有 remote
- 有 remote → 必须完成步骤 6（merge + push）
- 无 remote → 跳过步骤 5 的 fetch、步骤 6 和步骤 8 的远程分支删除

## 文件维护规则

> **以下文件都由 Claude Code 自主维护，每次功能变更后必须同步更新。**

- **CLAUDE.md**（本文件）：架构、约定、关键路径变化时更新，只改变化的部分，保持简洁
- **AGENTS.md**（Codex 读取）：**与 CLAUDE.md 保持关键内容同步**——这是 CC/Codex coding 时的行为纪律：往其中一个写新内容时，把相同的意思也写进另一个（不要求逐字一致）。正常状态它是指向本文件的 symlink（改一处即同步），不要改成独立文件；若两者已是独立文件，不要用 symlink 覆盖已有内容，逐次同步意思即可
- **README.md**：面向用户的文档，功能、使用流程变化时同步更新，保持与实际代码一致
- **TEST.md**：测试指南，新增功能时同步添加测试用例和文档
- **PROGRESS.md**：见下方「经验教训沉淀」

## 测试规范

**开发时必须主动使用测试，不是事后补充！**

- **改代码前**：先跑测试，确认基线全绿
- **改代码后**：再跑一遍确认无回归
- **新增功能**：同步新增测试用例，更新 TEST.md
- **修 bug**：先写复现 bug 的测试（红），修复后确认变绿

## 经验教训沉淀

每次遇到问题或完成重要改动后，要在 PROGRESS.md 中记录：
- 遇到了什么问题
- 如何解决的
- 以后如何避免
- **必须附上 git commit ID**

**同样的问题不要犯两次！**

## 注意事项

- 在 worktree 中工作时，不要切换到其他分支
- 完成任务后确保代码可运行、测试通过

## 当前实验方案

- 完整协议位于 `docs/experiment_plan.md`，用户入口为 `README.md`，文档检查指南为 `TEST.md`。
- 主模型固定为官方 `Qwen/Qwen3-8B`，默认统一启用 thinking 模式；实施时固定 revision 并核验许可证、模板与后端兼容性。
- 保留纯数学与混合双环境、序列概率比 AM/GM 配对比较及五个 seed（17、29、43、71、101），共 20 个正式运行。AM 组不是标准 token 级 GRPO。
- 已实现 `src/{data,rewards,objectives,train}.py`、`scripts/run_seed.py` 和 `configs/seed17.json`。Node01现有torch2.10/transformers4.57.6/peft0.17.1/vLLM0.17.1用于BF16 LoRA单卡训练，后端选择见 `docs/method_decision.md`。
- 首seed17在PAI GPU0–3同时运行数学AM/GM和混合AM/GM。正式采用成对早停，最多200轮。2026-11-17 seed17/29/43共12/20已完整验收，seed71四run已完成初始验证并进入第1轮采样，seed101待续接。
- 当前服务器工作根 `/mount-data/GRPO-Sequence-Ratio`，官方模型 `/mount-data/Models/Qwen3-8B`，数据 `/mount-data/Datasets/GRPO-Sequence-Ratio/data-v1`。Node01历史试运行及停止证据保留原目录。仓库提交源码、文档和小摘要。
- 测试 `python -m unittest discover -s tests -v`，19项包含小型Qwen两次优化更新。修改后重新测试，原失败日志保留。

## 最新执行调整

- 2026-11-15用户要求停止Node01并在PAI从原始Qwen3-8B重新训练，GPU0/1数学AM/GM，GPU2/3混合AM/GM，seed17完成后继续29/43/71/101。
- 最新配置 `configs/pai-seed17.json`，`scripts/run_seed.py --parallel-environments` 同时启动两个配对控制器，每轮checkpoint屏障后共同决定继续或停止。
- 早停为验证间隔10、最小20、patience2、min_delta0.005及每对12小时软上限，最大200轮。记录真实轮数、共同停止原因及最终测试。Node01试运行证据独立保留。

- 稳定训练使用本聊天原生heartbeat `grpo-h20-30` 每1小时监督，Goal续接共用 `reports/pai-seed17/supervision.json` 的下一观测门。10:00仅切换需用户判断事项的决策权，其他授权工作和Goal持续完成。每项实际完成后更新Notion待办与百分比。

## 后续seed效率版本

2026-11-16用户要求先优化再继续后续seed。实际source-v4，CUDA Graph/64请求槽，1536token以内关闭梯度检查点，更长序列启用。microbatch1、科学参数及12小时预算保持，21项测试与H20工程核验通过。不同版本seed的统计须披露后端条件，证据docs/efficiency-v4.md。
