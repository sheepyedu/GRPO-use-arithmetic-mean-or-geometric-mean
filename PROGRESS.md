# 项目进展与经验

## 2026-11-15 · Qwen3-8B 实验方案发布

- 已完成完整协议 `docs/experiment_plan.md`、README 入口及文档检查指南，并在项目指南中统一官方主模型 `Qwen/Qwen3-8B`。保留双环境、AM/GM 序列概率比对照和五个配对 seed，共 20 个正式运行。
- 方案提交为 `7c07e1ff80402f652b346120c54fbca7791a1197`，已推送至 GitHub `main`。通过 `git ls-remote` 确认远端 SHA，并 fetch 后逐个比对方案、README、TEST、CLAUDE 与 AGENTS 链接内容。
- 实际检查包括修改前及提交前的 `git diff --check`、模型和阶段声明、矩阵与预算算术、表格列数、本地链接、LaTeX 定界符配对，以及 `AGENTS.md` 相对符号链接同步。另人工核对公式、数据隔离及统计约束。仓库尚无现成测试套件，训练与实验结果仍待后续工作。
- 遇到的问题是远端为空，`git fetch origin` 后仍没有 `origin/main`，按默认命令建立 worktree 失败。随后从本地 `main` 建立独立任务 worktree，检查远端仍为空后首次正常推送 `main`。
- 后续应先检查远端分支是否存在，再选择 worktree 起点。`.claude-manager/` 仅加入本地 Git exclude，避免工作区及运行时产物进入版本控制。`AGENTS.md` 保持指向 `CLAUDE.md` 的现有相对链接。
