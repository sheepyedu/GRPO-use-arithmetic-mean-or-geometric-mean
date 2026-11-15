# 项目进展与经验

## 2026-11-15 · Qwen3-8B 实验方案发布

- 已完成完整协议 `docs/experiment_plan.md`、README 入口及文档检查指南，并在项目指南中统一官方主模型 `Qwen/Qwen3-8B`。保留双环境、AM/GM 序列概率比对照和五个配对 seed，共 20 个正式运行。
- 方案提交为 `7c07e1ff80402f652b346120c54fbca7791a1197`，已推送至 GitHub `main`。通过 `git ls-remote` 确认远端 SHA，并 fetch 后逐个比对方案、README、TEST、CLAUDE 与 AGENTS 链接内容。
- 实际检查包括修改前及提交前的 `git diff --check`、模型和阶段声明、矩阵与预算算术、表格列数、本地链接、LaTeX 定界符配对，以及 `AGENTS.md` 相对符号链接同步。另人工核对公式、数据隔离及统计约束。仓库尚无现成测试套件，训练与实验结果仍待后续工作。
- 遇到的问题是远端为空，`git fetch origin` 后仍没有 `origin/main`，按默认命令建立 worktree 失败。随后从本地 `main` 建立独立任务 worktree，检查远端仍为空后首次正常推送 `main`。
- 后续应先检查远端分支是否存在，再选择 worktree 起点。`.claude-manager/` 仅加入本地 Git exclude，避免工作区及运行时产物进入版本控制。`AGENTS.md` 保持指向 `CLAUDE.md` 的现有相对链接。

## 2026-11-15 · 首seed实施和校准

- 实现提交 `7c13333e9524c52aa8f5e321904ed5883199c5ec`，覆盖数据、严格奖励、AM/GM目标、单卡BF16 LoRA/vLLM睡眠切换训练器、两卡配对调度和9项测试。软件与数据策略见实施决策。正式完成0/20。
- Node01复用官方Qwen3-8B固定revision，9个模型文件已完成SHA256，清单保存在 `reports/model-manifest.json`。GSM8K/ARC-Easy/BoolQ固定revision经镜像站下载，两个训练环境各6000题、验证2570题、锁定测试6965题，全部划分内容hash核验通过。
- 用户分别明确授权停止原GPU1 Curator及GPU0 scnode01 RMBench。按UID和项目命令行匹配后停止，保留桌面和其他项目进程。两卡试运行实际加载成功。
- 两卡各48个验证题×8采样校准通过，三来源零方差组均低于95%退化门，截断率4.43%/3.65%。当前为独立纯数学试运行第1轮，尚无正式完成run。Notion根项目及首seed子项目已创建并更新任务清单。
- 原首次直连下载延迟、数据卡片序列化失败、目标精度测试失败及Python3.10不支持file_digest的管理脚本失败均保留。分别改用镜像、card.to_dict、输入dtype的长度log及分块SHA256后完成验证。
- 进一步收紧截断最终答案的完整性规则，使用已保存校准文本复评分，两卡均0条奖励差异。新规则在候选版本测试通过，正在运行的v4试运行源保持原样，后续正式版本以提交及源码清单固定。
