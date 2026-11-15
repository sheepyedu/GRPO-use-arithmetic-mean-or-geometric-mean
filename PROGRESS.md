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

## 2026-11-15 23:00 · 首轮闭环，目标继续

- 当前main实施提交 `858e2b3be890a51f3121acae9ce7617ff212f55a`，12项CPU测试通过，模型5个分片SHA256与官方固定revision的LFS摘要全部一致。后续正式实施候选为服务器 `candidates/v8`，当前仍处试运行，正式0/4，根计划0/20。
- 正在运行的原试运行版本为服务器根目录v4源码，manager PID110627，纯数学AM/GM PID110628/110629，UID1005，两者start_ticks2081029505。按实际 `pair.json` 和 `scripts/status.py` 获取后续阶段和身份，不用历史PID认定未来作业。唯一试运行目录 `/home/remoteadmin/grpo-sequence-ratio/artifacts/pilot-seed17-v1`，先两侧math各10轮，后两侧mixed各10轮。不能在已有目录重复派发。
- 两侧首轮512回答、两epoch及16次优化已完成，checkpoint-001原生CPU读取通过，各504个AdamW参数状态、1512个张量均有限，scheduler/RNG/调度状态齐全。原核验SSH超过50秒留124，读取实际回执确认已完成；另一个四线程只读核验也通过，无模型或optimizer重放。小回执见 `reports/pilot-first-checkpoint-verification.json`。
- 两侧已载入checkpoint-001 LoRA进入第2轮rollout，观测快照见 `reports/pilot-latest-status.json`，该文件含原观测时间。首轮耗时约20–21分钟，按200轮和顺序双环境，首seed正式4run粗略约一周量级，实际依后续吞吐及评价开销修正。用户已获告知，原预算保持。
- Native goal保持active，首seed四个正式run尚未完成。继续监测并修复真实故障，四条件试运行通过后固定共同配置，从原始Qwen3-8B重新初始化正式四run。正式阶段通过 `scripts/run_seed.py --phase formal`，200轮，数学配对完成才转混合。结束后用 `scripts/summarize_seed.py` 验证完整四run及检查点哈希，再提交固定Notion审核聊天。当前goal先完成首seed四run。
- Notion根项目 `4195e704-b6a8-8148-bbfc-d9b3c8240945`，子项目 `4195e704-b6a8-8130-8564-c2afff182ec4`。每完成一个任务及时勾选子页并同步父摘要，正式完成数仅按真实终态更新。候选源码和运行中旧试运行源码分别保存，不覆盖运行中版本。后续正式版本保留HF/vLLM logprob差异及严格截断解析的实施披露。

## 2026-11-15 23:13 · 第二轮持续推进

- 本轮按实际/proc核对原两训练进程和各自vLLM子进程仍live、身份匹配。两侧更新后LoRA已完成第二轮512条采样和冻结概率计算，进入第二轮AdamW优化；未出现FAILURE，正式完成仍0/4。最新观测时刻及身份、GPU/资源见 `reports/pilot-latest-status.json`。
- 对两侧已保存的首两轮共2048条回答，以候选v8严格截断解析复评分，奖励差异全部为0，未新增模型调用。回执 `reports/pilot-strict-reward-rescore.json`。Notion子页及根摘要已同步当前优化阶段，native goal保持active。
