"""Build the final paired report from accepted runs and read-only curve data."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import mean, median, stdev
from xml.sax.saxutils import escape
import csv
import io
import json
import math

REPORTS = Path(__file__).resolve().parents[1] / "reports"
SEEDS = (17, 29, 43, 71, 101)
ENVIRONMENTS = ("math", "mixed")
OPERATORS = ("am", "gm")
TZ = timezone(timedelta(hours=8))


def read(path):
    return json.loads((REPORTS / path).read_text())


def paired_stats(values):
    avg = mean(values)
    n = len(values)
    flips = [abs(mean(sign * value for sign, value in zip(signs, values)))
             for signs in product((-1, 1), repeat=n)]
    half_width = 2.7764451052 * stdev(values) / math.sqrt(n)  # t(4), 97.5%
    return {
        "mean_pp": avg, "median_pp": median(values), "sd_pp": stdev(values),
        "t95_ci_pp": [avg - half_width, avg + half_width],
        "exact_two_sided_sign_flip_p": sum(x >= abs(avg) - 1e-12 for x in flips) / len(flips),
        "gm_wins": sum(x > 0 for x in values), "n": n,
        "mean_pp_source_v4_only": mean(values[1:]),
    }


def chart(curves, kind):
    label, axis, ticks = {
        "validation": ("Validation by seed", "Primary validation score (%)", [70, 75, 80, 85, 90, 95, 100]),
        "reward": ("Training reward by seed", "Mean rollout reward", [0.6, 0.7, 0.8, 0.9, 1]),
        "kl": ("Reference KL by seed", "KL (log scale)", [-4, -2, 0, 2, 4]),
    }[kind]
    if kind == "reward":
        low = min(m["reward"] for seed in curves.values() for run in seed.values() for m in run["metrics"])
        ticks = sorted({math.floor(low*10)/10, *ticks})
    if kind == "kl":
        min_kl = min(m["kl"] for seed in curves.values() for run in seed.values() for m in run["metrics"])
        assert min_kl > 0, "A log plot requires a strictly positive observed KL"
        ticks = sorted({2*math.floor(math.log10(min_kl)/2), *ticks})
    colors = {"am": "#176887", "gm": "#BC502E"}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1160" viewBox="0 0 1000 1160" role="img">',
           f"<title>{escape(label)}</title>", '<rect width="100%" height="100%" fill="white"/>']
    def text(x, y, value, size=12):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="#172B3A" font-family="Arial,sans-serif">{escape(str(value))}</text>')
    text(54, 35, label, 24)
    text(54, 60, axis)
    text(375, 60, "Round on x-axis  |  AM: blue solid  |  GM: orange dashed")
    lo, hi = ticks[0], ticks[-1]
    for row, seed in enumerate(SEEDS):
        for col, env in enumerate(ENVIRONMENTS):
            left, top = 64+col*491, 105+row*207
            width, height = 410, 136
            text(left, top-10, f"Seed {seed} · {env.capitalize()}", 15)
            def project(round_number, value):
                return left+width*round_number/80, top+height*(hi-value)/(hi-lo)
            for mark in range(0, 81, 20):
                px, _ = project(mark, lo)
                text(px-7, top+height+18, mark)
            for tick in ticks:
                _, py = project(0, tick)
                svg.append(f'<path d="M {left} {py:.2f} h {width}" stroke="#DCE3EA"/>')
                text(left-43, py+4, f"10^{tick}" if kind=="kl" else f"{tick:g}")
            for op in OPERATORS:
                run = curves[str(seed)][f"formal-{env}-{op}"]
                if kind == "validation":
                    points = [project(int(k), v*100) for k,v in sorted(run["validation"].items(), key=lambda x:int(x[0]))]
                else:
                    points = [project(m["round"], math.log10(m["kl"]) if kind=="kl" else m["reward"]) for m in run["metrics"]]
                pts = " ".join(f"{x:.2f},{y:.2f}" for x,y in points)
                dash = ' stroke-dasharray="5 3"' if op=="gm" else ""
                svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[op]}" stroke-width="1.7"{dash}/>')
                if kind == "validation":
                    svg.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{colors[op]}"/>' for x,y in points)
    svg.append('</svg>')
    (REPORTS / f"five-seed-{kind}.svg").write_text("\n".join(svg)+"\n")


def main():
    curves = read("five-seed-curve-data.json")
    runs, provenance = [], {}
    for seed in SEEDS:
        file = f"pai-seed{seed}/final-verification.json"
        record = read(file)
        accepted = record.get("result", record.get("RESULT.json", record))
        summary = accepted["summary"]
        assert summary["seed"] == seed and summary["complete_runs"] == 4
        provenance[str(seed)] = {
            "verified_at": accepted["verified_at"],
            "record": file,
            "record_sha256": sha256((REPORTS/file).read_bytes()).hexdigest(),
            "source_manifest_hash": summary["source_manifest_hash"],
        }
        tokens = {(v["environment"],v["operator"]):v for v in accepted["token_usage"]}
        for run in summary["runs"]:
            env, op, rounds = run["environment"],run["operator"],run["rounds"]
            curve = curves[str(seed)][f"formal-{env}-{op}"]
            metrics = curve["metrics"]
            test = run["test_metrics"]
            assert [v["round"] for v in metrics] == list(range(1,rounds+1))
            assert [v["updates"] for v in metrics] == list(range(16,16*(rounds+1),16))
            expected_rounds = sorted(set(range(0, rounds + 1, 10)) | {rounds})
            assert [int(v) for v in curve["validation"]] == expected_rounds
            assert [test[k]["count"] for k in ("gsm8k","arc","boolq")] == [1319,2376,3270]
            assert math.isclose(run["primary"],test["gsm8k"]["accuracy"] if env=="math" else test["mixed_weighted"])
            t = tokens[(env,op)]
            assert sum(p["output_tokens"] for p in t["phases"].values()) == t["output_tokens"]
            assert sum(p["input_tokens_per_sequence"] for p in t["phases"].values()) == t["input_tokens_per_sequence"]
            assert sum(m["tokens"] for m in metrics) == t["phases"]["training"]["output_tokens"]
            kl = max(metrics,key=lambda m:m["kl"])
            runs.append({
                "seed":seed,"source":"v2" if seed==17 else "v4","environment":env,"operator":op,
                "rounds":rounds,"updates":16*rounds,"stop_reason":run["stop_reason"],
                "primary_pct":run["primary"]*100,"gsm8k_pct":test["gsm8k"]["accuracy"]*100,
                "arc_pct":test["arc"]["accuracy"]*100,"boolq_pct":test["boolq"]["accuracy"]*100,
                "test_questions":6965,"allocated_gpu_hours":run["allocated_gpu_hours"],
                "input_tokens_per_sequence":t["input_tokens_per_sequence"],
                "input_tokens_per_prompt":t["input_tokens_per_prompt"],
                **{f"{k}_output_tokens":t["phases"][k]["output_tokens"] for k in ("training","validation","test")},
                "output_tokens":t["output_tokens"],
                "logical_tokens":t["output_tokens"]+t["input_tokens_per_sequence"],
                "peak_kl":kl["kl"],"peak_kl_round":kl["round"],"loss_at_peak":kl["loss"],
                "max_clip_fraction":max(m["clip_fraction"] for m in metrics),
                "median_zero_advantage_fraction":median(m["zero_advantage_fraction"] for m in metrics),
                "median_am_gm_log_gap":median(m["am_gm_log_gap"] for m in metrics),
            })
    index = {(r["seed"],r["environment"],r["operator"]):r for r in runs}
    assert len(runs)==len(index)==20
    pairs=[]
    for seed,env in product(SEEDS, ENVIRONMENTS):
        am,gm=(index[(seed,env,op)] for op in OPERATORS)
        assert am["rounds"]==gm["rounds"] and am["stop_reason"]==gm["stop_reason"]
        pairs.append({"seed":seed,"environment":env,"rounds":am["rounds"],"am_pct":am["primary_pct"],
                      "gm_pct":gm["primary_pct"],"gm_minus_am_pp":gm["primary_pct"]-am["primary_pct"]})
    stats={env:paired_stats([p["gm_minus_am_pp"] for p in pairs if p["environment"]==env]) for env in ENVIRONMENTS}
    totals={k:sum(r[k] for r in runs) for k in ("rounds","updates","test_questions","allocated_gpu_hours",
            "training_output_tokens","validation_output_tokens","test_output_tokens","output_tokens",
            "input_tokens_per_sequence","input_tokens_per_prompt","logical_tokens")}
    result={"provenance":provenance,"runs":runs,"pairs":pairs,"statistics":stats,"totals":totals}
    (REPORTS/"five-seed-summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(runs[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(runs)
    (REPORTS/"five-seed-runs.csv").write_text(stream.getvalue())
    for key in ("validation","reward","kl"):
        chart(curves,key)

    names={"math":"纯数学","mixed":"混合"}
    lines=["# GRPO 序列概率比五种子对照报告","",
           "20 组训练均完成，最后共同检查点的锁定测试与完整状态验收均通过。**本实验没有观察到 GM 稳定优于 AM。** 两个环境分别以种子内 GM−AM 为比较量。","",
           "| 环境 | GM−AM均值（百分点） | 95% t区间（百分点） | GM胜出 | 双侧符号翻转 p | 仅v4均值（百分点） |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for env,s in stats.items():
        l,h=s["t95_ci_pp"]
        lines.append(f"| {names[env]} | {s['mean_pp']:+.4f} | [{l:+.4f}, {h:+.4f}] | {s['gm_wins']}/5 | {s['exact_two_sided_sign_flip_p']:.4f} | {s['mean_pp_source_v4_only']:+.4f} |")
    lines+=["","每个环境独立使用5个配对差值。区间为小样本 t 区间，p 值在零效应下对5个差值作32种符号翻转，依赖符号可交换假设。两个环境共享种子，不将10对视为独立重复。样本不足以证明等效或排除小效应。","",
            "## 锁定测试与训练暴露","",
            "| Seed | 环境 | 共同轮数 | AM（%） | GM（%） | GM−AM（百分点） |",
            "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for p in pairs:
        lines.append(f"| {p['seed']} | {names[p['environment']]} | {p['rounds']} | {p['am_pct']:.4f} | {p['gm_pct']:.4f} | {p['gm_minus_am_pp']:+.4f} |")
    lines+=["","纯数学主指标为 GSM8K，混合主指标为 0.5×GSM8K + 0.25×ARC + 0.25×BoolQ。每组测试6,965题，共139,300个测试回答，题目在各运行间复用。每对训练题调度、起点和停止轮次匹配。每轮64题，各8个回答，16次优化更新。每10轮验证，至少20轮，双方连续两次未达0.5个百分点改善时早停，另有每对12小时软预算。种子17使用 source-v2，随后四个使用优化后的 source-v4。种子间的轮数、生成长度及后端有差异，去掉17的均值只用于敏感性检查。","",
            "## 算力与 token","",
            "| Seed | H20 GPU 小时 | 训练输出 | 验证输出 | 测试输出 | 按回答计输入 | 逻辑合计 | CPU验收（北京时间） |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for seed in SEEDS:
        group=[r for r in runs if r["seed"]==seed]
        sums=[sum(r[k] for r in group) for k in ["training_output_tokens","validation_output_tokens","test_output_tokens","input_tokens_per_sequence","logical_tokens"]]
        when=datetime.fromisoformat(provenance[str(seed)]["verified_at"]).astimezone(TZ)
        lines.append(f"| {seed} | {sum(r['allocated_gpu_hours'] for r in group):.3f} | "+" | ".join(f"{v:,}" for v in sums)+f" | {when:%m-%d %H:%M:%S} |")
    lines+=["",f"合计 **{totals['allocated_gpu_hours']:.3f} H20 GPU小时，{totals['logical_tokens']:,} 个逻辑 token**，其中输出 {totals['output_tokens']:,}。GPU小时按每组占用一张H20的运行墙钟相加，含初始验证和最终测试。按每个 prompt 计一次的输入共 {totals['input_tokens_per_prompt']:,}，GPU实际缓存命中、prefill计算量与重复前反向 token 未测。单组墙钟范围为5.799至13.383小时。训练使用同一台阿里云PAI节点的GPU0至GPU3，每组单卡。节点可见8张H20、184核CPU和1800GiB cgroup内存，模型、数据和完整证据留在CPFS。","",
            "## 曲线与机制","",
            "- [验证准确率](five-seed-validation.svg)包含初始验证、每10轮验证及非整十轮结束时的最后验证，最后共同检查点用于锁定测试。",
            "- [逐轮训练奖励](five-seed-reward.svg)显示采样奖励随轮数的变化。各轮采样题目不同，固定测试用于方法比较。",
            "- [逐轮参考 KL](five-seed-kl.svg)采用对数纵轴，保留瞬时峰值。完整逐轮数据、每次更新数和原文件哈希见 [曲线数据](five-seed-curve-data.json)。","",
            "设每个回答 token 的新旧策略对数概率差为 δ。AM 使用 mean(exp(δ))，GM 使用 exp(mean(δ))。同一回答、同一策略下，AM 的对数比不小于 GM，且 AM 的对数比梯度在 token 维度按 exp(δ) 加权，GM 为均匀权重。概率差较大的少数 token 因而可能更强地影响 AM 的序列比与进入裁剪区间的概率。优势的正负和裁剪方向共同决定最终更新。不同运行的参数与采样轨迹会分化，不能由同一回答上的不等式推出哪种方法得分更高。","",
            "逐轮汇总显示，20组中记录的最大裁剪比例为0.1953%，出现在数学AM。四个环境与方法组合中，先按运行取逐轮中位数再按种子取中位数，零优势比例为90.6%至92.2%，AM与GM的平均对数比差量级约为0.0006。任务奖励组内差异小、两种比率在绝大多数记录中远离裁剪边界，是差异较小的可能解释。这些汇总未测量单条序列的尾部影响，也不足以建立因果关系。","",
            "参考 KL 使用当前策略与基座参考策略的 token 对数概率差，与 AM/GM 使用的新旧策略比有不同用途。高 KL 尖峰支持定位异常轮次，现有记录不足以确定由哪个 token、哪条序列或哪次微批更新触发。各组最终检查点有限，过程仍出现明显波动。","",
            "| Seed | 环境 | 方法 | 轮数 | 峰值KL | 同轮损失 |",
            "| ---: | --- | --- | ---: | ---: | ---: |"]
    for r in sorted(runs,key=lambda r:r["peak_kl"],reverse=True)[:5]:
        lines.append(f"| {r['seed']} | {names[r['environment']]} | {r['operator'].upper()} | {r['peak_kl_round']} | {r['peak_kl']:.6g} | {r['loss_at_peak']:.6g} |")
    lines+=["","全部20组的峰值、裁剪比例和零优势占比见 [逐运行 CSV](five-seed-runs.csv)。种子71数学AM第40轮从0.937到0.942，名义改善0.005，原二进制浮点计算的目标为0.9420000000000001，使计数未重置。此处沿用实际原生停止结果。","",
            "复核入口为 [配对统计和来源哈希](five-seed-summary.json) 及各种子原始交付。 [17](seed17-results.md) · [29](seed29-results.md) · [43](seed43-results.md) · [71](seed71-results.md) · [101](seed101-results.md)",""]
    (REPORTS/"five-seed-results.md").write_text("\n".join(lines))
    print(json.dumps({"statistics":stats,"totals":totals},indent=2))


if __name__ == "__main__":
    main()
