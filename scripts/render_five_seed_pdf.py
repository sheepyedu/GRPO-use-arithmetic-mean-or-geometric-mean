"""Render the five-seed report and panels as a printable black-typeface PDF."""
from pathlib import Path
import json
import os
import re
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "output/pdf/GRPO五种子实验汇报_黑体_20261118.pdf"
FONT = Path(os.environ.get("GRPO_HEITI_FONT", "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/53fe5be564086fefc7523ccd0a31200acf92e0e5.asset/AssetData/STHEITI.ttf"))
INK = "#192F42"
COLORS = {"am": "#146B85", "gm": "#B84A2C"}


def chart(curves, kind, destination):
    labels = {"validation": "主验证分数（%）", "reward": "训练采样平均奖励", "kl": "参考 KL（对数轴）"}
    fig, axes = plt.subplots(5, 2, figsize=(9, 10.4), sharex=True, sharey=True)
    for row, seed in enumerate((17, 29, 43, 71, 101)):
        for col, env in enumerate(("math", "mixed")):
            ax = axes[row, col]
            for op, style in (("am", "-"), ("gm", "--")):
                run = curves[str(seed)][f"formal-{env}-{op}"]
                if kind == "validation":
                    points = sorted((int(k), v*100) for k,v in run["validation"].items())
                else:
                    points = [(m["round"],m[kind]) for m in run["metrics"]]
                x, y = zip(*points)
                ax.plot(x,y,style,color=COLORS[op],label=op.upper(),linewidth=1.35,
                        marker="o" if kind=="validation" else None,markersize=2.8)
            if kind == "kl":
                ax.set_yscale("log")
            ax.set_title(f"Seed {seed}  {'纯数学' if env=='math' else '混合'}",loc="left",fontsize=10)
            ax.set_xlim(0,80)
            ax.set_xticks([0,20,40,60,80])
            ax.grid(axis="y",alpha=.22)
            ax.spines[["top","right"]].set_visible(False)
            if row == 4:
                ax.set_xlabel("训练轮次")
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc="upper right",bbox_to_anchor=(.97,.99),ncol=2,frameon=False)
    fig.text(.08,.963,labels[kind],fontsize=10)
    fig.subplots_adjust(left=.095,right=.98,top=.92,bottom=.056,wspace=.12,hspace=.55)
    fig.savefig(destination,dpi=175)
    plt.close(fig)


def main():
    pdfmetrics.registerFont(TTFont("Heiti",str(FONT)))
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=FONT).get_name(),
                         "axes.unicode_minus":False,"font.size":9})
    styles = {key: ParagraphStyle(key,fontName="Heiti",fontSize=size,leading=leading,
                               textColor=colors.HexColor(INK),spaceAfter=gap,wordWrap="CJK")
              for key,size,leading,gap in (("title",22,33,15),("heading",16,25,10),("body",10.2,15,6),("cell",8.1,12,0))}
    story = []

    def add(text, style="body"):
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)",r"\1",text).replace("**","").replace("`","")
        story.append(Paragraph(escape(clean),styles[style]))

    def table(lines):
        rows=[[Paragraph(escape(s.strip()),styles["cell"]) for s in row.strip("|").split("|")]
              for row in lines if not row.startswith("| ---")]
        n=len(rows[0])
        widths = {6: [65,82,89,71,82,112], 8: [33,49,58,58,58,72,75,98]}[n]
        el=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
        el.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DFEAF0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F6F8")]),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
        ]))
        story.extend((el,Spacer(1,11)))

    lines=(REPORTS/"five-seed-results.md").read_text().splitlines()
    i=0
    while i < len(lines):
        line=lines[i]
        if line.startswith("|"):
            start=i
            while i<len(lines) and lines[i].startswith("|"):
                i+=1
            table(lines[start:i])
            continue
        if line.startswith("# "):
            add(line[2:],"title")
            add("Qwen3-8B · 阿里云 PAI H20 · 2026-11-18")
        elif line.startswith("## "):
            if line=="## 算力与 token":
                story.append(PageBreak())
            add(line[3:],"heading")
        elif line.startswith("- [验证准确率]"):
            add("后附验证分数、逐轮训练奖励及参考 KL 曲线。验证包含初始、每10轮与最终验证。")
        elif line and not line.startswith("- ["):
            add(line)
        i+=1

    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#C9D8DF"))
        canvas.line(45,42,550,42)
        canvas.setFont("Heiti",9)
        canvas.setFillColor(colors.HexColor(INK))
        canvas.drawString(45,27,"GRPO 序列概率比对照")
        canvas.drawRightString(550,27,str(doc.page))

    curves=json.loads((REPORTS/"five-seed-curve-data.json").read_text())
    with TemporaryDirectory() as tmp:
        for kind,title in [("validation","验证曲线"),("reward","训练奖励曲线"),("kl","参考 KL 曲线")]:
            image=Path(tmp)/f"{kind}.png"
            chart(curves,kind,image)
            story.append(PageBreak())
            add(title,"heading")
            story.append(Image(str(image),width=501,height=579))
        doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,topMargin=51,bottomMargin=53,
                              leftMargin=45,rightMargin=45,title="GRPO 序列概率比五种子对照报告")
        doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    main()
