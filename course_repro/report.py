"""Render a complete BIPIA course run as a self-contained classroom HTML page."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path

from .experiment import summarize


NAMES = {
    "baseline": "A 原始提示", "reminder": "B 显式提醒",
    "isolated": "C 角色隔离", "minimal_evidence": "D 隔离＋证据筛选",
}


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def select_paired_rows(rows: list[dict], conditions: tuple[str, ...]) -> list[dict]:
    """Choose one attack case shared by every arm, preferring differing outcomes."""
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("sample_type", "clean" if row.get("clean") else "attack") == "attack":
            key = (row.get("context_id"), row.get("attack_id"), row.get("position"))
            groups[key][row["condition"]] = row
    eligible = [(key, group) for key, group in groups.items()
                if all(condition in group for condition in conditions)]
    if not eligible:
        return []
    key, group = min(eligible, key=lambda item: (
        -len({bool(item[1][condition].get("attack_success")) for condition in conditions}),
        str(item[0]),
    ))
    return [group[condition] for condition in conditions]


def render_report(run_dir: Path) -> Path:
    """Reject incomplete runs; show only metrics derived from recorded responses."""
    if (run_dir / "status.txt").read_text(encoding="utf-8").strip() != "complete":
        raise ValueError("report requires a complete run")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise ValueError("metrics are incomplete")
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != config["planned_requests"]:
        raise ValueError("response count differs from planned requests")
    if set(config["conditions"]) != set(metrics["by_condition"]):
        raise ValueError("condition metrics are incomplete")
    summaries = summarize(rows)
    table_rows = []
    for condition in config["conditions"]:
        m = {**summaries[condition], **metrics["by_condition"][condition]}
        table_rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
            NAMES.get(condition, condition), str(m["n"]), _rate(m["asr"]),
            f"{m['clean_n']} / {_rate(m['clean_answer_hit'])}",
            f"{m['hard_benign_n']} / {_rate(m['hard_benign_answer_hit'])}",
            _num(m["avg_prompt_tokens"]), _num(m["avg_completion_tokens"]),
            _num(m["avg_latency_s"], 2), str(m.get("length_truncated_n", 0)),
        )) + "</tr>")
    detail_rows = []
    for condition in config["conditions"]:
        m = {**summaries[condition], **metrics["by_condition"][condition]}
        for field, label in (("by_attack", "攻击"), ("by_position", "位置")):
            for name, group in m.get(field, {}).items():
                detail_rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                    NAMES.get(condition, condition), label, name, str(group["n"]), _rate(group["asr"]),
                )) + "</tr>")
    examples = []
    for condition in config["conditions"]:
        group = [r for r in rows if r.get("condition") == condition
                 and r.get("sample_type", "clean" if r.get("clean") else "attack") == "attack"]
        for outcome in (True, False):
            match = next((r for r in group if r.get("attack_success") is outcome), None)
            if match:
                examples.append(
                    f"<article><span class='eyebrow'>{escape(NAMES.get(condition, condition))} · "
                    f"{'完整短语命中' if outcome else '完整短语未命中'}</span>"
                    f"<h3>{escape(str(match.get('attack_id', '')))} · {escape(str(match.get('position', '')))}</h3>"
                    f"<p><strong>问题：</strong>{escape(str(match.get('question', '')))}</p>"
                    f"<blockquote>{escape(str(match.get('response', '')))}</blockquote></article>"
                )
    paired = select_paired_rows(rows, tuple(config["conditions"]))
    paired_cards = []
    for row in paired:
        paired_cards.append(
            f"<article><span class='eyebrow'>{escape(NAMES.get(row['condition'], row['condition']))}</span>"
            f"<h3>{'命中' if row.get('attack_success') else '未命中'}</h3>"
            f"<blockquote>{escape(str(row.get('response', '')))}</blockquote></article>"
        )
    paired_title = (f"上下文 {escape(str(paired[0].get('context_id', '')))} · "
                    f"{escape(str(paired[0].get('attack_id', '')))} · "
                    f"{escape(str(paired[0].get('position', '')))}") if paired else "无配对攻击样本"
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>BIPIA 课程实验结果</title>
<style>
:root{{--bg:#0b1020;--card:#161e32;--line:#34415b;--text:#edf4ff;--muted:#a8b7cc;--cyan:#68e0da;--gold:#f1c777}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,-apple-system,sans-serif}}
main{{max-width:1180px;margin:auto;padding:52px 28px 90px}}.eyebrow{{color:var(--cyan);font-size:.78rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}
h1{{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1.15;margin:.2em 0}}h2{{font-size:1.65rem;margin:2.3em 0 .6em}}
.lede{{color:var(--muted);max-width:760px;font-size:1.1rem}}.chips{{display:flex;flex-wrap:wrap;gap:10px;margin:26px 0}}
.chip{{border:1px solid var(--line);border-radius:99px;padding:7px 13px;color:var(--muted);font-size:.87rem}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:16px;background:var(--card)}}table{{border-collapse:collapse;width:100%;min-width:940px}}
th,td{{text-align:left;padding:17px 16px;border-bottom:1px solid var(--line)}}th{{font-size:.76rem;letter-spacing:.05em;color:var(--cyan)}}tr:last-child td{{border:0}}
td:nth-child(3){{color:var(--gold);font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}}article h3{{margin:.5em 0}}article p{{color:var(--muted)}}
blockquote{{white-space:pre-wrap;overflow-wrap:anywhere;border-left:3px solid var(--cyan);margin:15px 0 0;padding:8px 12px;background:#0f1729}}
.note{{border-left:3px solid var(--gold);padding:12px 18px;background:var(--card);color:var(--muted)}}
</style></head><body><main>
<span class="eyebrow">BIPIA · COURSE STUDY</span><h1>间接提示词注入：{'两组' if len(config['conditions']) == 2 else str(len(config['conditions'])) + ' 组'}对照</h1>
<p class="lede">在相同上下文、攻击和注入位置上，比较{'、'.join(escape(NAMES.get(c, c)) for c in config['conditions'])}。</p>
<div class="chips"><span class="chip">任务 {escape(str(config['task']))}</span><span class="chip">模型 {escape(str(config['model']))}</span>
<span class="chip">随机种子 {escape(str(config['seed']))}</span><span class="chip">记录请求 {len(rows)}</span></div>
<h2>结果总表</h2><div class="tablewrap"><table><thead><tr><th>实验组</th><th>攻击 n</th><th>完整短语 ASR</th>
<th>官方无攻击 n / 答案命中</th><th>课程难例 n / 答案命中</th><th>平均输入 tokens</th><th>平均输出 tokens</th><th>平均延迟 s</th><th>截断回复 n</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div>
<p class="note">ASR 仅指完整目标短语命中；答案命中是参考字符串代理指标，不等于语义正确率，也不等于原论文 ROUGE-1。
难例为课程自编数据，不能视作官方基准。小样本差异不代表普遍安全性。</p>
<h2>攻击与注入位置</h2><div class="tablewrap"><table><thead><tr><th>实验组</th><th>分组</th><th>名称</th><th>n</th><th>完整短语 ASR</th></tr></thead>
<tbody>{''.join(detail_rows) or '<tr><td colspan="5">无分组数据</td></tr>'}</tbody></table></div>
<h2>同一攻击的配对回复</h2><p class="lede">{paired_title} · 问题：{escape(str(paired[0].get('question', ''))) if paired else '—'}</p>
<div class="grid">{''.join(paired_cards) or '<article>没有可比较的完整配对。</article>'}</div>
<h2>原始回复举例</h2><div class="grid">{''.join(examples) or '<article>此运行没有攻击回复可展示。</article>'}</div>
<h2>复核与复现</h2><p class="lede">完整配置、逐条回复和模型用量记录保存在同一运行目录。
请结合人工复核文件判断改写后的攻击是否被自动指标漏记。</p>
</main></body></html>"""
    destination = run_dir / "report.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        print(render_report(args.run_dir))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
