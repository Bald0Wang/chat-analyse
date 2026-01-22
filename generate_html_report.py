#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML报告生成器
从 chat 文件夹中提取指定日期的消息并生成报告页面
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple



TECH_KEYWORDS = [
    "AI", "LLM", "GPT", "Claude", "Python", "JavaScript",
    "编程", "代码", "算法", "框架", "数据库", "API",
    "Docker", "Git", "Linux", "机器学习", "深度学习",
    "Prompt", "提示词", "微调", "训练", "推理",
]


def load_notes(notes_dir: Path, target_date: str) -> List[Dict[str, Any]]:
    notes = []
    for file_path in sorted(notes_dir.glob(f"{target_date}_*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        notes.append(parse_note(file_path.name, content))
    return notes


def parse_note(filename: str, content: str) -> Dict[str, Any]:
    lines = [line.rstrip() for line in content.splitlines()]
    title = ""
    sections: Dict[str, List[str]] = {}
    current_section = None

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue
        if current_section:
            sections[current_section].append(line)

    key_points = [line[2:].strip() for line in sections.get("关键讨论点", []) if line.strip().startswith("- ")]
    resources = [line[2:].strip() for line in sections.get("推荐资源", []) if line.strip().startswith("- ")]
    keywords_line = ""
    for line in sections.get("相关关键词", []):
        if line.strip():
            keywords_line = line.strip()
            break
    keywords = [kw.strip() for kw in keywords_line.split(",") if kw.strip()]

    qa_lines = [line for line in sections.get("问答精选", []) if line.strip()]
    summary = ""
    for line in sections.get("核心问题/分享", []):
        if line.strip():
            summary = line.strip()
            break

    return {
        "filename": filename,
        "title": title or filename,
        "summary": summary,
        "key_points": key_points,
        "resources": resources,
        "keywords": keywords,
        "qa_lines": qa_lines,
    }


def build_report_html(
    target_date: str,
    group_name: str,
    stats: Dict[str, Any],
    keywords: List[Tuple[str, int]],
    topics: List[Dict[str, Any]],
    resources: List[str],
    qa_items: List[str],
) -> str:
    total_notes = stats.get("total_notes", 0)
    total_keypoints = stats.get("total_keypoints", 0)
    total_resources = stats.get("total_resources", 0)
    total_qa = stats.get("total_qa", 0)

    keyword_items = "\n".join(
        [
            f'<span class="keyword">{word} <em>{count}</em></span>'
            for word, count in keywords
        ]
    ) or '<span class="keyword">暂无关键词</span>'

    topic_items = "\n".join(
        [
            f"""                <div class="topic-card">
                    <div class="topic-header">
                        <h3>{topic.get('title', '未命名话题')}</h3>
                    </div>
                    <p class="topic-summary">{topic.get('summary', '')}</p>
                    <div class="topic-meta">
                        <span>要点 {len(topic.get('key_points', []))} 条</span>
                        <span>资源 {len(topic.get('resources', []))} 个</span>
                    </div>
                    <div class="topic-points">
                        {"".join([f"<p>• {point}</p>" for point in topic.get("key_points", [])[:5]])}
                    </div>
                </div>"""
            for topic in topics
        ]
    ) or '<div class="topic-card">暂无话题</div>'

    resource_items = "\n".join([f"<li>{item}</li>" for item in resources]) or "<li>暂无资源</li>"
    qa_blocks = "\n".join([f"<p>{item}</p>" for item in qa_items]) or "<p>暂无问答精选</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_date} 群聊分析报告</title>
    <style>
        :root {{
            --cyber-pink: #ff1cf7;
            --cyber-blue: #00f0ff;
            --cyber-green: #27ff76;
            --cyber-yellow: #fff200;
            --dark-bg: #09090f;
            --darker-bg: #05050b;
            --card-bg: #161625;
            --text-primary: #ffffff;
            --text-secondary: #b8b8d8;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: "Courier New", monospace;
            background-color: var(--dark-bg);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}

        header {{
            background: linear-gradient(135deg, var(--cyber-pink), var(--cyber-blue));
            padding: 40px;
            margin-bottom: 36px;
            border: 8px solid var(--text-primary);
            box-shadow: 14px 14px 0 var(--cyber-green);
            position: relative;
        }}

        header::before {{
            content: "NEUBRUTAL CYBER REPORT";
            position: absolute;
            top: 12px;
            right: 24px;
            font-size: 2.4rem;
            font-weight: 700;
            opacity: 0.15;
            letter-spacing: 8px;
        }}

        h1 {{
            font-size: 3rem;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 4px 4px 0 var(--dark-bg);
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.4rem;
            color: var(--dark-bg);
            font-weight: 700;
            text-transform: uppercase;
        }}

        .stats-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}

        .stat-card {{
            background: var(--card-bg);
            border: 5px solid var(--text-primary);
            padding: 24px;
            transition: transform 0.2s ease;
        }}

        .stat-card:hover {{
            transform: translate(-4px, -4px);
            box-shadow: 10px 10px 0 var(--cyber-pink);
        }}

        .stat-number {{
            font-size: 3rem;
            font-weight: bold;
            display: block;
        }}

        .stat-label {{
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
        }}

        .section {{
            margin-bottom: 36px;
            padding: 28px;
            border: 5px solid var(--text-primary);
            background: var(--darker-bg);
        }}

        h2 {{
            font-size: 2.2rem;
            text-transform: uppercase;
            margin-bottom: 20px;
            border-bottom: 5px solid var(--cyber-pink);
            display: inline-block;
            padding-bottom: 6px;
        }}

        .activity-chart {{
            display: flex;
            align-items: flex-end;
            height: 260px;
            gap: 4px;
            padding: 16px;
            background: var(--card-bg);
            border: 3px solid var(--cyber-blue);
        }}

        .hour-bar {{
            flex: 1;
            background: linear-gradient(to top, var(--cyber-blue), var(--cyber-pink));
            min-height: 8px;
            position: relative;
        }}

        .hour-bar .tooltip {{
            position: absolute;
            bottom: 110%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--cyber-pink);
            color: var(--dark-bg);
            padding: 4px 8px;
            font-size: 0.75rem;
            opacity: 0;
            transition: opacity 0.2s ease;
            white-space: nowrap;
            z-index: 5;
        }}

        .hour-bar:hover .tooltip {{
            opacity: 1;
        }}

        .keyword-cloud {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            padding: 20px;
            background: var(--card-bg);
            border: 3px solid var(--cyber-green);
        }}

        .keyword {{
            padding: 8px 16px;
            border: 3px solid var(--cyber-yellow);
            text-transform: uppercase;
            font-weight: 700;
        }}

        .keyword em {{
            font-style: normal;
            color: var(--cyber-yellow);
            margin-left: 6px;
        }}

        .topic-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}

        .topic-card {{
            background: var(--card-bg);
            border: 4px solid var(--cyber-yellow);
            padding: 18px;
        }}

        .topic-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        .topic-score {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--cyber-pink);
        }}

        .topic-summary {{
            color: var(--text-secondary);
            margin-bottom: 10px;
        }}

        .topic-meta {{
            display: flex;
            gap: 12px;
            font-size: 0.9rem;
            color: var(--cyber-blue);
        }}

        .topic-points p {{
            margin-top: 6px;
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}

        .resource-list {{
            list-style: none;
            padding-left: 0;
            display: grid;
            gap: 8px;
        }}

        .resource-list li {{
            background: var(--card-bg);
            border: 3px solid var(--cyber-blue);
            padding: 10px 12px;
        }}

        .qa-section {{
            display: grid;
            gap: 10px;
        }}

        .qa-section p {{
            background: var(--card-bg);
            border: 3px solid var(--cyber-green);
            padding: 10px 12px;
            color: var(--text-secondary);
        }}

        footer {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--cyber-green), var(--cyber-blue));
            border: 5px solid var(--text-primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>群聊数据分析报告</h1>
            <p class="subtitle">{target_date} · {group_name}</p>
        </header>

        <div class="stats-overview">
            <div class="stat-card">
                <span class="stat-number">{total_notes}</span>
                <span class="stat-label">笔记数量</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{total_keypoints}</span>
                <span class="stat-label">关键讨论点</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{total_resources}</span>
                <span class="stat-label">推荐资源</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{total_qa}</span>
                <span class="stat-label">问答条目</span>
            </div>
        </div>

        <section class="section">
            <h2>热门关键词云</h2>
            <div class="keyword-cloud">
{keyword_items}
            </div>
        </section>

        <section class="section">
            <h2>推荐资源</h2>
            <ul class="resource-list">
{resource_items}
            </ul>
        </section>

        <section class="section">
            <h2>话题精华</h2>
            <div class="topic-grid">
{topic_items}
            </div>
        </section>

        <section class="section">
            <h2>问答精选</h2>
            <div class="qa-section">
{qa_blocks}
            </div>
        </section>

        <footer>
            <p>Generated by AI Analysis System</p>
            <p>数据来源: {group_name} | 报告日期: {target_date}</p>
        </footer>
    </div>
</body>
</html>
"""


def generate_report(target_date: str, notes_dir: Path, output_path: Path,
                    group_name: str = "聊天分析笔记") -> Path:
    notes = load_notes(notes_dir, target_date)

    if not notes:
        raise SystemExit("未找到目标日期的笔记")

    stats = {
        "total_notes": len(notes),
        "total_keypoints": sum(len(note.get("key_points", [])) for note in notes),
        "total_resources": sum(len(note.get("resources", [])) for note in notes),
        "total_qa": sum(len(note.get("qa_lines", [])) for note in notes),
    }

    keyword_counts = Counter()
    for note in notes:
        for kw in note.get("keywords", []):
            keyword_counts[kw] += 1
    if not keyword_counts:
        for note in notes:
            for term in TECH_KEYWORDS:
                if term in note.get("summary", ""):
                    keyword_counts[term] += 1
    keywords = keyword_counts.most_common(16)

    resources = []
    for note in notes:
        resources.extend(note.get("resources", []))
    resources = resources[:12]

    qa_items = []
    for note in notes:
        qa_items.extend(note.get("qa_lines", [])[:6])
    qa_items = qa_items[:12]

    html = build_report_html(
        target_date=target_date,
        group_name=group_name,
        stats=stats,
        keywords=keywords,
        topics=notes[:8],
        resources=resources,
        qa_items=qa_items,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成聊天分析HTML报告")
    parser.add_argument("--date", default="2026-01-14", help="目标日期 YYYY-MM-DD")
    parser.add_argument("--notes-dir", default="notes", help="笔记目录")
    parser.add_argument("--output", default="output.html", help="输出HTML文件")
    args = parser.parse_args()

    output_path = generate_report(
        target_date=args.date,
        notes_dir=Path(args.notes_dir),
        output_path=Path(args.output),
    )
    print(f"✅ 报告已生成: {output_path}")


if __name__ == "__main__":
    main()
