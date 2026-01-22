#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信聊天记录分析系统 - 主程序

功能：
1. 数据预处理
2. 话题分割
3. QA问答对识别
4. 重要度评估
5. 笔记生成

作者: AI Assistant
创建时间: 2026-01-13
"""

import sys
import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入模块
from modules.preprocessor import DataCleaner, clean_raw_data
from modules.llm_client import get_llm_client
from modules.topic_segment import TopicSegmenter, segment_topics
from modules.qa_detector import QADetector, detect_qa_pairs
from modules.importance_scorer import ImportanceScorer, calculate_importance
from modules.note_generator import NoteGenerator, generate_notes
from generate_html_report import generate_report


class ChatAnalysisSystem:
    """聊天记录分析系统"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        # 初始化各模块
        self.cleaner = DataCleaner(config_path)
        self.segmenter = TopicSegmenter(config_path)
        self.qa_detector = QADetector(config_path)
        self.scorer = ImportanceScorer(config_path)
        self.generator = NoteGenerator(config_path)
        
        # 数据存储
        self.raw_messages = []
        self.cleaned_messages = []
        self.topics = []
        self.qa_pairs = []
        self.scores = []
        self.notes = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}")
            return {}
    
    def run_full_pipeline(self, input_file: str, use_llm: bool = True,
                          save_intermediate: bool = True) -> Dict:
        """
        运行完整分析流程
        
        Args:
            input_file: 输入的JSONL文件路径
            use_llm: 是否使用LLM
            save_intermediate: 是否保存中间结果
            
        Returns:
            分析结果摘要
        """
        print("=" * 60)
        print("🚀 微信聊天记录分析系统")
        print("=" * 60)
        print(f"输入文件: {input_file}")
        print(f"使用LLM: {'是' if use_llm else '否'}")
        print("=" * 60)
        
        start_time = datetime.now()
        run_limits = self.config.get('run_limits', {})
        max_topics = run_limits.get('max_topics')
        max_notes = run_limits.get('max_notes')
        progress_every = run_limits.get('progress_every', 5)
        
        try:
            # 1. 数据预处理
            print("\n📋 步骤1: 数据预处理...")
            step_start = datetime.now()
            self.cleaned_messages = self.cleaner.clean_file(input_file)
            print(f"✅ 清洗完成: {len(self.cleaned_messages)} 条消息")
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")
            
            if save_intermediate:
                self._save_json('cleaned_messages.json', self.cleaned_messages)
            
            # 2. 话题分割
            print("\n📊 步骤2: 话题分割...")
            step_start = datetime.now()
            light_messages = self._build_light_messages(self.cleaned_messages)
            intent_results = []
            if use_llm:
                llm = get_llm_client()
                intent_results = llm.analyze_conversation_intent(
                    light_messages,
                    window_size=self.segmenter.batch_size
                )
            if intent_results:
                self.topics = self.segmenter.segment_with_intents(light_messages, intent_results)
            else:
                self.topics = self.segmenter.segment(light_messages, use_llm=use_llm)
            if max_topics and len(self.topics) > max_topics:
                print(f"   ⚠️ 话题过多({len(self.topics)})，按配置只保留前 {max_topics} 个")
                self.topics = self.topics[:max_topics]
            print(f"✅ 分割完成: {len(self.topics)} 个话题")
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")
            
            if save_intermediate:
                self._save_json('topics.json', self.topics) 
            
            # 3. QA问答对识别
            print("\n❓ 步骤3: QA问答对识别...")
            step_start = datetime.now()
            light_messages = self._build_light_messages(self.cleaned_messages)
            if use_llm and intent_results:
                self.qa_pairs = self.qa_detector.detect_with_intents(light_messages, intent_results)
            else:
                self.qa_pairs = self.qa_detector.detect(light_messages, use_llm=use_llm)
            print(f"✅ 检测完成: {len(self.qa_pairs)} 个QA对")
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")
            
            if save_intermediate:
                self._save_json('qa_pairs.json', self.qa_pairs)
            
            # 4. 重要度评估
            print("\n⭐ 步骤4: 重要度评估...")
            step_start = datetime.now()
            threshold_value = 4.0
            self.scorer.threshold = threshold_value
            qa_pairs_list = [self._filter_qa_pairs_for_topic(topic, self.qa_pairs) for topic in self.topics]
            self.scores = self.scorer.batch_score(
                self.topics, 
                qa_pairs_list,
                use_llm=use_llm
            )
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")
            
            # 统计
            stats = self.scorer.get_statistics(self.scores)
            print(f"✅ 评估完成:")
            print(f"   - 总话题数: {stats.get('total_topics', len(self.scores))}")
            print(f"   - 通过阈值: {stats.get('passed_topics', 0)} ({stats.get('pass_rate', 0)}%)")
            print(f"   - 平均分: {stats.get('average_score', 0)}")
            
            if save_intermediate:
                self._save_json('scores.json', self.scores)
            
            # 5. 笔记生成
            print("\n📝 步骤5: 笔记生成...")
            step_start = datetime.now()
            threshold = getattr(self.scorer, "threshold", 4.0)
            passed_topics = self.scorer.get_topics_by_threshold(self.scores, threshold)
            print(f"✅ 通过阈值(≥{threshold})的话题: {len(passed_topics)} 个")
            
            if passed_topics:
                # 为通过阈值的话题生成笔记
                note_results = []
                total_topics = len(self.topics)
                for i, topic in enumerate(self.topics):
                    score = self.scores[i]
                    if score.get('pass_threshold', False):
                        result = self.generator.generate(topic, score, self.qa_pairs, use_llm=use_llm)
                        note_results.append(result)
                        if max_notes and len(note_results) >= max_notes:
                            print(f"   ⚠️ 已达到笔记数量上限({max_notes})，提前结束生成")
                            break
                        if len(note_results) % progress_every == 0 or len(note_results) == 1:
                            print(f"   ✅ 进度 {len(note_results)}/{len(passed_topics)} - {result['filename']}")
                
                self.notes = note_results
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")
            
            # 计算耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 生成结果摘要
            result = {
                'status': 'success',
                'duration': round(duration, 2),
                'input_file': input_file,
                'total_messages': len(self.cleaned_messages),
                'total_topics': len(self.topics),
                'total_qa_pairs': len(self.qa_pairs),
                'topics_above_threshold': len(passed_topics),
                'notes_generated': len(self.notes),
                'statistics': {
                    'message_stats': self.cleaner.get_statistics(self.cleaned_messages),
                    'topic_stats': stats,
                    'qa_stats': self.qa_detector.get_qa_statistics(self.qa_pairs)
                },
                'output_dir': self.generator.output_dir
            }
            
            print("\n" + "=" * 60)
            print("✅ 分析完成!")
            print(f"📊 总耗时: {duration:.2f} 秒")
            print(f"💬 消息数: {len(self.cleaned_messages)}")
            print(f"📑 话题数: {len(self.topics)}")
            print(f"❓ QA对数: {len(self.qa_pairs)}")
            print(f"📝 生成的笔记: {len(self.notes)}")
            print(f"📁 笔记位置: {self.generator.output_dir}")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _save_json(self, filename: str, data: Any):
        """保存JSON文件"""
        output_dir = Path(self.config.get('paths', {}).get('processed_data', './data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 保存中间结果: {filepath}")
    
    def run_step_by_step(self, input_file: str, use_llm: bool = True):
        """
        分步运行（可交互选择）
        
        Args:
            input_file: 输入文件路径
            use_llm: 是否使用LLM
        """
        print("\n🔧 分步运行模式")
        print("1. 数据预处理")
        print("2. 话题分割")
        print("3. QA检测")
        print("4. 重要度评估")
        print("5. 笔记生成")
        print("0. 全部运行")
        
        step = input("\n请选择步骤: ").strip()
        
        if step == '0' or step == '':
            self.run_full_pipeline(input_file, use_llm)
        elif step in ['1', '2', '3', '4', '5']:
            self._run_single_step(int(step), input_file, use_llm)
        else:
            print("❌ 无效选择")
    
    def _run_single_step(self, step: int, input_file: str, use_llm: bool):
        """运行单个步骤"""
        steps = {
            1: ("数据预处理", lambda: self._run_step1(input_file)),
            2: ("话题分割", lambda: self._run_step2()),
            3: ("QA检测", lambda: self._run_step3()),
            4: ("重要度评估", lambda: self._run_step4()),
            5: ("笔记生成", lambda: self._run_step5())
        }
        
        if step not in steps:
            print("❌ 无效步骤")
            return
        
        name, func = steps[step]
        print(f"\n运行步骤 {step}: {name}")
        func()
    
    def _run_step1(self, input_file: str):
        """运行步骤1: 数据预处理"""
        self.cleaned_messages = self.cleaner.clean_file(input_file)
        print(f"✅ 完成: {len(self.cleaned_messages)} 条消息")
    
    def _run_step2(self):
        """运行步骤2: 话题分割"""
        if not self.cleaned_messages:
            print("❌ 请先运行步骤1")
            return
        light_messages = self._build_light_messages(self.cleaned_messages)
        llm = get_llm_client()
        intent_results = llm.analyze_conversation_intent(
            light_messages,
            window_size=self.segmenter.batch_size
        )
        if intent_results:
            self.topics = self.segmenter.segment_with_intents(light_messages, intent_results)
        else:
            self.topics = self.segmenter.segment(light_messages)
        print(f"✅ 完成: {len(self.topics)} 个话题")
    
    def _run_step3(self):
        """运行步骤3: QA检测"""
        if not self.cleaned_messages:
            print("❌ 请先运行步骤1")
            return
        light_messages = self._build_light_messages(self.cleaned_messages)
        llm = get_llm_client()
        intent_results = llm.analyze_conversation_intent(
            light_messages,
            window_size=self.qa_detector.batch_size
        )
        if intent_results:
            self.qa_pairs = self.qa_detector.detect_with_intents(light_messages, intent_results)
        else:
            self.qa_pairs = self.qa_detector.detect(light_messages)
        print(f"✅ 完成: {len(self.qa_pairs)} 个QA对")

    def _build_light_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        构建轻量消息结构，仅保留说话人和内容（减少LLM负载）
        """
        light = []
        for msg in messages:
            light.append({
                "sender_name": msg.get("sender_name", ""),
                "content": msg.get("content", "")
            })
        return light
    
    def _run_step4(self):
        """运行步骤4: 重要度评估"""
        if not self.topics:
            print("❌ 请先运行步骤2")
            return
        qa_pairs_list = [self._filter_qa_pairs_for_topic(topic, self.qa_pairs) for topic in self.topics]
        self.scores = self.scorer.batch_score(self.topics, qa_pairs_list)
        print(f"✅ 完成: {len(self.scores)} 个话题评分")
    
    def _run_step5(self):
        """运行步骤5: 笔记生成"""
        if not self.scores:
            print("❌ 请先运行步骤4")
            return
        self.notes = self.generator.batch_generate(self.topics, self.scores)
        print(f"✅ 完成: {len(self.notes)} 个笔记")

    def run_full_llm(self, input_file: str, save_intermediate: bool = True) -> Dict:
        """
        全量记录一次性送入LLM总结（高质量模型）
        """
        print("=" * 60)
        print("🚀 全量LLM汇总模式")
        print("=" * 60)
        print(f"输入文件: {input_file}")
        print("=" * 60)

        start_time = datetime.now()
        try:
            print("\n📋 步骤1: 数据预处理...")
            step_start = datetime.now()
            self.cleaned_messages = self.cleaner.clean_file(input_file)
            print(f"✅ 清洗完成: {len(self.cleaned_messages)} 条消息")
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")

            if save_intermediate:
                self._save_json('cleaned_messages.json', self.cleaned_messages)

            print("\n🧠 步骤2: 全量LLM汇总...")
            step_start = datetime.now()
            llm = get_llm_client()
            model = self.config.get('llm', {}).get('full_llm_model', 'doubao-seed-1-8-251228')
            stats = self._compute_activity_stats(self.cleaned_messages)
            
            prompt = self._build_full_llm_prompt(self.cleaned_messages)
            response = llm.chat([
                {"role": "system", "content": "你是群聊内容分析专家，请输出结构化总结。"},
                {"role": "user", "content": prompt}
            ], temperature=0.4, model=model)

            report_date = datetime.now().strftime('%Y-%m-%d')
            stem = Path(input_file).stem
            summary_dir = Path(self.config.get("note_generator", {}).get("output_dir", "./notes"))
            summary_path = summary_dir / f"summary_{stem}_{report_date}.md"
            summary_path.write_text(response, encoding="utf-8")

            html_path = Path(f"{stem}_{report_date}_output.html")
            html_content = response
            if not self._looks_like_html(response):
                report_title = self._build_report_title(input_file)
                html_content = self._convert_to_html_with_llm(response, model, report_title, stats)
            else:
                html_content = self._inject_stats_html(html_content, stats)
            html_path.write_text(html_content, encoding="utf-8")

            parsed_summary = self._parse_markdown_summary(response)
            summary_json_path = summary_dir / f"summary_{stem}_{report_date}.json"
            summary_json_path.write_text(
                json.dumps(parsed_summary, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            print(f"✅ 汇总完成: {summary_path} / {html_path}")
            print(f"   ⏱️ 耗时: {(datetime.now() - step_start).total_seconds():.1f}s")

            duration = (datetime.now() - start_time).total_seconds()
            return {
                'status': 'success',
                'duration': round(duration, 2),
                'input_file': input_file,
                'total_messages': len(self.cleaned_messages),
                'summary_path': str(summary_path),
                'summary_json_path': str(summary_json_path),
                'html_path': str(html_path)
            }
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}

    def _looks_like_html(self, text: str) -> bool:
        lowered = text.lstrip().lower()
        return lowered.startswith("<!doctype html") or lowered.startswith("<html")

    def _convert_to_html_with_llm(self, markdown_text: str, model: str,
                                  report_title: str, stats: Dict[str, Any]) -> str:
        """
        使用LLM将Markdown总结转换为HTML
        """
        llm = get_llm_client()
        stats_block = self._build_stats_prompt_block(stats)
        reference_css = self._get_reference_css()
        prompt = f"""
请将以下Markdown内容转换为完整HTML文档（包含 <html><head><body>）。
要求：保留层级结构与列表，不要遗漏内容。
风格要求：新布鲁托主义（Neubrutalism），高对比、厚边框、块状卡片、强烈色块；严格参考下方CSS风格与配色。
标题要求：主标题固定为“{report_title}”，副标题可包含生成时间或数据量。
布局要求：顶部标题区 + 关键指标横向卡片 + 数据分析板块 + 主题卡片网格 + 深度总结区；使用栅格布局，留白充足，避免呆板。
内容要求：标题清晰，列表项可读性高，卡片层级分明。
请新增“数据分析”板块，包含：活跃人数、活跃时段统计图（按小时柱状图即可）、摸鱼榜（废话榜，字数少于6）、硬核榜（非废话）。
{stats_block}

参考CSS（请保持风格一致，可按需简化但不要偏离配色/字重/边框风格）：
<style>
{reference_css}
</style>

Markdown内容：
{markdown_text}
"""
        return llm.chat([
            {"role": "system", "content": "你是HTML格式化专家，只输出HTML文本。"},
            {"role": "user", "content": prompt}
        ], temperature=0.2, model=model)

    def _build_stats_prompt_block(self, stats: Dict[str, Any]) -> str:
        if not stats:
            return ""
        active_users = stats.get("active_users", 0)
        fish_rank = stats.get("fish_rank", [])
        hardcore_rank = stats.get("hardcore_rank", [])
        hourly_counts = stats.get("hourly_counts", {})
        lines = [
            "统计数据（请在HTML中使用这些数据）：",
            f"- 活跃人数: {active_users}",
            "- 活跃时段（小时: 条数）: " + ", ".join(
                [f"{hour}: {count}" for hour, count in sorted(hourly_counts.items())]
            ),
            "- 摸鱼榜（用户: 次数）: " + ", ".join(
                [f"{item['name']}: {item['count']}" for item in fish_rank]
            ),
            "- 硬核榜（用户: 次数）: " + ", ".join(
                [f"{item['name']}: {item['count']}" for item in hardcore_rank]
            ),
        ]
        return "\n".join(lines)

    def _get_reference_css(self) -> str:
        return """
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Arial Black', Arial, sans-serif;
    background-color: #f0f0f0;
    padding: 20px;
    line-height: 1.6;
}

/* 标题区样式 */
.header {
    background-color: #ff6b6b;
    border: 4px solid #000;
    padding: 25px;
    margin-bottom: 25px;
    border-radius: 0;
}

.header h1 {
    font-size: 2.5rem;
    color: #000;
    margin-bottom: 12px;
}

.header p {
    font-size: 1.2rem;
    color: #333;
    font-weight: bold;
}

/* 关键指标区样式 */
.key-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 35px;
}

.metric-card {
    background-color: #4ecdc4;
    border: 4px solid #000;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-card:nth-child(2) {
    background-color: #ffe66d;
}

.metric-card:nth-child(3) {
    background-color: #ffd166;
}

.metric-card:nth-child(4) {
    background-color: #1a535c;
    color: #fff;
}

.metric-card h3 {
    font-size: 1.4rem;
    margin-bottom: 10px;
}

.metric-card .value {
    font-size: 2.2rem;
    font-weight: 900;
}

/* 数据分析区样式 */
.data-analysis {
    background-color: #fff;
    border: 4px solid #000;
    padding: 25px;
    margin-bottom: 35px;
}

.data-analysis h2 {
    font-size: 2.2rem;
    margin-bottom: 20px;
    border-bottom: 4px solid #000;
    padding-bottom: 10px;
}

.section {
    margin-bottom: 30px;
}

.section h3 {
    font-size: 1.5rem;
    margin-bottom: 15px;
    background-color: #ffe66d;
    display: inline-block;
    padding: 6px 12px;
    border: 2px solid #000;
}

/* 活跃时段柱状图 */
.activity-chart {
    display: flex;
    align-items: flex-end;
    height: 220px;
    gap: 6px;
    padding: 15px;
    background-color: #f0f0f0;
    border: 3px solid #000;
}

.chart-bar {
    flex: 1;
    background-color: #4ecdc4;
    border: 2px solid #000;
    position: relative;
    transition: background-color 0.2s;
}

.chart-bar:hover {
    background-color: #ff6b6b;
}

.chart-bar::after {
    content: attr(data-hour);
    position: absolute;
    bottom: -22px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.9rem;
    font-weight: bold;
}

.chart-bar .count {
    position: absolute;
    top: -22px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.8rem;
    font-weight: 900;
}

/* 排行榜样式 */
.rank-list {
    list-style: none;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
}

.rank-list li {
    background-color: #f7fff7;
    border: 2px solid #000;
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    transition: background-color 0.2s;
}

.rank-list li:hover {
    background-color: #e6f9e6;
}

.rank-list li:nth-child(odd) {
    background-color: #edf7f6;
}

.rank-list .user {
    font-weight: bold;
}

.rank-list .count {
    background-color: #ff6b6b;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 900;
}

/* 主题卡片区样式 */
.topic-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 25px;
    margin-bottom: 35px;
}

.topic-card {
    background-color: #fff;
    border: 4px solid #000;
    padding: 22px;
    transition: transform 0.2s;
}

.topic-card:hover {
    transform: translateY(-5px);
}

.topic-card h2 {
    font-size: 1.8rem;
    margin-bottom: 15px;
    background-color: #4ecdc4;
    display: inline-block;
    padding: 6px 12px;
    border: 2px solid #000;
}

.topic-card h3 {
    font-size: 1.3rem;
    margin: 12px 0;
    color: #ff6b6b;
    border-left: 6px solid #ff6b6b;
    padding-left: 10px;
}

.topic-card ul {
    margin-left: 25px;
    margin-bottom: 18px;
}

.topic-card ul li {
    margin-bottom: 10px;
    font-weight: 500;
}

.topic-card a {
    color: #1a535c;
    text-decoration: underline;
    font-weight: bold;
}

.topic-card a:hover {
    color: #ff6b6b;
}

/* 深度总结区样式 */
.summary {
    background-color: #ffe66d;
    border: 4px solid #000;
    padding: 25px;
}

.summary h2 {
    font-size: 2.2rem;
    margin-bottom: 18px;
    border-bottom: 4px solid #000;
    padding-bottom: 10px;
}

.summary p {
    font-size: 1.15rem;
    margin-bottom: 12px;
    font-weight: 500;
}
"""

    def _parse_markdown_summary(self, markdown_text: str) -> Dict[str, Any]:
        section_map = {
            "总览": "overview",
            "关键主题": "key_topics",
            "重要观点与共识": "viewpoints",
            "争议与分歧": "disputes",
            "问答精选": "qa_pairs",
            "可执行建议": "suggestions",
            "参考资源": "resources",
        }
        result = {
            "overview": "",
            "key_topics": [],
            "viewpoints": [],
            "disputes": [],
            "qa_pairs": [],
            "suggestions": [],
            "resources": [],
        }
        current_key = ""
        lines = markdown_text.splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            header = line.lstrip("#").strip()
            matched_section = False
            for label, key in section_map.items():
                if label in header:
                    current_key = key
                    matched_section = True
                    trailing = header.split(label, 1)[-1].strip(" ：:、.-")
                    if trailing:
                        if key == "overview":
                            result["overview"] = trailing
                        elif key == "qa_pairs":
                            result["qa_pairs"].append({"q": trailing, "a": ""})
                        else:
                            result[key].append(trailing)
                    break
            if matched_section:
                continue

            if current_key == "overview":
                if result["overview"]:
                    result["overview"] += " " + line
                else:
                    result["overview"] = line
            elif current_key in ("key_topics", "viewpoints", "disputes", "suggestions", "resources"):
                item = re.sub(r"^(\d+[\.\)、]|[一二三四五六七八九十]+[、\.]|[\-\•])\s*", "", line).strip()
                if item:
                    result[current_key].append(item)
            elif current_key == "qa_pairs":
                qa_match = re.match(r"^(Q|问|A|答)\s*[:：]\s*(.*)$", line)
                if qa_match:
                    tag = qa_match.group(1)
                    text = qa_match.group(2).strip()
                    if tag in ("Q", "问"):
                        result["qa_pairs"].append({"q": text, "a": ""})
                    else:
                        if result["qa_pairs"]:
                            result["qa_pairs"][-1]["a"] = text
                        else:
                            result["qa_pairs"].append({"q": "", "a": text})
                else:
                    qa_inline = re.match(r"^Q\s*[：:]\s*(.+?)\s*A\s*[：:]\s*(.+)$", line)
                    if qa_inline:
                        result["qa_pairs"].append({"q": qa_inline.group(1).strip(), "a": qa_inline.group(2).strip()})
                    else:
                        result["qa_pairs"].append({"q": line, "a": ""})
        return result

    def _full_llm_styles(self) -> str:
        return """
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Arial Black', Arial, sans-serif; background:#f0f0f0; padding:20px; line-height:1.6; }
.header { background:#ff6b6b; border:4px solid #000; padding:25px; margin-bottom:25px; }
.header h1 { font-size:2.5rem; color:#000; margin-bottom:12px; }
.header p { font-size:1.2rem; color:#333; font-weight:bold; }
.key-metrics { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:20px; margin-bottom:35px; }
.metric-card { background:#4ecdc4; border:4px solid #000; padding:20px; text-align:center; }
.metric-card:nth-child(2) { background:#ffe66d; }
.metric-card:nth-child(3) { background:#ffd166; }
.metric-card:nth-child(4) { background:#1a535c; color:#fff; }
.metric-card h3 { font-size:1.4rem; margin-bottom:10px; }
.metric-card .value { font-size:2.2rem; font-weight:900; }
.topic-cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:25px; margin-bottom:35px; }
.topic-card { background:#fff; border:4px solid #000; padding:22px; }
.topic-card h2 { font-size:1.8rem; margin-bottom:15px; background:#4ecdc4; display:inline-block; padding:6px 12px; border:2px solid #000; }
.topic-card ul { margin-left:18px; }
.topic-card li { margin-bottom:6px; }
</style>
"""

    def _compute_activity_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        user_set = set()
        fish_counts: Dict[str, int] = {}
        hardcore_counts: Dict[str, int] = {}
        hourly_counts: Dict[int, int] = {h: 0 for h in range(24)}
        for msg in messages:
            sender = msg.get("sender_name") or "Unknown"
            content = (msg.get("content") or "").strip()
            timestamp = msg.get("timestamp")
            user_set.add(sender)
            if timestamp:
                hour = datetime.fromtimestamp(timestamp).hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            if not content:
                continue
            if len(content) < 6:
                fish_counts[sender] = fish_counts.get(sender, 0) + 1
            else:
                hardcore_counts[sender] = hardcore_counts.get(sender, 0) + 1
        fish_rank = sorted(
            [{"name": name, "count": count} for name, count in fish_counts.items()],
            key=lambda item: item["count"],
            reverse=True
        )[:10]
        hardcore_rank = sorted(
            [{"name": name, "count": count} for name, count in hardcore_counts.items()],
            key=lambda item: item["count"],
            reverse=True
        )[:10]
        return {
            "active_users": len(user_set),
            "hourly_counts": hourly_counts,
            "fish_rank": fish_rank,
            "hardcore_rank": hardcore_rank
        }

    def _build_stats_html(self, stats: Dict[str, Any]) -> str:
        if not stats:
            return ""
        hourly_counts = stats.get("hourly_counts", {})
        max_count = max(hourly_counts.values()) if hourly_counts else 0
        bars = []
        for hour in range(24):
            count = hourly_counts.get(hour, 0)
            height = 0 if max_count == 0 else int((count / max_count) * 120)
            bars.append(
                f'<div class="hour-bar"><span>{hour:02d}</span>'
                f'<div class="bar" style="height:{height}px"></div>'
                f'<em>{count}</em></div>'
            )
        fish_items = "".join(
            [f"<li>{item['name']}: {item['count']}</li>" for item in stats.get("fish_rank", [])]
        ) or "<li>暂无</li>"
        hardcore_items = "".join(
            [f"<li>{item['name']}: {item['count']}</li>" for item in stats.get("hardcore_rank", [])]
        ) or "<li>暂无</li>"
        return f"""
<section class="section data-analysis">
  <h2>数据分析</h2>
  <div class="stats-row">
    <div class="stat-card">
      <h3>活跃人数</h3>
      <p class="stat-number">{stats.get('active_users', 0)}</p>
    </div>
  </div>
  <div class="hourly-chart">
    <h3>活跃时段（按小时）</h3>
    <div class="hour-bars">
      {''.join(bars)}
    </div>
  </div>
  <div class="rankings">
    <div class="rank-card">
      <h3>摸鱼榜（废话榜）</h3>
      <ol>{fish_items}</ol>
    </div>
    <div class="rank-card">
      <h3>硬核榜</h3>
      <ol>{hardcore_items}</ol>
    </div>
  </div>
</section>
"""

    def _stats_inline_styles(self) -> str:
        return """
<style>
.data-analysis .stats-row { display:flex; gap:20px; margin-bottom:24px; flex-wrap:wrap; }
.data-analysis .stat-card { background:#161625; border:3px solid #ffffff; padding:16px 20px; min-width:180px; }
.data-analysis .stat-number { font-size:32px; margin:8px 0 0; }
.hourly-chart { margin:24px 0; }
.hour-bars { display:grid; grid-template-columns:repeat(12, minmax(0, 1fr)); gap:12px; align-items:end; }
.hour-bar { display:flex; flex-direction:column; align-items:center; gap:6px; font-size:12px; }
.hour-bar .bar { width:100%; background:#00f0ff; border:2px solid #ffffff; }
.rankings { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:20px; }
.rank-card { background:#161625; border:3px solid #ffffff; padding:16px 20px; }
.rank-card ol { margin:10px 0 0 20px; }
</style>
"""

    def _build_report_title(self, input_file: str) -> str:
        """
        根据输入文件名生成报告标题
        """
        stem = Path(input_file).stem
        if not stem:
            return "群聊分析"
        return f"{stem} 群聊分析"

    def _inject_stats_html(self, html: str, stats: Dict[str, Any]) -> str:
        stats_html = self._build_stats_html(stats)
        styles = self._stats_inline_styles()
        if not stats_html:
            return html
        lower_html = html.lower()
        head_idx = lower_html.find("</head>")
        if head_idx != -1:
            html = html[:head_idx] + styles + html[head_idx:]
            lower_html = html.lower()
        insert_idx = lower_html.rfind("</body>")
        if insert_idx == -1:
            return html + styles + stats_html
        return html[:insert_idx] + stats_html + html[insert_idx:]
    def _build_full_llm_prompt(self, messages: List[Dict]) -> str:
        """
        构造全量LLM汇总提示词
        """
        lines = []
        for msg in messages:
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content", "")
            if not content:
                continue
            lines.append(f"{sender}: {content}")

        merged = "\n".join(lines)
        prompt = f"""
你是群聊内容分析专家，请基于以下完整聊天记录生成结构化总结报告。注意：聊天记录中的链接已被标准化为【描述】<URL>格式。

输出要求（Markdown）：
1. 总览（1段，50~120字）
2. 关键主题（3~8条，每条一句）
3. 重要观点与共识（3~8条）
4. 争议与分歧（如无则写“无明显争议”）
5. 问答精选（5条，Q/A 形式，必须来自原文语句）
6. 可执行建议（3~6条，具体可落地）
7. 参考资源（仅列出文中出现的链接，保留原URL）

规则：
- 不要编造未出现的信息
- 合并相近观点，避免重复
- 如某部分信息不足，写“暂无”

聊天记录：
{merged}
"""
        return prompt

    def _filter_qa_pairs_for_topic(self, topic: Dict, qa_pairs: List[Dict]) -> List[Dict]:
        """
        根据话题边界过滤QA对
        """
        if not qa_pairs:
            return []
        start_idx = topic.get('start_index', 0)
        end_idx = topic.get('end_index', -1)
        filtered = []
        for pair in qa_pairs:
            q_idx = pair.get('question_index')
            a_idx = pair.get('answer_index')
            if q_idx is not None and start_idx <= q_idx <= end_idx:
                filtered.append(pair)
            elif a_idx is not None and start_idx <= a_idx <= end_idx:
                filtered.append(pair)
        return filtered


def get_timestamp_range(target_date: str) -> tuple:
    """
    获取目标日期的时间戳范围
    """
    normalized_date = _normalize_date(target_date)
    try:
        date_obj = datetime.strptime(normalized_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"日期格式错误: {target_date}，应为 YYYY-MM-DD，例如 2026-01-20"
        ) from exc
    start_datetime = datetime(
        date_obj.year,
        date_obj.month,
        date_obj.day,
        0, 0, 0
    )
    end_datetime = datetime(
        date_obj.year,
        date_obj.month,
        date_obj.day,
        23, 59, 59
    )
    return int(start_datetime.timestamp()), int(end_datetime.timestamp())


def extract_messages_by_date(
    file_path: str,
    target_date: str,
    message_types: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    从JSONL文件中提取指定日期的消息
    """
    start_ts, end_ts = get_timestamp_range(target_date)
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                try:
                    data = json.loads(line.strip())
                    if data.get('_type') == 'message':
                        timestamp = data.get('timestamp', 0)
                        if start_ts <= timestamp <= end_ts:
                            if message_types is None or data.get('type') in message_types:
                                data['_line_number'] = line_number
                                messages.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []
    return messages


def save_messages_to_jsonl(
    messages: List[Dict[str, Any]],
    output_file: str
) -> bool:
    """
    将消息保存为JSONL格式
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for msg in messages:
                msg_copy = {k: v for k, v in msg.items() if k != '_line_number'}
                f.write(json.dumps(msg_copy, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False


def extract_daily_messages(
    input_file: str,
    target_date: str,
    output_file: Optional[str] = None,
    message_types: Optional[List[int]] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    提取指定日期的消息
    """
    if verbose:
        print(f"正在从 {input_file} 中提取 {target_date} 的消息...")
    try:
        messages = extract_messages_by_date(input_file, target_date, message_types)
    except ValueError as exc:
        print(f"错误: {exc}")
        return []
    if verbose:
        print(f"找到 {len(messages)} 条消息")
    if output_file:
        if save_messages_to_jsonl(messages, output_file):
            if verbose:
                print(f"消息已保存到 {output_file}")
        else:
            if verbose:
                print("保存失败")
    return messages


def get_file_date_range(file_path: str) -> Dict[str, str]:
    """
    获取文件中消息的日期范围
    """
    min_timestamp = None
    max_timestamp = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get('_type') == 'message':
                        timestamp = data.get('timestamp', 0)
                        if min_timestamp is None or timestamp < min_timestamp:
                            min_timestamp = timestamp
                        if max_timestamp is None or timestamp > max_timestamp:
                            max_timestamp = timestamp
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return {}
    if min_timestamp and max_timestamp:
        return {
            'earliest': datetime.fromtimestamp(min_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'latest': datetime.fromtimestamp(max_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'earliest_date': datetime.fromtimestamp(min_timestamp).strftime('%Y-%m-%d'),
            'latest_date': datetime.fromtimestamp(max_timestamp).strftime('%Y-%m-%d')
        }
    return {}


def _parse_message_types(raw_value: Optional[str]) -> Optional[List[int]]:
    if not raw_value:
        return None
    return [int(item) for item in raw_value.split(',') if item.strip()]


def _normalize_date(target_date: str) -> str:
    parts = target_date.strip().split("-")
    if len(parts) != 3:
        return target_date
    year, month, day = parts
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return target_date
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def _print_extraction_stats(messages: List[Dict[str, Any]]) -> None:
    if not messages:
        print("\n未找到符合条件的消息")
        return
    print(f"\n提取统计:")
    print(f"  总消息数: {len(messages)}")
    type_counts: Dict[Any, int] = {}
    for msg in messages:
        msg_type = msg.get('type', 'Unknown')
        type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
    print("  消息类型分布:")
    for msg_type, count in sorted(type_counts.items(), key=lambda item: str(item[0])):
        type_name = {
            0: '文本',
            1: '图片',
            5: '表情',
            80: '系统',
            99: '链接'
        }.get(msg_type, f'类型{msg_type}')
        print(f"    {type_name} (Type {msg_type}): {count} 条")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='微信聊天记录分析系统')
    parser.add_argument('input', nargs='?', help='输入的JSONL文件路径')
    parser.add_argument('--config', '-c', default='config.yaml', help='配置文件路径')
    parser.add_argument('--no-llm', action='store_true', help='不使用LLM（仅规则）')
    parser.add_argument('--step', '-s', type=int, choices=[1, 2, 3, 4, 5], help='运行单个步骤')
    parser.add_argument('--full-llm', action='store_true', help='全量记录一次性LLM汇总（默认模式）')
    parser.add_argument('--pipeline', action='store_true', help='使用标准分步流水线（非全量汇总）')
    parser.add_argument('--info', action='store_true', help='显示文件信息')
    parser.add_argument('--html-report', action='store_true', help='生成HTML报告')
    parser.add_argument('--report-date', help='HTML报告日期 YYYY-MM-DD（默认今天）')
    parser.add_argument('--html-output', help='HTML报告输出路径')
    parser.add_argument('--extract-date', help='先提取指定日期消息 YYYY-MM-DD')
    parser.add_argument('--extract-output', help='提取后的JSONL输出路径')
    parser.add_argument('--extract-types', help='提取的消息类型，逗号分隔，如 0,1')
    parser.add_argument('--extract-only', action='store_true', help='仅执行提取，不继续分析')
    
    args = parser.parse_args()
    
    # 如果指定了--info
    if args.info:
        if args.input:
            date_range = get_file_date_range(args.input)
            if date_range:
                print(f"📅 日期范围: {date_range['earliest_date']} - {date_range['latest_date']}")
            cleaner = DataCleaner(args.config)
            messages = cleaner.clean_file(args.input)
            stats = cleaner.get_statistics(messages)
            print(f"📊 文件统计:")
            print(f"   总消息数: {stats['total_messages']}")
            print(f"   参与人数: {stats['participant_count']}")
            print(f"   问题数: {stats['question_count']}")
            print(f"   日期范围: {list(stats['date_distribution'].keys())[:5]}...")
        else:
            print("❌ 请指定输入文件")
        return
    
    # 检查输入文件
    if not args.input:
        print("❌ 请指定输入文件")
        print("用法: python main.py <input_file> [--config config.yaml] [--no-llm]")
        print("示例: python main.py chat.jsonl")
        print("     python main.py chat.jsonl --no-llm")
        sys.exit(1)
    
    # 检查文件是否存在
    if not Path(args.input).exists():
        print(f"❌ 文件不存在: {args.input}")
        sys.exit(1)

    analysis_input = args.input
    if args.extract_date:
        message_types = _parse_message_types(args.extract_types)
        output_file = args.extract_output
        if not output_file:
            stem = Path(args.input).stem
            output_file = f"{stem}_{args.extract_date}.jsonl"
        extracted_messages = extract_daily_messages(
            input_file=args.input,
            target_date=args.extract_date,
            output_file=output_file,
            message_types=message_types
        )
        _print_extraction_stats(extracted_messages)
        analysis_input = output_file
        if args.extract_only:
            return
    
    # 初始化系统
    system = ChatAnalysisSystem(args.config)
    
    # 运行
    if args.step:
        system.run_step_by_step(analysis_input, not args.no_llm)
    elif args.full_llm or not args.pipeline:
        system.run_full_llm(analysis_input)
    else:
        result = system.run_full_pipeline(analysis_input, not args.no_llm)
        if args.html_report and result.get('status') == 'success':
            report_date = args.report_date or args.extract_date or datetime.now().strftime('%Y-%m-%d')
            notes_dir = Path(result.get('output_dir', './notes'))
            if args.html_output:
                output_path = Path(args.html_output)
            else:
                stem = Path(analysis_input).stem
                output_name = f"{stem}_{report_date}_output.html"
                output_path = Path(output_name)
            generate_report(report_date, notes_dir, output_path)
            print(f"✅ HTML报告已生成: {output_path}")


if __name__ == "__main__":
    main()
