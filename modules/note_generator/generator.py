"""
笔记生成模块 - 生成器

功能：
1. 生成结构化Markdown笔记
2. 保存到文件和数据库

作者: AI Assistant
创建时间: 2026-01-13
"""

import json
import yaml
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# 导入LLM客户端
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.llm_client import get_llm_client


class NoteGenerator:
    """笔记生成器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化笔记生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.note_config = self.config.get('note_generator', {})
        
        # 获取配置
        self.output_dir = self.note_config.get('output_dir', './notes')
        self.filename_format = self.note_config.get('filename_format', '{date}_{topic_title}.md')
        self.template_config = self.note_config.get('template', {})
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # LLM客户端
        self.llm = None
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def _get_llm_client(self):
        """获取LLM客户端（延迟初始化）"""
        if self.llm is None:
            self.llm = get_llm_client()
        return self.llm
    
    def _append_log(self, label: str, content: str) -> None:
        debug = self.config.get('debug', {})
        if not debug.get('log_to_file', False):
            return
        log_path = debug.get('log_path')
        if not log_path:
            return
        try:
            from pathlib import Path
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat()
            with path.open('a', encoding='utf-8') as f:
                f.write(f"\n[{timestamp}] {label}\n{content}\n")
        except Exception:
            pass
    
    def generate(self, topic_info: Dict, importance_score: Dict,
                 qa_pairs: List[Dict] = None, use_llm: bool = True) -> Dict:
        """
        生成话题笔记
        
        Args:
            topic_info: 话题信息
            importance_score: 重要度评分结果
            qa_pairs: QA对列表
            use_llm: 是否使用LLM生成
            
        Returns:
            生成结果信息
        """
        # 1. 提取关键信息
        topic_title = topic_info.get('title', '未知话题')
        topic_summary = topic_info.get('summary', '')
        messages = topic_info.get('messages', [])
        
        # 2. 生成笔记内容
        if use_llm:
            content = self._generate_with_llm(topic_info, messages, qa_pairs)
        else:
            content = self._generate_template(topic_info, messages, qa_pairs)
        
        # 3. 生成文件名
        date = datetime.now().strftime('%Y-%m-%d')
        safe_title = re.sub(r'[^\w\s-]', '', topic_title).strip()[:30]
        safe_title = safe_title.replace(' ', '_')
        filename = f"{date}_{safe_title}.md"
        filepath = Path(self.output_dir) / filename
        
        # 4. 保存到文件
        filepath.write_text(content, encoding='utf-8')
        
        # 5. 构建结果
        result = {
            'topic_id': topic_info.get('topic_id', 0),
            'title': topic_title,
            'filename': filename,
            'filepath': str(filepath),
            'content': content,
            'importance_score': importance_score.get('importance_score', 0),
            'generated_at': datetime.now().isoformat(),
            'word_count': len(content)
        }
        
        return result
    
    def _generate_with_llm(self, topic_info: Dict, messages: List[Dict],
                           qa_pairs: List[Dict] = None) -> str:
        """
        使用LLM生成笔记内容
        
        Args:
            topic_info: 话题信息
            messages: 消息列表
            qa_pairs: QA对列表
            
        Returns:
            Markdown格式的笔记内容
        """
        llm = self._get_llm_client()
        
        try:
            prompt = self._build_llm_prompt(topic_info, messages, qa_pairs)
            content = llm.chat([
                {"role": "system", "content": "你是技术文档整理专家，只能基于给定聊天内容输出结构化笔记。"},
                {"role": "user", "content": prompt}
            ], temperature=0.4)
            return content
        except Exception as e:
            print(f"    ⚠️ LLM生成失败: {e}，使用模板")
            return self._generate_template(topic_info, messages, qa_pairs)
    
    def _generate_template(self, topic_info: Dict, messages: List[Dict],
                           qa_pairs: List[Dict] = None) -> str:
        """
        使用模板生成笔记内容
        
        Args:
            topic_info: 话题信息
            messages: 消息列表
            qa_pairs: QA对列表
            
        Returns:
            Markdown格式的笔记内容
        """
        # 提取信息
        title = topic_info.get('title', '未知话题')
        summary = topic_info.get('summary', '')
        date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # 提取关键讨论点
        key_points = self._extract_key_points(messages)
        
        # 提取资源
        resources = self._extract_resources(messages)
        
        # 提取关键词
        keywords = self._extract_keywords(messages)
        
        # 提取QA内容
        qa_content = self._format_qa_pairs(qa_pairs)
        
        # 构建Markdown
        md_content = f"""# {title}

> 生成时间: {date} | 重要度: {topic_info.get('importance_score', 'N/A')}分

## 核心问题/分享

{summary if summary else '本话题主要讨论了' + title}

---

## 关键讨论点

{chr(10).join([f"- {point}" for point in key_points[:8]])}

---

## 推荐资源

{chr(10).join([f"- {resource}" for resource in resources])}

---

## 问答精选

{qa_content if qa_content else '本话题暂无明显的问答对。'}

---

## 相关关键词

{', '.join(keywords[:10])}

---

## 原始消息统计

- 消息数量: {len(messages)}
- 参与人数: {topic_info.get('participant_count', 'N/A')}
- 时间范围: {topic_info.get('start_time', 'N/A')} - {topic_info.get('end_time', 'N/A')}

---

*由聊天记录分析系统自动生成*
"""
        return md_content

    def _build_llm_prompt(self, topic_info: Dict, messages: List[Dict],
                          qa_pairs: List[Dict] = None) -> str:
        """
        构建LLM笔记生成提示词
        """
        core_messages = []
        for msg in messages[:60]:
            content = msg.get('content', '')
            if not content:
                continue
            time_str = msg.get('time', '')[:8]
            sender = msg.get('sender_name', 'Unknown')
            core_messages.append(f"- [{time_str}] {sender}: {content[:200]}")

        qa_text = ""
        if qa_pairs:
            qa_lines = []
            for i, pair in enumerate(qa_pairs[:5], 1):
                question = pair.get('question_content', '')
                answer = pair.get('answer_content', '')
                if not question:
                    continue
                qa_lines.append(f"{i}. Q: {question[:120]}")
                if answer:
                    qa_lines.append(f"   A: {answer[:200]}")
            qa_text = "\n".join(qa_lines)

        prompt = f"""
请基于以下群聊内容生成结构化Markdown笔记，不要编造信息。

话题标题：{topic_info.get('title', '未知话题')}
话题摘要：{topic_info.get('summary', '')}

核心消息片段：
{chr(10).join(core_messages)}

问答摘要（如有）：
{qa_text if qa_text else "无"}

输出要求：
1. 使用Markdown标题与小节
2. 必须包含：核心问题/分享、关键讨论点、推荐资源、问答精选、相关关键词
3. 关键讨论点3-8条，资源仅包含消息中出现的链接或媒体
4. 相关关键词5-10个，使用逗号分隔
5. 保持简洁，避免重复表达
"""
        if self.config.get('debug', {}).get('print_prompts', False):
            print("\n[Prompt][note_generator][note]\n" + prompt.strip() + "\n")
        self._append_log("Prompt.note_generator.note", prompt.strip())
        return prompt
    
    def _extract_key_points(self, messages: List[Dict]) -> List[str]:
        """
        提取关键讨论点
        
        Args:
            messages: 消息列表
            
        Returns:
            讨论点列表
        """
        points = []
        
        for msg in messages:
            content = msg.get('content', '')
            
            # 跳过太短的消息
            if len(content) < 10:
                continue
            
            # 跳过纯链接
            if content.startswith('http') and len(content) < 100:
                continue
            
            # 添加有意义的消息作为讨论点
            points.append(content[:200] + '...' if len(content) > 200 else content)
        
        # 只返回前8个
        return points[:8]
    
    def _extract_resources(self, messages: List[Dict]) -> List[str]:
        """
        提取资源链接
        
        Args:
            messages: 消息列表
            
        Returns:
            资源列表
        """
        resources = []
        
        for msg in messages:
            content = msg.get('content', '')
            media_info = msg.get('media_info', {})
            
            # 提取链接
            if 'http' in content.lower():
                urls = re.findall(r'https?://[^\s<>"]+', content)
                for url in urls[:2]:  # 每条消息最多2个
                    if url not in resources:
                        resources.append(url)
            
            # 提取媒体信息
            if media_info:
                title = media_info.get('title', '')
                url = media_info.get('url', '')
                if title and url:
                    resources.append(f"[{title}]({url})")
                elif url:
                    resources.append(url)
        
        return resources[:5]
    
    def _extract_keywords(self, messages: List[Dict]) -> List[str]:
        """
        提取关键词
        
        Args:
            messages: 消息列表
            
        Returns:
            关键词列表
        """
        keywords = set()
        
        # 常见技术关键词
        tech_terms = [
            'AI', 'LLM', 'GPT', 'Claude', 'Python', 'JavaScript',
            '编程', '代码', '算法', '框架', '数据库', 'API',
            'Docker', 'Git', 'Linux', '机器学习', '深度学习',
            'Prompt', '提示词', '微调', '训练', '推理'
        ]
        
        for msg in messages:
            content = msg.get('content', '')
            
            for term in tech_terms:
                if term in content:
                    keywords.add(term)
        
        return list(keywords)
    
    def _format_qa_pairs(self, qa_pairs: List[Dict]) -> str:
        """
        格式化QA对
        
        Args:
            qa_pairs: QA对列表
            
        Returns:
            格式化后的QA内容
        """
        if not qa_pairs:
            return ''
        
        formatted = []
        
        for i, pair in enumerate(qa_pairs[:3], 1):  # 最多3个QA
            question = pair.get('question_content', '')
            answers = pair.get('answer_contents', [])
            
            qa_text = f"**Q{i}:** {question}\n\n"
            qa_text += '**A:**\n'
            for j, answer in enumerate(answers[:2], 1):  # 每个问题最多2个回答
                qa_text += f"{j}. {answer[:150]}"
                if len(answer) > 150:
                    qa_text += '...'
                qa_text += '\n'
            
            formatted.append(qa_text)
        
        return '\n'.join(formatted)
    
    def save_to_database(self, result: Dict, db_path: str = None):
        """
        保存到SQLite数据库
        
        Args:
            result: 生成结果
            db_path: 数据库路径
        """
        if db_path is None:
            db_path = self.config.get('database', {}).get('path', './data/chat_analysis.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                title TEXT,
                filename TEXT,
                filepath TEXT,
                content TEXT,
                importance_score REAL,
                word_count INTEGER,
                generated_at TEXT
            )
        ''')
        
        # 插入数据
        cursor.execute('''
            INSERT INTO notes (
                topic_id, title, filename, filepath, content,
                importance_score, word_count, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('topic_id', 0),
            result.get('title', ''),
            result.get('filename', ''),
            result.get('filepath', ''),
            result.get('content', ''),
            result.get('importance_score', 0),
            result.get('word_count', 0),
            result.get('generated_at', '')
        ))
        
        conn.commit()
        conn.close()
    
    def batch_generate(self, topics: List[Dict], scores: List[Dict],
                       qa_pairs_list: List[List[Dict]] = None,
                       use_llm: bool = True) -> List[Dict]:
        """
        批量生成笔记
        
        Args:
            topics: 话题列表
            scores: 评分列表
            qa_pairs_list: QA对列表
            use_llm: 是否使用LLM
            
        Returns:
            生成结果列表
        """
        results = []
        
        for i, topic in enumerate(topics):
            print(f"  生成话题 {i+1}/{len(topics)} 的笔记...")
            
            score = scores[i] if i < len(scores) else {}
            
            # 只为通过阈值的话题生成笔记
            if score.get('pass_threshold', False):
                qa_pairs = qa_pairs_list[i] if qa_pairs_list and i < len(qa_pairs_list) else None
                
                result = self.generate(topic, score, qa_pairs, use_llm)
                results.append(result)
                
                print(f"    ✅ 已生成: {result['filename']}")
            else:
                print(f"    ⏭️  跳过: {topic.get('title', '未知话题')} (重要度 {score.get('importance_score', 0)} < {self.threshold})")
        
        return results
    
    def get_notes_statistics(self, db_path: str = None) -> Dict:
        """
        获取笔记统计信息
        
        Args:
            db_path: 数据库路径
            
        Returns:
            统计信息
        """
        if db_path is None:
            db_path = self.config.get('database', {}).get('path', './data/chat_analysis.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询数量
        cursor.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]
        
        # 查询平均分
        cursor.execute("SELECT AVG(importance_score) FROM notes")
        avg_score = cursor.fetchone()[0] or 0
        
        # 查询词数统计
        cursor.execute("SELECT AVG(word_count) FROM notes")
        avg_words = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_notes': count,
            'average_score': round(avg_score, 2),
            'average_words': round(avg_words, 0),
            'pass_threshold': self.threshold
        }


# ==================== 便捷函数 ====================

def generate_notes(topics: List[Dict], scores: List[Dict],
                   config_path: str = "config.yaml",
                   use_llm: bool = True) -> List[Dict]:
    """
    便捷的批量笔记生成函数
    
    Args:
        topics: 话题列表
        scores: 评分列表
        config_path: 配置文件路径
        use_llm: 是否使用LLM
        
    Returns:
        生成结果列表
    """
    generator = NoteGenerator(config_path)
    return generator.batch_generate(topics, scores, use_llm=use_llm)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试笔记生成器...")
    
    # 创建测试数据
    test_topic = {
        'topic_id': 1,
        'title': 'AI编程助手使用技巧',
        'summary': '讨论Claude、Cursor等AI编程工具的使用技巧',
        'message_count': 15,
        'participant_count': 5,
        'messages': [
            {'content': '大家好，请问怎么使用Claude Code？', 'message_type': 0},
            {'content': 'Claude Code是一个AI编程助手，可以帮助编写代码', 'message_type': 0},
            {'content': '能举个例子吗？', 'message_type': 0},
            {'content': 'https://claude.com', 'message_type': 99}
        ]
    }
    
    test_score = {
        'importance_score': 7.5,
        'pass_threshold': True
    }
    
    # 初始化生成器
    generator = NoteGenerator()
    print("✅ 笔记生成器初始化成功")
    
    # 生成笔记
    result = generator.generate(test_topic, test_score, use_llm=False)
    print(f"✅ 笔记生成成功: {result['filename']}")
    print(f"   路径: {result['filepath']}")
    print(f"   字数: {result['word_count']}")
