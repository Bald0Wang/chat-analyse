"""
话题分割模块 - 分割器（增强版）

主要功能：
1. 使用LLM意图识别检测话题边界（主要方案）
2. 时间间隔规则作为备选

改进：
- 以LLM理解对话上下文为核心
- 关键字仅作为快速过滤
- 话题边界由语义分析决定

作者: AI Assistant
创建时间: 2026-01-13
"""

import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

# 导入LLM客户端
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.llm_client import get_llm_client


class TopicSegmenter:
    """话题分割器（增强版）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化话题分割器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.segment_config = self.config.get('topic_segment', {})
        
        # 获取配置参数
        self.time_gap_threshold = self.segment_config.get('time_gap_threshold', 1800)
        self.batch_size = self.segment_config.get('batch_size', 20)
        self.min_messages = self.segment_config.get('min_messages', 3)
        
        # LLM客户端（延迟初始化）
        self.llm = None
        
        # 关键字备选（仅用于快速过滤，不作为主要判断）
        self.keywords = {
            'new_topic': ['大家好', '请问', '请教', '分享', '突然想到', '插个话题'],
            'question': ['?', '怎么', '如何', '求助', '报错', '求教']
        }
    
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
    
    def segment(self, messages: List[Dict], use_llm: bool = True) -> List[Dict]:
        """
        分割话题（主要方法）
        
        策略：
        1. 主要方案：使用LLM理解对话上下文，识别话题边界
        2. 备选方案：使用时间间隔规则
        
        Args:
            messages: 清洗后的消息列表
            use_llm: 是否使用LLM
            
        Returns:
            分割后的话题列表
        """
        if not messages:
            return []
        
        if use_llm:
            # 主要方案：LLM意图识别
            return self._segment_with_llm(messages)
        else:
            # 备选方案：规则分割
            return self._segment_with_rules(messages)
    
    def _segment_with_llm(self, messages: List[Dict]) -> List[Dict]:
        """
        使用LLM意图识别进行话题分割（主要方法）
        
        Args:
            messages: 消息列表
            
        Returns:
            分割后的话题列表
        """
        print(f"  使用LLM意图识别进行话题分割...")
        
        llm = self._get_llm_client()
        
        # 1. 使用LLM检测话题边界
        topic_boundaries = llm.detect_topic_boundaries_llm(messages, window_size=self.batch_size)
        
        if not topic_boundaries:
            print("    ⚠️ 未检测到话题边界，使用备选方案")
            return self._segment_with_rules(messages)
        
        print(f"    ✅ LLM检测到 {len(topic_boundaries)} 个话题边界")
        
        # 2. 根据边界提取话题
        topics = []
        for i, boundary in enumerate(topic_boundaries):
            start_idx = boundary['start_index']
            end_idx = boundary['end_index']
            
            # 提取话题消息
            topic_messages = messages[start_idx:end_idx + 1]
            
            if len(topic_messages) < self.min_messages:
                continue
            
            # 3. 为每个话题生成标题（使用LLM）
            topic_info = self._generate_topic_info(topic_messages, i + 1)
            
            topic = {
                'topic_id': i + 1,
                'messages': topic_messages,
                'start_index': start_idx,
                'end_index': end_idx,
                'title': topic_info.get('title', f'话题{i+1}'),
                'summary': topic_info.get('summary', ''),
                'transition_reason': boundary.get('transition_reason', ''),
                'segmentation_method': 'llm_intent',
                'message_count': len(topic_messages)
            }
            
            # 计算参与人数
            participants = set()
            for msg in topic_messages:
                participants.add(msg.get('sender_name', ''))
            topic['participant_count'] = len(participants)
            
            # 计算时间范围
            if topic_messages:
                times = [msg.get('timestamp', 0) for msg in topic_messages if msg.get('timestamp')]
                if times:
                    topic['start_time'] = min(times)
                    topic['end_time'] = max(times)
            
            topics.append(topic)
        
        return topics

    def segment_with_intents(self, messages: List[Dict], intent_results: List[Dict]) -> List[Dict]:
        """
        使用已有意图结果进行话题分割（避免重复LLM调用）
        """
        if not messages:
            return []

        topic_boundaries = self._boundaries_from_intents(intent_results, len(messages))
        if not topic_boundaries:
            print("    ⚠️ 未检测到话题边界，使用备选方案")
            return self._segment_with_rules(messages)

        print(f"    ✅ 意图结果生成 {len(topic_boundaries)} 个话题边界")

        topics = []
        for i, boundary in enumerate(topic_boundaries):
            start_idx = boundary['start_index']
            end_idx = boundary['end_index']
            topic_messages = messages[start_idx:end_idx + 1]
            if len(topic_messages) < self.min_messages:
                continue
            topic_info = self._generate_topic_info(topic_messages, i + 1)
            topic = {
                'topic_id': i + 1,
                'messages': topic_messages,
                'start_index': start_idx,
                'end_index': end_idx,
                'title': topic_info.get('title', f'话题{i+1}'),
                'summary': topic_info.get('summary', ''),
                'transition_reason': boundary.get('transition_reason', ''),
                'segmentation_method': 'llm_intent',
                'message_count': len(topic_messages)
            }
            participants = set()
            for msg in topic_messages:
                participants.add(msg.get('sender_name', ''))
            topic['participant_count'] = len(participants)
            topics.append(topic)
        return topics

    def _boundaries_from_intents(self, intent_results: List[Dict], total_messages: int) -> List[Dict]:
        """
        根据意图结果生成话题边界
        """
        if not intent_results:
            return []
        topic_boundaries = []
        current_topic_start = 0
        for i, intent in enumerate(intent_results):
            is_new_topic = bool(intent.get('is_new_topic', False))
            relevance_raw = intent.get('topic_relevance', 10)
            try:
                relevance = float(relevance_raw) if relevance_raw is not None else 10.0
            except (TypeError, ValueError):
                relevance = 10.0
            if is_new_topic or relevance < 4:
                if i > current_topic_start:
                    topic_boundaries.append({
                        'start_index': current_topic_start,
                        'end_index': i - 1,
                        'transition_reason': intent.get('reason', '话题切换')
                    })
                current_topic_start = i
        if current_topic_start < total_messages:
            topic_boundaries.append({
                'start_index': current_topic_start,
                'end_index': total_messages - 1,
                'transition_reason': '对话结束'
            })
        return topic_boundaries
    
    def _generate_topic_info(self, topic_messages: List[Dict], topic_num: int) -> Dict:
        """
        使用LLM为话题生成标题和摘要
        
        Args:
            topic_messages: 话题消息列表
            topic_num: 话题编号
            
        Returns:
            话题信息字典
        """
        llm = self._get_llm_client()
        
        # 构建消息文本
        messages_text = []
        for i, msg in enumerate(topic_messages[:50]):  # 最多20条
            time_str = msg.get('time', '')[:8]
            sender = msg.get('sender_name', 'Unknown')
            content = msg.get('content', '')[:100]
            messages_text.append(f"[{time_str}] {sender}: {content}")
        
        dialogue = "\n".join(messages_text)
        
        prompt = f"""
你是群聊话题分析专家。请仅根据给定对话内容生成话题标题和摘要，不要臆测或补充未出现的信息。

对话片段：
{dialogue}

输出要求：
1. title: 10字以内，突出核心话题，避免使用笼统词
2. summary: 50字以内，概括主要讨论内容
3. key_points: 2-5条要点，短语化
4. 如信息不足，允许返回空摘要或空要点

输出JSON格式（不要使用Markdown代码块，不要包含额外文字）：
{{
  "title": "标题",
  "summary": "摘要",
  "key_points": ["要点1", "要点2"]
}}
"""
        if self.config.get('debug', {}).get('print_prompts', False):
            print("\n[Prompt][topic_segment][title_summary]\n" + prompt.strip() + "\n")
        self._append_log("Prompt.topic_segment.title_summary", prompt.strip())
        
        try:
            result = llm.extract_json(prompt)
            return {
                'title': result.get('title', f'话题{topic_num}'),
                'summary': result.get('summary', ''),
                'key_points': result.get('key_points', [])
            }
        except Exception as e:
            print(f"      ⚠️ 生成标题失败: {e}")
            return {
                'title': f'话题{topic_num}',
                'summary': '',
                'key_points': []
            }
    
    def _segment_with_rules(self, messages: List[Dict]) -> List[Dict]:
        """
        使用规则进行话题分割（备选方案）
        仅当LLM不可用时使用
        
        Args:
            messages: 消息列表
            
        Returns:
            分割后的话题列表
        """
        segments = []
        current_segment = {
            'start_index': 0,
            'messages': [messages[0]],
            'last_sender': messages[0].get('sender_name', '')
        }
        
        for i in range(1, len(messages)):
            msg = messages[i]
            current_time = msg.get('timestamp', 0)
            prev_time = messages[i-1].get('timestamp', 0)
            sender = msg.get('sender_name', '')
            
            # 规则1: 时间间隔 > 30分钟
            time_gap = current_time - prev_time
            if time_gap > self.time_gap_threshold:
                segments.append(current_segment)
                current_segment = {
                    'start_index': i,
                    'messages': [msg],
                    'last_sender': sender
                }
                continue
            
            current_segment['messages'].append(msg)
            current_segment['last_sender'] = sender
        
        # 添加最后一个片段
        if current_segment['messages']:
            segments.append(current_segment)
        
        # 构建话题信息
        topics = []
        for i, segment in enumerate(segments):
            if len(segment['messages']) < self.min_messages:
                continue
            
            topic = {
                'topic_id': i + 1,
                'messages': segment['messages'],
                'start_index': segment['start_index'],
                'title': f'话题{i+1}',
                'summary': '',
                'segmentation_method': 'rules',
                'message_count': len(segment['messages']),
                'transition_reason': '时间间隔分割'
            }
            
            # 计算参与人数
            participants = set()
            for msg in segment['messages']:
                participants.add(msg.get('sender_name', ''))
            topic['participant_count'] = len(participants)
            
            topics.append(topic)
        
        return topics
    
    def quick_filter_new_topic(self, messages: List[Dict], window_size: int = 10) -> List[int]:
        """
        快速过滤可能的新话题起始位置（关键字备选）
        
        仅作为LLM分析的预处理，快速筛选可能的话题边界
        不作为最终判断
        
        Args:
            messages: 消息列表
            window_size: 窗口大小
            
        Returns:
            可能的新话题起始索引列表
        """
        candidates = []
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            
            # 检查是否包含新话题关键词
            is_new_topic_candidate = False
            
            for keyword in self.keywords['new_topic']:
                if keyword in content:
                    is_new_topic_candidate = True
                    break
            
            # 检查是否是问题
            if not is_new_topic_candidate:
                for keyword in self.keywords['question']:
                    if keyword in content:
                        is_new_topic_candidate = True
                        break
            
            if is_new_topic_candidate:
                candidates.append(i)
        
        return candidates


# ==================== 便捷函数 ====================

def segment_topics(messages: List[Dict], config_path: str = "config.yaml", 
                   use_llm: bool = True) -> List[Dict]:
    """
    便捷的话题分割函数
    
    Args:
        messages: 消息列表
        config_path: 配置文件路径
        use_llm: 是否使用LLM（主要方案）
        
    Returns:
        分割后的话题列表
    """
    segmenter = TopicSegmenter(config_path)
    return segmenter.segment(messages, use_llm)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试话题分割器（增强版）...")
    
    # 创建测试数据
    test_messages = [
        {
            'timestamp': 1736706000,
            'sender_name': '用户A',
            'content': '大家好',
            'message_type': 0,
            'time': '10:00:00'
        },
        {
            'timestamp': 1736706100,
            'sender_name': '用户A',
            'content': '年假多长时间呀',
            'message_type': 0,
            'time': '10:01:00'
        },
        {
            'timestamp': 1736706200,
            'sender_name': '用户B',
            'content': '一般是5天',
            'message_type': 0,
            'time': '10:02:00'
        },
        {
            'timestamp': 1738706400,  # 6小时后
            'sender_name': '用户C',
            'content': '大家好，分享一个AI工具',
            'message_type': 99,
            'time': '16:00:00'
        }
    ]
    
    # 初始化分割器
    segmenter = TopicSegmenter()
    print("✅ 话题分割器初始化成功")
    
    # 测试（使用规则作为备选）
    topics = segmenter.segment(test_messages, use_llm=False)
    print(f"✅ 分割完成: {len(topics)} 个话题")
    
    for i, topic in enumerate(topics):
        print(f"\n话题 {i+1}:")
        print(f"  标题: {topic.get('title', 'N/A')}")
        print(f"  消息数: {topic.get('message_count', 0)}")
        print(f"  方法: {topic.get('segmentation_method', 'N/A')}")
