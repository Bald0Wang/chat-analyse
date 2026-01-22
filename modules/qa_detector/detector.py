"""
QA问答对识别模块 - 检测器（增强版）

主要功能：
1. 使用LLM意图识别检测问题和回答（主要方案）
2. 关键字仅作为快速过滤备选

改进：
- 以LLM理解语义为核心
- 关键字仅用于预处理筛选
- 问答对由语义分析决定

作者: AI Assistant
创建时间: 2026-01-13
"""

import yaml
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import sys

# 导入LLM客户端
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.llm_client import get_llm_client


class QADetector:
    """QA问答对检测器（增强版）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化QA检测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.qa_config = self.config.get('qa_detector', {})
        
        # 获取配置
        self.answer_window = self.qa_config.get('answer_window', 15)
        self.batch_size = self.qa_config.get('batch_size', 10)
        
        # LLM客户端（延迟初始化）
        self.llm = None
        
        # 关键字备选（仅用于快速过滤）
        self.question_keywords = self.qa_config.get('question_keywords', [
            '?', '怎么', '如何', '求助', '报错', '求教', 
            '请问', '为什么', '什么意思', '怎么做'
        ])
    
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
    
    def detect(self, messages: List[Dict], use_llm: bool = True) -> List[Dict]:
        """
        检测QA问答对（主要方法）
        
        策略：
        1. 主要方案：使用LLM意图识别，理解语义
        2. 备选方案：使用关键字规则
        
        Args:
            messages: 消息列表
            use_llm: 是否使用LLM
            
        Returns:
            检测到的QA对列表
        """
        if not messages:
            return []
        
        if use_llm:
            # 主要方案：LLM意图识别
            return self._detect_with_llm(messages)
        else:
            # 备选方案：关键字规则
            return self._detect_with_rules(messages)

    def detect_with_intents(self, messages: List[Dict], intent_results: List[Dict]) -> List[Dict]:
        """
        使用已有意图结果进行QA检测（避免重复LLM调用）
        """
        if not messages or not intent_results:
            return []
        qa_pairs = self._extract_qa_from_intents(messages, intent_results)
        for qa_pair in qa_pairs:
            qa_pair = self._assess_qa_quality(qa_pair, messages)
        print(f"    ✅ 检测到 {len(qa_pairs)} 个QA对（复用意图结果）")
        return qa_pairs
    
    def _detect_with_llm(self, messages: List[Dict]) -> List[Dict]:
        """
        使用LLM意图识别检测QA对（主要方法）
        
        Args:
            messages: 消息列表
            
        Returns:
            QA对列表
        """
        print(f"  使用LLM意图识别进行QA检测...")
        
        llm = self._get_llm_client()
        
        # 1. 使用LLM分析所有消息的意图
        intent_results = self._analyze_intents_with_llm(llm, messages)
        
        if not intent_results:
            print("    ⚠️ 未获取到意图分析结果，使用备选方案")
            return self._detect_with_rules(messages)
        
        print(f"    ✅ 意图分析完成，检测到 {len(intent_results)} 条消息的意图")
        
        # 2. 根据意图结果提取QA对
        qa_pairs = self._extract_qa_from_intents(messages, intent_results)
        
        # 3. 对每个QA对进行质量评估
        for qa_pair in qa_pairs:
            qa_pair = self._assess_qa_quality(qa_pair, messages)
        
        print(f"    ✅ 检测到 {len(qa_pairs)} 个QA对")
        
        return qa_pairs

    def _analyze_intents_with_llm(self, llm, messages: List[Dict]) -> List[Dict]:
        """
        使用LLM按批次分析消息意图
        """
        results: List[Dict] = []
        batch_size = self.batch_size
        total_batches = (len(messages) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(messages))
            batch_messages = messages[start_idx:end_idx]
            prompt = self._build_intent_prompt(batch_messages, start_idx)
            try:
                batch_result = llm.extract_json(prompt)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    raise ValueError("LLM返回非列表结果")
            except Exception as e:
                print(f"    ⚠️ 意图批次解析失败: {e}，使用备选方案")
                return []

        ordered = []
        index_map = {}
        for item in results:
            try:
                idx = int(item.get("index", 0))
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(messages):
                index_map[idx] = item

        for i in range(1, len(messages) + 1):
            ordered.append(index_map.get(i, {
                "index": i,
                "intent_type": "other",
                "is_new_topic": False,
                "is_answer": False,
                "question_index": None,
                "topic_relevance": 5,
                "confidence": 0.2,
                "reason": "默认占位"
            }))

        return ordered

    def _build_intent_prompt(self, batch_messages: List[Dict], start_offset: int) -> str:
        """
        构建意图识别提示词
        """
        messages_text = []
        for i, msg in enumerate(batch_messages):
            time_str = msg.get('time', '')[:8]
            sender = msg.get('sender_name', 'Unknown')
            content = msg.get('content', '')[:160]
            msg_type = msg.get('message_type', 0)
            type_marker = ""
            if msg_type == 99:
                type_marker = "[分享链接]"
            elif msg_type == 5:
                type_marker = "[表情]"
            elif msg_type == 80:
                type_marker = "[系统]"
            messages_text.append(f"{i + start_offset + 1}. [{time_str}] {sender}: {type_marker}{content}")

        dialogue = "\n".join(messages_text)

        prompt = f"""
你是群聊意图识别专家，请逐条判断消息意图，禁止编造。

对话片段：
{dialogue}

输出要求：
1. intent_type 仅允许: question, answer, share, chat, system, other
2. 如果是回答，给出 question_index（从1开始）
3. is_new_topic 标记明显话题切换
4. confidence 0-1，越不确定越低
5. 每条消息必须输出一条记录

仅输出JSON数组（不要使用Markdown，不要附加说明）：
[
  {{
    "index": 消息序号,
    "intent_type": "question|answer|share|chat|system|other",
    "is_new_topic": true/false,
    "is_answer": true/false,
    "question_index": 数字或null,
    "topic_relevance": 0-10,
    "confidence": 0.0-1.0,
    "reason": "简短原因"
  }}
]
"""
        if self.config.get('debug', {}).get('print_prompts', False):
            print("\n[Prompt][qa_detector][intent]\n" + prompt.strip() + "\n")
        prompt = self._truncate_prompt(prompt)
        self._append_log("Prompt.qa_detector.intent", prompt.strip())
        return prompt

    def _truncate_prompt(self, prompt: str) -> str:
        """
        按最大上下文长度截断prompt（保留头尾）
        """
        max_chars = int(self.config.get('llm', {}).get('max_context_chars', 32000))
        if len(prompt) <= max_chars:
            return prompt
        head_len = min(2000, max_chars // 4)
        tail_len = max_chars - head_len - 40
        if tail_len <= 0:
            return prompt[-max_chars:]
        return prompt[:head_len] + "\n... [内容过长已截断] ...\n" + prompt[-tail_len:]
    
    def _extract_qa_from_intents(self, messages: List[Dict], 
                                  intent_results: List[Dict]) -> List[Dict]:
        """
        从意图分析结果中提取QA对
        
        Args:
            messages: 原始消息列表
            intent_results: 意图分析结果
            
        Returns:
            QA对列表
        """
        qa_pairs = []
        pending_questions = []  # 待回答的问题
        
        for i, intent in enumerate(intent_results):
            intent_type = intent.get('intent_type', 'other')
            confidence = intent.get('confidence', 0.5)
            
            msg = messages[i]
            
            # 如果是问题（意图识别为question）
            if intent_type == 'question':
                pending_questions.append({
                    'index': i,
                    'message': msg,
                    'confidence': confidence,
                    'intent': intent
                })
            
            # 如果是回答（意图识别为answer）
            elif intent_type == 'answer':
                # 尝试找到它回答的问题
                question_idx = self._find_answered_question(
                    messages, intent_results, i, pending_questions
                )
                
                if question_idx is not None:
                    # 构建QA对
                    qa_pair = {
                        'question_index': question_idx,
                        'answer_index': i,
                        'question': messages[question_idx],
                        'answer': msg,
                        'question_content': messages[question_idx].get('content', ''),
                        'answer_content': msg.get('content', ''),
                        'confidence': confidence,
                        'detection_method': 'llm_intent',
                        'reason': f"LLM识别: {intent.get('reason', '意图识别为回答')}"
                    }
                    
                    qa_pairs.append(qa_pair)
                    
                    # 从待回答列表中移除
                    pending_questions = [q for q in pending_questions if q['index'] != question_idx]
        
        # 3. 添加未回答的问题（可选）
        for q in pending_questions:
            if q['confidence'] > 0.7:  # 只添加高置信度的问题
                qa_pair = {
                    'question_index': q['index'],
                    'answer_index': None,
                    'question': q['message'],
                    'answer': None,
                    'question_content': q['message'].get('content', ''),
                    'answer_content': '',
                    'confidence': q['confidence'],
                    'detection_method': 'llm_intent',
                    'reason': f"问题未得到回答: {q['intent'].get('reason', '')}",
                    'is_unanswered': True
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _find_answered_question(self, messages: List[Dict], 
                                intent_results: List[Dict],
                                answer_idx: int,
                                pending_questions: List[Dict]) -> Optional[int]:
        """
        找到回答对应的问题
        
        Args:
            messages: 消息列表
            intent_results: 意图分析结果
            answer_idx: 回答消息的索引
            pending_questions: 待回答的问题列表
            
        Returns:
            问题索引，未找到返回None
        """
        answer_msg = messages[answer_idx]
        answer_sender = answer_msg.get('sender_name', '')
        answer_content = answer_msg.get('content', '').lower()
        
        # 如果LLM已经指明了回答的问题
        answer_intent = intent_results[answer_idx]
        if 'question_index' in answer_intent:
            q_raw = answer_intent.get('question_index')
            try:
                q_int = int(q_raw) if q_raw is not None else None
            except (TypeError, ValueError):
                q_int = None
            if q_int is not None:
                q_idx = q_int - 1  # 转为0基索引
            if 0 <= q_idx < len(messages):
                return q_idx
        
        # 否则，根据语义匹配
        # 1. 找到最近的、未回答的问题
        best_match = None
        best_match_idx = -1
        
        for q in pending_questions:
            q_idx = q['index']
            
            # 跳过同一人的自问自答
            if q['message'].get('sender_name', '') == answer_sender:
                continue
            
            # 2. 检查回答内容是否包含问题的关键词
            q_content = q['message'].get('content', '').lower()
            
            # 提取问题中的关键词（简化：取前5个词）
            q_words = q_content.split()[:5]
            
            # 计算匹配度
            matches = sum(1 for word in q_words if len(word) > 2 and word in answer_content)
            
            # 3. 选择匹配度最高的问题
            if matches > 0 and q_idx > best_match_idx:
                best_match = q
                best_match_idx = q_idx
        
        return best_match_idx if best_match_idx >= 0 else None
    
    def _assess_qa_quality(self, qa_pair: Dict, 
                          all_messages: List[Dict]) -> Dict:
        """
        评估QA对的质量
        
        Args:
            qa_pair: QA对
            all_messages: 所有消息
            
        Returns:
            评估后的QA对
        """
        question = qa_pair.get('question_content', '')
        answer = qa_pair.get('answer_content', '')
        
        # 检查是否有代码
        has_code = '```' in answer or any(marker in answer for marker in 
                        ['def ', 'class ', 'import ', 'from ', '    '])
        
        # 检查是否有链接
        has_link = 'http' in answer.lower() or 'www.' in answer.lower()
        
        # 检查回答长度
        answer_length = len(answer)
        
        # 基础质量分（0-10）
        quality_score = 5.0
        
        if has_code:
            quality_score += 2.0
        if has_link:
            quality_score += 1.5
        if answer_length > 100:
            quality_score += 1.0
        elif answer_length > 50:
            quality_score += 0.5
        elif answer_length < 10:
            quality_score -= 1.0
        
        quality_score = min(max(quality_score, 0), 10)
        
        # 更新QA对信息
        qa_pair['has_code'] = has_code
        qa_pair['has_link'] = has_link
        qa_pair['answer_length'] = answer_length
        qa_pair['quality_score'] = quality_score
        qa_pair['is_valid'] = quality_score >= 4.0
        qa_pair['solved'] = quality_score >= 6.0
        
        return qa_pair
    
    def _detect_with_rules(self, messages: List[Dict]) -> List[Dict]:
        """
        使用规则检测QA对（备选方案）
        仅当LLM不可用时使用
        
        Args:
            messages: 消息列表
            
        Returns:
            QA对列表
        """
        qa_pairs = []
        pending_questions = []
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            sender = msg.get('sender_name', '')
            
            # 跳过太短的消息
            if len(content) < 3:
                continue
            
            # 检查是否是问题（关键字匹配）
            is_question = self._is_question_by_keyword(content)
            
            if is_question:
                # 保存问题
                pending_questions.append({
                    'index': i,
                    'message': msg,
                    'content': content
                })
            
            elif pending_questions:
                # 检查是否是有效回答
                last_q = pending_questions[-1]
                
                # 跳过自己回答自己
                if last_q['message'].get('sender_name', '') == sender:
                    continue
                
                # 跳过表情和系统消息
                msg_type = msg.get('message_type', 0)
                if msg_type in [5, 80]:
                    continue
                
                # 检查是否有实际内容
                if self._has_meaningful_content(content):
                    qa_pair = {
                        'question_index': last_q['index'],
                        'answer_index': i,
                        'question': last_q['message'],
                        'answer': msg,
                        'question_content': last_q['content'],
                        'answer_content': content,
                        'confidence': 0.5,  # 规则匹配置信度较低
                        'detection_method': 'keywords',
                        'reason': '基于关键字规则检测'
                    }
                    
                    qa_pair = self._assess_qa_quality(qa_pair, messages)
                    qa_pairs.append(qa_pair)
                    
                    # 从待回答列表中移除
                    pending_questions.pop()
        
        # 添加未回答的问题
        for q in pending_questions:
            qa_pair = {
                'question_index': q['index'],
                'answer_index': None,
                'question': q['message'],
                'answer': None,
                'question_content': q['content'],
                'answer_content': '',
                'confidence': 0.4,
                'detection_method': 'keywords',
                'reason': '问题未得到回答',
                'is_unanswered': True
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _is_question_by_keyword(self, content: str) -> bool:
        """
        使用关键字判断是否为问题（备选方案）
        
        Args:
            content: 消息内容
            
        Returns:
            是否为问题
        """
        if not content:
            return False
        
        # 检查结尾问号
        if content.strip().endswith('?'):
            return True
        
        # 检查疑问词
        for keyword in self.question_keywords:
            if keyword in content:
                return True
        
        return False
    
    def _has_meaningful_content(self, content: str) -> bool:
        """
        检查是否有实际内容（备选方案）
        
        Args:
            content: 消息内容
            
        Returns:
            是否有实际内容
        """
        if not content:
            return False
        
        # 跳过表情包
        if content.startswith('[') and content.endswith(']'):
            return False
        
        # 跳过系统消息
        if any(keyword in content for keyword in ['撤回', '邀请', '加入', '移出']):
            return False
        
        # 检查长度
        if len(content) < 3:
            return False
        
        # 检查是否有有意义的内容
        meaningful_words = ['是', '可以', '应该', '需要', '用', '把', '到', 
                          'http', 'www', '代码', '方法', '步骤']
        has_meaningful = any(word in content for word in meaningful_words)
        
        return has_meaningful or len(content) > 10
    
    def quick_filter_questions(self, messages: List[Dict]) -> List[int]:
        """
        快速过滤可能的问题消息（关键字备选）
        
        仅作为LLM分析的预处理，快速筛选可能的问题
        不作为最终判断
        
        Args:
            messages: 消息列表
            
        Returns:
            可能的问题消息索引列表
        """
        question_indices = []
        
        for i, msg in enumerate(messages):
            if self._is_question_by_keyword(msg.get('content', '')):
                question_indices.append(i)
        
        return question_indices


    def get_qa_statistics(self, qa_pairs: List[Dict]) -> Dict[str, Any]:
        """
        获取QA对统计信息
        """
        if not qa_pairs:
            return {
                "total_qa_pairs": 0,
                "llm_detected": 0,
                "rule_detected": 0,
                "avg_quality_score": 0.0,
                "avg_answer_delay": 0.0
            }
        
        llm_count = sum(1 for qa in qa_pairs if qa.get("detection_method") == "llm_intent")
        rule_count = sum(1 for qa in qa_pairs if qa.get("detection_method") == "keywords")
        
        quality_scores = [qa.get("quality_score", 0) for qa in qa_pairs]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "total_qa_pairs": len(qa_pairs),
            "llm_detected": llm_count,
            "rule_detected": rule_count,
            "avg_quality_score": round(avg_quality, 2),
            "avg_answer_delay": 0.0  # 可以扩展为计算平均回答延迟
        }

# ==================== 便捷函数 ====================

def detect_qa_pairs(messages: List[Dict], config_path: str = "config.yaml",
                    use_llm: bool = True) -> List[Dict]:
    """
    便捷的QA检测函数
    
    Args:
        messages: 消息列表
        config_path: 配置文件路径
        use_llm: 是否使用LLM（主要方案）
        
    Returns:
        检测到的QA对列表
    """
    detector = QADetector(config_path)
    return detector.detect(messages, use_llm)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试QA检测器（增强版）...")
    
    # 创建测试数据
    test_messages = [
        {
            'sender_name': '用户A',
            'content': '大家好，请问怎么使用Python的装饰器？',
            'message_type': 0,
            'is_question': True
        },
        {
            'sender_name': '用户B',
            'content': '装饰器是一个函数，接受另一个函数作为参数',
            'message_type': 0,
            'is_question': False
        },
        {
            'sender_name': '用户C',
            'content': '能举个例子吗？',
            'message_type': 0,
            'is_question': True
        },
        {
            'sender_name': '用户B',
            'content': '''def my_decorator(func):
    def wrapper():
        print("执行前")
        func()
        print("执行后")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")''',
            'message_type': 0,
            'is_question': False
        }
    ]
    
    # 初始化检测器
    detector = QADetector()
    print("✅ QA检测器初始化成功")
    
    # 测试（使用规则作为备选）
    qa_pairs = detector.detect(test_messages, use_llm=False)
    print(f"✅ 检测到 {len(qa_pairs)} 个QA对")
    
    for i, pair in enumerate(qa_pairs):
        print(f"\nQA对 {i+1}:")
        print(f"  问题: {pair['question_content'][:50]}...")
        print(f"  回答: {pair['answer_content'][:50] if pair['answer_content'] else '无'}...")
        print(f"  方法: {pair.get('detection_method', 'N/A')}")
        print(f"  质量分: {pair.get('quality_score', 'N/A')}")
