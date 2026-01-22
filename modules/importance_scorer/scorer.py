"""
重要度评估模块 - 评分器（增强版）

主要功能：
1. 使用LLM理解语义进行综合评分（主要方案）
2. 规则计算作为辅助参考

改进：
- 以LLM语义理解为核心
- 规则分数作为参考输入
- 更准确的评分判断

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


class ImportanceScorer:
    """重要度评分器（增强版）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化评分器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.scorer_config = self.config.get('importance_scorer', {})
        
        # 获取配置
        self.threshold = self.scorer_config.get('threshold', 6.0)
        self.weights = self.scorer_config.get('weights', {
            'engagement': 0.30,
            'depth': 0.25,
            'qa_quality': 0.20,
            'share_value': 0.15,
            'timeliness': 0.10
        })
        
        # LLM客户端（延迟初始化）
        self.llm = None
        
        # 技术关键词（仅作为参考，不作为主要判断）
        self.tech_keywords = [
            'AI', 'LLM', 'GPT', 'Claude', 'Python', 'JavaScript',
            '代码', '编程', '算法', '框架', '库', 'API',
            '数据库', 'Docker', 'Kubernetes', 'Git', 'Linux',
            '机器学习', '深度学习', '神经网络', 'Transformer',
            'Prompt', '提示词', '微调', '训练', '推理'
        ]
    
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
    
    def score(self, topic_info: Dict, qa_pairs: List[Dict] = None, 
              use_llm: bool = True) -> Dict:
        """
        评估话题重要度（主要方法）
        
        策略：
        1. 主要方案：使用LLM理解话题内容，综合评估
        2. 备选方案：使用规则计算各维度分数
        
        Args:
            topic_info: 话题信息
            qa_pairs: QA对列表
            use_llm: 是否使用LLM
            
        Returns:
            评分结果
        """
        # 1. 使用规则计算各维度分数（作为参考）
        rule_scores = self._calc_rule_scores(topic_info, qa_pairs)
        
        if use_llm:
            # 主要方案：LLM综合评估
            return self._score_with_llm(topic_info, qa_pairs, rule_scores)
        else:
            # 备选方案：仅使用规则分数
            return self._score_with_rules(topic_info, rule_scores)
    
    def _score_with_llm(self, topic_info: Dict, qa_pairs: List[Dict],
                        rule_scores: Dict) -> Dict:
        """
        使用LLM进行综合评分（主要方法）
        
        Args:
            topic_info: 话题信息
            qa_pairs: QA对列表
            rule_scores: 规则计算的分数字典
            
        Returns:
            评分结果
        """
        llm = self._get_llm_client()
        
        # 构建话题信息摘要
        topic_title = topic_info.get('title', '未知话题')
        topic_summary = topic_info.get('summary', '')
        message_count = topic_info.get('message_count', 0)
        participant_count = topic_info.get('participant_count', 0)
        
        # 统计QA信息
        qa_count = len(qa_pairs) if qa_pairs else 0
        answered_count = sum(1 for q in qa_pairs if q.get('answer') and not q.get('is_unanswered', False))
        solved_count = sum(1 for q in qa_pairs if q.get('solved', False))
        avg_quality = sum(q.get('quality_score', 5) for q in qa_pairs) / qa_count if qa_count > 0 else 0
        
        # 统计消息中的技术关键词
        messages = topic_info.get('messages', [])
        keyword_count = 0
        has_code = False
        has_link = False
        
        for msg in messages:
            content = msg.get('content', '')
            
            for keyword in self.tech_keywords:
                if keyword in content:
                    keyword_count += 1
            
            if '```' in content or any(marker in content for marker in ['def ', 'class ']):
                has_code = True
            
            if 'http' in content.lower():
                has_link = True
        
        # 构建提示词
        prompt = f"""
你是技术群话题重要度评估专家。请基于以下信息给出0-10分的评分，并保持输出为严格JSON。

话题信息：
- 标题: {topic_title}
- 摘要: {topic_summary}
- 消息数量: {message_count}
- 参与人数: {participant_count}

规则参考分数（仅作参考，最终评分由你决定）：
- engagement_score: {rule_scores.get('engagement_score', 0)}/10
- depth_score: {rule_scores.get('depth_score', 0)}/10
- qa_score: {rule_scores.get('qa_score', 0)}/10
- share_score: {rule_scores.get('share_score', 0)}/10
- timeliness_score: {rule_scores.get('timeliness_score', 0)}/10

补充信息：
- QA对数量: {qa_count}
- 已回答问题: {answered_count}
- 已解决问题: {solved_count}
- 平均QA质量: {avg_quality:.1f}/10
- 技术关键词出现次数: {keyword_count}
- 包含代码示例: {"是" if has_code else "否"}
- 包含链接资源: {"是" if has_link else "否"}

评分规则：
1. 评分基于内容价值、实用性、可复用性
2. 高质量问答或代码示例可加分
3. 纯闲聊或重复信息需降分
4. 各维度分数与最终score保持一致性

仅输出JSON（不要使用Markdown，不要附加说明）：
{{
  "score": X.X,
  "engagement_score": X.X,
  "depth_score": X.X,
  "qa_score": X.X,
  "share_score": X.X,
  "timeliness_score": X.X,
  "reason": "简洁的评分理由",
  "highlights": ["亮点1", "亮点2"],
  "improvements": ["改进建议1"]
}}
"""
        if self.config.get('debug', {}).get('print_prompts', False):
            print("\n[Prompt][importance_scorer][score]\n" + prompt.strip() + "\n")
        self._append_log("Prompt.importance_scorer.score", prompt.strip())
        
        try:
            result = llm.extract_json(prompt)
            
            final_score = result.get('score', rule_scores.get('weighted_score', 5.0))
            
            return {
                'importance_score': round(final_score, 2),
                'engagement_score': result.get('engagement_score', rule_scores.get('engagement_score', 0)),
                'depth_score': result.get('depth_score', rule_scores.get('depth_score', 0)),
                'qa_score': result.get('qa_score', rule_scores.get('qa_score', 0)),
                'share_score': result.get('share_score', rule_scores.get('share_score', 0)),
                'timeliness_score': result.get('timeliness_score', rule_scores.get('timeliness_score', 0)),
                'weighted_score': round(rule_scores.get('weighted_score', 0), 2),
                'threshold': self.threshold,
                'pass_threshold': final_score >= self.threshold,
                'reasons': [result.get('reason', '')],
                'highlights': result.get('highlights', []),
                'improvements': result.get('improvements', []),
                'scored_by': 'llm',
                'scored_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ⚠️ LLM评分失败: {e}，使用规则分数")
            return self._score_with_rules(topic_info, rule_scores)
    
    def _score_with_rules(self, topic_info: Dict, rule_scores: Dict) -> Dict:
        """
        仅使用规则评分（备选方案）
        
        Args:
            topic_info: 话题信息
            rule_scores: 规则计算的分数字典
            
        Returns:
            评分结果
        """
        weighted_score = rule_scores.get('weighted_score', 0)
        
        return {
            'importance_score': round(weighted_score, 2),
            'engagement_score': rule_scores.get('engagement_score', 0),
            'depth_score': rule_scores.get('depth_score', 0),
            'qa_score': rule_scores.get('qa_score', 0),
            'share_score': rule_scores.get('share_score', 0),
            'timeliness_score': rule_scores.get('timeliness_score', 0),
            'weighted_score': round(weighted_score, 2),
            'threshold': self.threshold,
            'pass_threshold': weighted_score >= self.threshold,
            'reasons': ['基于规则计算'],
            'highlights': [],
            'improvements': [],
            'scored_by': 'rules',
            'scored_at': datetime.now().isoformat()
        }
    
    def _calc_rule_scores(self, topic_info: Dict, qa_pairs: List[Dict]) -> Dict:
        """
        使用规则计算各维度分数（作为LLM的参考）
        
        Args:
            topic_info: 话题信息
            qa_pairs: QA对列表
            
        Returns:
            各维度分数字典
        """
        messages = topic_info.get('messages', [])
        message_count = topic_info.get('message_count', 0)
        participant_count = topic_info.get('participant_count', 0)
        
        # 参与度分数
        msg_score = min(message_count / 10 * 5, 5)
        part_score = min(participant_count / 5 * 5, 5)
        engagement_score = min(msg_score + part_score, 10)
        
        # 内容深度分数
        keyword_count = 0
        has_code = False
        has_link = False
        
        for msg in messages:
            content = msg.get('content', '')
            
            for keyword in self.tech_keywords:
                if keyword in content:
                    keyword_count += 1
            
            if '```' in content or any(marker in content for marker in ['def ', 'class ']):
                has_code = True
            
            if 'http' in content.lower():
                has_link = True
        
        keyword_score = min(keyword_count / 5 * 5, 5)
        bonus = 0
        if has_code:
            bonus += 2.5
        if has_link:
            bonus += 1.5
        depth_score = min(keyword_score + bonus, 10)
        
        # QA质量分数
        if qa_pairs:
            valid_pairs = [q for q in qa_pairs if q.get('is_valid', True)]
            
            qa_count_score = min(len(valid_pairs) / 3 * 4, 4)
            
            solved_pairs = [q for q in valid_pairs if q.get('solved', False)]
            solve_rate = len(solved_pairs) / len(valid_pairs) if valid_pairs else 0
            solve_score = solve_rate * 3
            
            quality_scores = [q.get('quality_score', 5) for q in valid_pairs if 'quality_score' in q]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                quality_score = (avg_quality / 10) * 3
            else:
                quality_score = 1.5
            
            qa_score = min(qa_count_score + solve_score + quality_score, 10)
        else:
            qa_score = 0
        
        # 分享价值分数
        share_count = sum(1 for msg in messages if msg.get('message_type', 0) == 99)
        share_score = min(share_count / 2 * 5, 5)
        discuss_score = min(share_count / 3 * 5, 5)
        share_value_score = min(share_score + discuss_score, 10)
        
        # 时效性分数（默认7分）
        timeliness_score = 7.0
        
        # 加权总分
        weighted_score = (
            engagement_score * self.weights['engagement'] +
            depth_score * self.weights['depth'] +
            qa_score * self.weights['qa_quality'] +
            share_value_score * self.weights['share_value'] +
            timeliness_score * self.weights['timeliness']
        )
        
        return {
            'engagement_score': round(engagement_score, 2),
            'depth_score': round(depth_score, 2),
            'qa_score': round(qa_score, 2),
            'share_score': round(share_value_score, 2),
            'timeliness_score': round(timeliness_score, 2),
            'weighted_score': round(weighted_score, 2)
        }
    
    def batch_score(self, topics: List[Dict], qa_pairs_list: List[List[Dict]] = None,
                    use_llm: bool = True) -> List[Dict]:
        """
        批量评估话题重要度
        
        Args:
            topics: 话题列表
            qa_pairs_list: QA对列表
            use_llm: 是否使用LLM
            
        Returns:
            评分结果列表
        """
        results = []
        
        for i, topic in enumerate(topics):
            print(f"  评估话题 {i+1}/{len(topics)}...")
            
            qa_pairs = qa_pairs_list[i] if qa_pairs_list and i < len(qa_pairs_list) else None
            
            result = self.score(topic, qa_pairs, use_llm)
            result['topic_id'] = topic.get('topic_id', i + 1)
            
            results.append(result)
            
            if result.get('pass_threshold'):
                print(f"    ✅ 通过: {result['importance_score']}分")
            else:
                print(f"    ⏭️  未通过: {result['importance_score']}分")
        
        return results
    
    def get_topics_by_threshold(self, scores: List[Dict], 
                                 threshold: float = None) -> List[Dict]:
        """
        根据阈值筛选话题
        
        Args:
            scores: 评分结果列表
            threshold: 阈值
            
        Returns:
            通过阈值的话题列表
        """
        if threshold is None:
            threshold = self.threshold
        
        return [score for score in scores if score.get('pass_threshold', False)]
    
    def get_statistics(self, scores: List[Dict]) -> Dict:
        """
        获取评分统计信息
        
        Args:
            scores: 评分结果列表
            
        Returns:
            统计信息
        """
        if not scores:
            return {}
        
        scores_list = [s.get('importance_score', 0) for s in scores]
        
        llm_count = sum(1 for s in scores if s.get('scored_by') == 'llm')
        rules_count = sum(1 for s in scores if s.get('scored_by') == 'rules')
        
        return {
            'total_topics': len(scores),
            'passed_topics': sum(1 for s in scores if s.get('pass_threshold', False)),
            'failed_topics': sum(1 for s in scores if not s.get('pass_threshold', False)),
            'average_score': round(sum(scores_list) / len(scores_list), 2),
            'max_score': max(scores_list),
            'min_score': min(scores_list),
            'pass_rate': round(sum(1 for s in scores if s.get('pass_threshold', False)) / len(scores) * 100, 2),
            'scored_by_llm': llm_count,
            'scored_by_rules': rules_count
        }


# ==================== 便捷函数 ====================

def calculate_importance(topic_info: Dict, qa_pairs: List[Dict] = None,
                         config_path: str = "config.yaml",
                         use_llm: bool = True) -> Dict:
    """
    便捷的重要度评估函数
    
    Args:
        topic_info: 话题信息
        qa_pairs: QA对列表
        config_path: 配置文件路径
        use_llm: 是否使用LLM
        
    Returns:
        评分结果
    """
    scorer = ImportanceScorer(config_path)
    return scorer.score(topic_info, qa_pairs, use_llm)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试重要度评分器（增强版）...")
    
    # 创建测试数据
    test_topic = {
        'title': 'AI编程助手使用技巧',
        'summary': '讨论Claude、Cursor等AI编程工具的使用技巧',
        'message_count': 15,
        'participant_count': 5,
        'messages': [
            {'content': '大家好，请问怎么使用Claude Code？', 'message_type': 0},
            {'content': 'Claude Code是一个AI编程助手，可以帮助编写代码', 'message_type': 0},
            {'content': '能举个例子吗？', 'message_type': 0},
            {'content': '```python\ndef example():\n    print("Hello")\n```', 'message_type': 0},
            {'content': 'https://claude.com', 'message_type': 99}
        ]
    }
    
    test_qa_pairs = [
        {
            'question_content': '怎么使用Claude Code？',
            'answer_content': 'Claude Code是一个AI编程助手',
            'is_valid': True,
            'solved': True,
            'quality_score': 8
        }
    ]
    
    # 初始化评分器
    scorer = ImportanceScorer()
    print("✅ 重要度评分器初始化成功")
    
    # 测试（使用规则作为备选）
    result = scorer.score(test_topic, test_qa_pairs, use_llm=False)
    print(f"✅ 评估完成: {result}")
    print(f"   重要度分数: {result['importance_score']}")
    print(f"   通过阈值: {result['pass_threshold']}")
    print(f"   评分方式: {result.get('scored_by', 'N/A')}")
