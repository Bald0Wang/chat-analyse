"""
聊天记录分析系统 - 模块包

包含：
- preprocessor: 数据预处理
- topic_segment: 话题分割
- qa_detector: QA问答对识别
- importance_scorer: 重要度评估
- note_generator: 笔记生成

作者: AI Assistant
创建时间: 2026-01-13
"""

from .llm_client import LLMManager, get_llm_client, ZhipuAIClient

__all__ = ['LLMManager', 'get_llm_client', 'ZhipuAIClient']
