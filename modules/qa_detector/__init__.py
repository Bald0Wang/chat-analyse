"""
QA问答对识别模块

功能：
- 基于规则的问题检测
- LLM辅助的QA对识别

作者: AI Assistant
创建时间: 2026-01-13
"""

from .detector import QADetector, detect_qa_pairs

__all__ = ['QADetector', 'detect_qa_pairs']
