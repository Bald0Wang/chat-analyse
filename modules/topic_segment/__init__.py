"""
话题分割模块

功能：
- 基于时间间隔的粗分割
- LLM细分割和话题启动检测

作者: AI Assistant
创建时间: 2026-01-13
"""

from .segmenter import TopicSegmenter, segment_topics

__all__ = ['TopicSegmenter', 'segment_topics']
