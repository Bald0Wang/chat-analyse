"""
笔记生成模块

功能：
- 生成结构化Markdown笔记
- 保存到文件和数据库

作者: AI Assistant
创建时间: 2026-01-13
"""

from .generator import NoteGenerator, generate_notes

__all__ = ['NoteGenerator', 'generate_notes']
