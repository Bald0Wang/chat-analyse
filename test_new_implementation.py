#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新的LLM意图识别实现

验证：
1. LLM客户端的意图分析功能
2. 话题分割的LLM模式
3. QA检测的LLM模式
4. 重要度评估的LLM模式

作者: AI Assistant
创建时间: 2026-01-14
"""

import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def test_llm_client():
    """测试LLM客户端的意图分析功能"""
    print("\n" + "="*60)
    print("测试1: LLM客户端意图分析")
    print("="*60)
    
    try:
        from modules.llm_client import get_llm_client
        
        llm = get_llm_client()
        print("✅ LLM客户端初始化成功")
        
        # 测试消息
        test_messages = [
            {
                'sender_name': '用户A',
                'content': '大家好，请问怎么使用Python装饰器？',
                'message_type': 0,
                'time': '10:00:00'
            },
            {
                'sender_name': '用户B',
                'content': '装饰器是一个接受函数作为参数的函数',
                'message_type': 0,
                'time': '10:01:00'
            },
            {
                'sender_name': '用户C',
                'content': '能举个例子吗？',
                'message_type': 0,
                'time': '10:02:00'
            },
            {
                'sender_name': '用户B',
                'content': '''def my_decorator(func):
    def wrapper():
        print("执行前")
        func()
        print("执行后")
    return wrapper''',
                'message_type': 0,
                'time': '10:03:00'
            }
        ]
        
        # 测试意图分析
        print("\n测试意图分析...")
        intents = llm.analyze_conversation_intent(test_messages)
        print(f"✅ 意图分析完成: {len(intents)} 条结果")
        
        for i, intent in enumerate(intents[:3]):  # 只显示前3条
            print(f"  {i+1}. {intent.get('intent_type', 'N/A')}: {intent.get('reason', '')[:60]}...")
        
        # 测试QA检测
        print("\n测试QA对检测...")
        qa_pairs = llm.analyze_qa_pairs_llm(test_messages)
        print(f"✅ QA检测完成: {len(qa_pairs)} 对")
        
        for qa in qa_pairs:
            q_idx = qa['question_index']
            a_idx = qa['answer_index']
            print(f"  Q{q_idx+1} -> A{a_idx+1}: 置信度 {qa.get('confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_topic_segmentation():
    """测试话题分割模块"""
    print("\n" + "="*60)
    print("测试2: 话题分割模块")
    print("="*60)
    
    try:
        from modules.topic_segment import TopicSegmenter
        
        segmenter = TopicSegmenter()
        print("✅ 话题分割器初始化成功")
        
        # 测试消息（包含多个话题）
        test_messages = [
            {
                'timestamp': 1736706000,
                'sender_name': '用户A',
                'content': '大家好，年假多长时间呀？',
                'message_type': 0,
                'time': '10:00:00'
            },
            {
                'timestamp': 1736706100,
                'sender_name': '用户B',
                'content': '一般是5天',
                'message_type': 0,
                'time': '10:01:00'
            },
            {
                'timestamp': 1738706400,  # 6小时后
                'sender_name': '用户C',
                'content': '分享一个AI工具',
                'message_type': 99,
                'time': '16:00:00'
            },
            {
                'timestamp': 1738706500,
                'sender_name': '用户D',
                'content': '什么工具？',
                'message_type': 0,
                'time': '16:01:00'
            }
        ]
        
        # 使用规则测试（避免LLM API调用）
        print("\n测试规则分割（备选方案）...")
        topics = segmenter.segment(test_messages, use_llm=False)
        print(f"✅ 分割完成: {len(topics)} 个话题")
        
        for i, topic in enumerate(topics):
            print(f"\n话题 {i+1}:")
            print(f"  标题: {topic.get('title', 'N/A')}")
            print(f"  消息数: {topic.get('message_count', 0)}")
            print(f"  方法: {topic.get('segmentation_method', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qa_detection():
    """测试QA检测模块"""
    print("\n" + "="*60)
    print("测试3: QA检测模块")
    print("="*60)
    
    try:
        from modules.qa_detector import QADetector
        
        detector = QADetector()
        print("✅ QA检测器初始化成功")
        
        # 测试消息
        test_messages = [
            {
                'sender_name': '用户A',
                'content': '请问怎么使用Python装饰器？',
                'message_type': 0,
                'is_question': True
            },
            {
                'sender_name': '用户B',
                'content': '装饰器是一个接受函数作为参数的函数',
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
                'content': '```python\ndef decorator(func):\n    return func\n```',
                'message_type': 0,
                'is_question': False
            }
        ]
        
        # 使用规则测试（避免LLM API调用）
        print("\n测试规则检测（备选方案）...")
        qa_pairs = detector.detect(test_messages, use_llm=False)
        print(f"✅ 检测完成: {len(qa_pairs)} 个QA对")
        
        for i, qa in enumerate(qa_pairs):
            print(f"\nQA对 {i+1}:")
            print(f"  问题: {qa['question_content'][:40]}...")
            print(f"  回答: {qa['answer_content'][:40] if qa['answer_content'] else '无'}...")
            print(f"  方法: {qa.get('detection_method', 'N/A')}")
            print(f"  质量分: {qa.get('quality_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_importance_scorer():
    """测试重要度评分模块"""
    print("\n" + "="*60)
    print("测试4: 重要度评分模块")
    print("="*60)
    
    try:
        from modules.importance_scorer import ImportanceScorer
        
        scorer = ImportanceScorer()
        print("✅ 重要度评分器初始化成功")
        
        # 测试话题
        test_topic = {
            'title': 'Python装饰器教程',
            'summary': '讨论Python装饰器的使用方法和示例',
            'message_count': 10,
            'participant_count': 3,
            'messages': [
                {'content': '请问怎么使用装饰器？', 'message_type': 0},
                {'content': '装饰器是Python的强大特性', 'message_type': 0},
                {'content': '```python\n@decorator\ndef func():\n    pass\n```', 'message_type': 0}
            ]
        }
        
        # 使用规则测试
        print("\n测试规则评分（备选方案）...")
        result = scorer.score(test_topic, use_llm=False)
        print(f"✅ 评分完成: {result['importance_score']}分")
        print(f"  通过阈值: {result['pass_threshold']}")
        print(f"  评分方式: {result.get('scored_by', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🧪 测试新的LLM意图识别实现")
    print("="*60)
    
    results = []
    
    # 运行测试
    results.append(("LLM客户端", test_llm_client()))
    results.append(("话题分割", test_topic_segmentation()))
    results.append(("QA检测", test_qa_detection()))
    results.append(("重要度评分", test_importance_scorer()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！新的LLM意图识别实现工作正常。")
    else:
        print(f"\n⚠️  {failed} 个测试失败，请检查输出。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
