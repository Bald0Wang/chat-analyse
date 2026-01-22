"""
数据预处理模块 - 清洗器

功能：
1. 读取JSONL格式原始数据
2. 解析XML内容（提取纯文本、链接、视频等）
3. 转换为结构化格式
4. 保存到数据库

作者: AI Assistant
创建时间: 2026-01-13
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化数据清洗器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.preprocess_config = self.config.get('preprocessor', {})
        self.link_records: List[Dict[str, Any]] = []
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def clean_file(self, input_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        清洗整个文件
        
        Args:
            input_path: 输入JSONL文件路径
            output_path: 可选，输出SQLite数据库路径
            
        Returns:
            清洗后的消息列表
        """
        messages = []
        
        # 读取并清洗每条消息
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    raw_data = json.loads(line.strip())
                    cleaned = self.clean_message(raw_data, line_num)
                    if cleaned:
                        messages.append(cleaned)
                except json.JSONDecodeError:
                    continue
        
        # 保存到数据库（如果指定）
        if output_path:
            self._save_to_database(messages, output_path)

        # 写入链接信息
        self._save_links()
        
        return messages
    
    def clean_message(self, raw_data: Dict, line_num: int = 0) -> Optional[Dict]:
        """
        清洗单条消息
        
        Args:
            raw_data: 原始数据字典
            line_num: 行号
            
        Returns:
            清洗后的消息字典，如果无效则返回None
        """
        # 跳过非消息类型
        if raw_data.get('_type') != 'message':
            return None
        
        # 解析时间戳
        timestamp = raw_data.get('timestamp', 0)
        if timestamp:
            try:
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                date_str = dt.strftime('%Y-%m-%d')
            except:
                time_str = ""
                date_str = ""
        else:
            time_str = ""
            date_str = ""
        
        # 解析内容
        content = raw_data.get('content', '')
        content_type = raw_data.get('type', 0)
        
        # 解析XML内容
        media_info = None
        if content.startswith('<?xml') or content.startswith('<msg'):
            media_info = self._parse_xml_content(content)
            # 提取纯文本内容
            content = media_info.get('text', content)
        
        # 构建清洗后的数据
        cleaned = {
            'raw_line': line_num,
            'timestamp': timestamp,
            'time': time_str,
            'date': date_str,
            'sender_id': raw_data.get('sender', ''),
            'sender_name': raw_data.get('accountName', 'Unknown'),
            'content': content.strip(),
            'message_type': content_type,
            'media_info': media_info,
            'is_question': self._is_question(content),
            'processed_at': datetime.now().isoformat()
        }
        
        # 解析链接：替换内容并记录链接信息
        if media_info:
            url = media_info.get('url', '')
            title = media_info.get('title', '')
            description = media_info.get('description', '')
            link_desc = title or description
            if url and link_desc:
                cleaned['content'] = f"【{link_desc}】<{url}>"
                self.link_records.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'sender_name': cleaned.get('sender_name', ''),
                    'timestamp': cleaned.get('timestamp', 0),
                    'date': cleaned.get('date', ''),
                    'raw_line': cleaned.get('raw_line', 0)
                })

        # 过滤无效消息
        if not cleaned['sender_name'] or not content.strip():
            return None
        
        return cleaned

    def _save_links(self):
        """
        将解析出的链接写入JSON文件
        """
        if not self.link_records:
            return
        dates = {r.get('date') for r in self.link_records if r.get('date')}
        if len(dates) == 1:
            date_str = next(iter(dates))
        else:
            date_str = 'mixed'
        output_dir = Path(self.config.get('paths', {}).get('processed_data', './data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"links_{date_str}.json"
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.link_records, f, ensure_ascii=False, indent=2)
            print(f"   🔗 保存链接信息: {filepath}")
        except Exception:
            pass
    
    def _parse_xml_content(self, xml_content: str) -> Dict[str, Any]:
        """
        解析XML格式内容
        
        Args:
            xml_content: XML格式的内容字符串
            
        Returns:
            解析后的媒体信息字典
        """
        result = {
            'text': '',
            'type': 'unknown',
            'title': '',
            'description': '',
            'url': '',
            'app_name': '',
            'thumb_url': '',
            'media_id': ''
        }
        
        try:
            # 解析XML
            root = ET.fromstring(xml_content)
            
            # 提取文本内容
            text_elem = root.find('.//plain')
            if text_elem is not None:
                result['text'] = text_elem.text or ''
            
            # 如果没有plain，尝试提取所有文本
            if not result['text']:
                result['text'] = ' '.join(root.itertext()).strip()
            
            # 检测消息类型并提取相应信息
            appmsg = root.find('appmsg')
            if appmsg is not None:
                result['type'] = 'app'
                result['title'] = appmsg.findtext('title', '')
                result['description'] = appmsg.findtext('des', '')
                result['app_name'] = appmsg.findtext('appname', '')
                
                # 提取URL
                url_elem = appmsg.find('url')
                if url_elem is not None:
                    result['url'] = url_elem.text or ''
                
                # 提取缩略图
                thumb_elem = appmsg.find('.//thumburl')
                if thumb_elem is not None:
                    result['thumb_url'] = thumb_elem.text or ''
                
                # 提取媒体ID
                media_id_elem = appmsg.find('.//mediaid')
                if media_id_elem is not None:
                    result['media_id'] = media_id_elem.text or ''
                
                # 根据app类型进一步分类
                msg_type = appmsg.findtext('type', '')
                if msg_type == '5':  # 分享链接
                    result['subtype'] = 'link'
                elif msg_type == '6':  # 附件
                    result['subtype'] = 'file'
                elif msg_type == '3':  # 图片
                    result['subtype'] = 'image'
                elif msg_type == '4':  # 语音
                    result['subtype'] = 'voice'
                else:
                    result['subtype'] = 'unknown'
            
            # 检测是否为引用消息
            replaysource = root.find('replaysource')
            if replaysource is not None:
                result['type'] = 'reply'
                result['text'] = f"引用回复: {result['text']}"
        
        except ET.ParseError:
            # XML解析失败，返回原始内容
            result['text'] = xml_content[:500] if len(xml_content) > 500 else xml_content
        
        return result
    
    def _is_question(self, content: str) -> bool:
        """
        判断内容是否为问题
        
        Args:
            content: 消息内容
            
        Returns:
            是否为问题的布尔值
        """
        if not content or not isinstance(content, str):
            return False
        
        # 检查结尾是否有问号
        if content.strip().endswith('?'):
            return True
        
        # 检查是否包含疑问词
        question_patterns = [
            '怎么', '如何', '请问', '为什么', '什么意思',
            '怎么做', '哪里', '什么', '能不能', '会不会',
            '求助', '求教', '报错', '错误', '问题'
        ]
        
        content_lower = content.lower()
        for pattern in question_patterns:
            if pattern in content:
                return True
        
        return False
    
    def _save_to_database(self, messages: List[Dict], db_path: str):
        """
        保存到SQLite数据库
        
        Args:
            messages: 消息列表
            db_path: 数据库路径
        """
        import sqlite3
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_line INTEGER,
                timestamp INTEGER,
                time TEXT,
                date TEXT,
                sender_id TEXT,
                sender_name TEXT,
                content TEXT,
                message_type INTEGER,
                is_question INTEGER,
                media_type TEXT,
                media_title TEXT,
                media_description TEXT,
                media_url TEXT,
                processed_at TEXT
            )
        ''')
        
        # 插入数据
        for msg in messages:
            cursor.execute('''
                INSERT INTO messages (
                    raw_line, timestamp, time, date, sender_id, sender_name,
                    content, message_type, is_question, media_type,
                    media_title, media_description, media_url, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                msg.get('raw_line', 0),
                msg.get('timestamp', 0),
                msg.get('time', ''),
                msg.get('date', ''),
                msg.get('sender_id', ''),
                msg.get('sender_name', ''),
                msg.get('content', ''),
                msg.get('message_type', 0),
                1 if msg.get('is_question') else 0,
                msg.get('media_info', {}).get('subtype', ''),
                msg.get('media_info', {}).get('title', ''),
                msg.get('media_info', {}).get('description', ''),
                msg.get('media_info', {}).get('url', ''),
                msg.get('processed_at', '')
            ))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self, messages: List[Dict]) -> Dict:
        """
        获取消息统计信息
        
        Args:
            messages: 消息列表
            
        Returns:
            统计信息字典
        """
        if not messages:
            return {}
        
        # 基本统计
        total = len(messages)
        
        # 按类型统计
        type_counts = {}
        for msg in messages:
            msg_type = msg.get('message_type', 0)
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        
        # 发送者统计
        sender_counts = {}
        for msg in messages:
            sender = msg.get('sender_name', 'Unknown')
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        # 日期统计
        date_counts = {}
        for msg in messages:
            date = msg.get('date', 'Unknown')
            date_counts[date] = date_counts.get(date, 0) + 1
        
        # 问题统计
        question_count = sum(1 for msg in messages if msg.get('is_question'))
        
        # 参与人数
        participant_count = len(sender_counts)
        
        return {
            'total_messages': total,
            'type_distribution': type_counts,
            'participant_count': participant_count,
            'top_senders': dict(sorted(sender_counts.items(), key=lambda x: -x[1])[:10]),
            'date_distribution': date_counts,
            'question_count': question_count,
            'question_ratio': round(question_count / total * 100, 2) if total > 0 else 0
        }


# ==================== 便捷函数 ====================

def clean_raw_data(input_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    便捷的数据清洗函数
    
    Args:
        input_path: 输入文件路径
        output_path: 可选输出数据库路径
        
    Returns:
        清洗后的消息列表
    """
    cleaner = DataCleaner()
    return cleaner.clean_file(input_path, output_path)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试数据清洗器...")
    
    # 初始化
    cleaner = DataCleaner()
    print("✅ 数据清洗器初始化成功")
    
    # 测试清洗单条消息
    test_data = {
        "_type": "message",
        "sender": "wxid_test",
        "accountName": "测试用户",
        "timestamp": 1736706350,
        "type": 0,
        "content": "你好，请问怎么使用这个功能？"
    }
    
    cleaned = cleaner.clean_message(test_data)
    if cleaned:
        print(f"✅ 单条消息清洗成功: {cleaned['sender']} - {cleaned['content'][:30]}...")
        print(f"   识别为问题: {cleaned['is_question']}")
    else:
        print("❌ 消息清洗失败")
