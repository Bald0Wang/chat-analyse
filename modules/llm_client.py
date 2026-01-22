"""
LLM 客户端模块 (增强版)
支持智谱AI，增加批量分析和意图识别功能

作者: AI Assistant
创建时间: 2026-01-13
"""

import json
import os
import yaml
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from pathlib import Path


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送对话请求"""
        pass
    
    @abstractmethod
    def extract_json(self, prompt: str) -> Dict[str, Any]:
        """提取JSON格式响应"""
        pass


class ZhipuAIClient(BaseLLMClient):
    """智谱AI客户端"""
    
    def __init__(self, api_key: str, model: str = "glm-4.7", 
                 max_tokens: int = 16384, temperature: float = 0.7,
                 print_responses: bool = False,
                 timeout: int = 60):
        """
        初始化智谱AI客户端
        
        Args:
            api_key: API密钥
            model: 模型名称
            max_tokens: 最大输出token数
            temperature: 温度参数
            timeout: 超时时间（秒）
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.print_responses = print_responses
        
        # 智谱AI API地址
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        
        # 请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            **kwargs: 其他参数（thinking, temperature, max_tokens等）
            
        Returns:
            assistant的回复文本
        """
        import requests
        
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        # 添加深度思考模式（如果需要）
        if kwargs.get("thinking"):
            data["thinking"] = {
                "type": "enabled"
            }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            if self.print_responses:
                print("\n[LLM][response]\n" + str(content) + "\n")
            return content
            
        except requests.exceptions.Timeout:
            raise Exception(f"请求超时（>{self.timeout}秒）")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {str(e)}")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析响应失败: {str(e)}")
    
    def extract_json(self, prompt: str) -> Dict[str, Any]:
        """
        提取JSON格式响应
        
        Args:
            prompt: 提示词
            
        Returns:
            解析后的JSON字典
        """
        messages = [
            {"role": "system", "content": "你是一个JSON提取专家。请始终输出有效的JSON格式，不要包含其他文字。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, temperature=0.1)
        
        # 尝试解析JSON
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从markdown代码块中提取
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取 {...} 格式
            curly_match = re.search(r'\{[\s\S]*\}', response)
            if curly_match:
                try:
                    return json.loads(curly_match.group())
                except json.JSONDecodeError:
                    pass
            
            raise Exception(f"无法解析JSON响应: {response[:200]}...")


class DoubaoClient(BaseLLMClient):
    """豆包客户端（OpenAI SDK）"""

    def __init__(self, api_key_env: str, base_url: str, model: str,
                 api_key: Optional[str] = None,
                 max_tokens: int = 16384, temperature: float = 0.7,
                 print_responses: bool = False,
                 timeout: int = 60):
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env, "")
        if not self.api_key:
            raise Exception(f"未设置环境变量 {api_key_env}")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.print_responses = print_responses

        try:
            from openai import OpenAI
        except ImportError as e:
            raise Exception("缺少依赖 openai，请先安装：pip install openai") from e

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        data = {
            "model": kwargs.get("model") or self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        try:
            response = self.client.chat.completions.create(**data)
            content = response.choices[0].message.content
            if self.print_responses:
                print("\n[LLM][response]\n" + str(content) + "\n")
            return content
        except Exception as e:
            raise Exception(f"API请求失败: {str(e)}")

    def extract_json(self, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "你是一个JSON提取专家。请始终输出有效的JSON格式，不要包含其他文字。"},
            {"role": "user", "content": prompt}
        ]
        response = self.chat(messages, temperature=0.1)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            curly_match = re.search(r'\{[\s\S]*\}', response)
            if curly_match:
                try:
                    return json.loads(curly_match.group())
                except json.JSONDecodeError:
                    pass
            raise Exception(f"无法解析JSON响应: {response[:200]}...")


class LLMManager:
    """LLM管理器 - 统一管理不同LLM客户端（增强版）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化LLM管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.llm_config = self.config.get('llm', {})
        self.client = self._init_client()
        self.retries = int(self.llm_config.get('retries', 0))
        self.retry_backoff = float(self.llm_config.get('retry_backoff', 0))
        self.debug_config = self.config.get('debug', {})
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise Exception(f"配置文件不存在: {config_path}")
        except yaml.YAMLError as e:
            raise Exception(f"配置文件解析错误: {e}")
    
    def _init_client(self) -> BaseLLMClient:
        """初始化LLM客户端"""
        llm_config = self.config.get('llm', {})
        provider = llm_config.get('provider', 'zhipu')
        
        if provider == 'zhipu':
            return ZhipuAIClient(
                api_key=llm_config.get('api_key', ''),
                model=llm_config.get('model', 'glm-4.7'),
                max_tokens=llm_config.get('max_tokens', 16384),
                temperature=llm_config.get('temperature', 0.7),
                print_responses=bool(self.config.get('debug', {}).get('print_responses', False)),
                timeout=llm_config.get('timeout', 60)
            )
        if provider == 'doubao':
            return DoubaoClient(
                api_key_env=llm_config.get('api_key_env', 'ARK_API_KEY'),
                base_url=llm_config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3'),
                model=llm_config.get('model', 'doubao-seed-1-6-flash-250828'),
                api_key=llm_config.get('api_key'),
                max_tokens=llm_config.get('max_tokens', 16384),
                temperature=llm_config.get('temperature', 0.7),
                print_responses=bool(self.config.get('debug', {}).get('print_responses', False)),
                timeout=llm_config.get('timeout', 60)
            )
        else:
            raise Exception(f"不支持的LLM提供商: {provider}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送对话请求"""
        attempts = self.retries + 1
        last_error = None
        model_override = self._select_model_for_messages(messages)
        messages = self._truncate_messages(messages)
        for attempt in range(attempts):
            try:
                if model_override:
                    kwargs["model"] = model_override
                content = self.client.chat(messages, **kwargs)
                self._append_log("LLM.response", content)
                return content
            except Exception as e:
                last_error = e
                if attempt >= attempts - 1:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))
        raise last_error
    
    def extract_json(self, prompt: str) -> Dict[str, Any]:
        """提取JSON格式响应"""
        attempts = self.retries + 1
        last_error = None
        model_override = self._select_model_for_prompt(prompt)
        prompt = self._apply_context_cap(prompt)
        for attempt in range(attempts):
            try:
                if model_override:
                    response = self.client.chat([
                        {"role": "system", "content": "你是一个JSON提取专家。请始终输出有效的JSON格式，不要包含其他文字。"},
                        {"role": "user", "content": prompt}
                    ], temperature=0.1, model=model_override)
                    result = self._parse_json_from_response(response)
                else:
                    result = self.client.extract_json(prompt)
                self._append_log("LLM.response.json", json.dumps(result, ensure_ascii=False))
                return result
            except Exception as e:
                last_error = e
                if attempt >= attempts - 1:
                    break
                time.sleep(self.retry_backoff * (attempt + 1))
        raise last_error

    def _append_log(self, label: str, content: str) -> None:
        if not self.debug_config.get('log_to_file', False):
            return
        log_path = self.debug_config.get('log_path')
        if not log_path:
            return
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().isoformat()
            with path.open('a', encoding='utf-8') as f:
                f.write(f"\n[{timestamp}] {label}\n{content}\n")
        except Exception:
            pass

    def _select_model_for_prompt(self, prompt: str) -> Optional[str]:
        max_chars = int(self.llm_config.get('max_context_chars', 32000))
        if len(prompt) > max_chars:
            if not self.llm_config.get('allow_long_context_model', True):
                return None
            return self.llm_config.get('long_context_model')
        return None

    def _select_model_for_messages(self, messages: List[Dict[str, str]]) -> Optional[str]:
        max_chars = int(self.llm_config.get('max_context_chars', 32000))
        total = sum(len(m.get('content', '') or '') for m in messages)
        if total > max_chars:
            if not self.llm_config.get('allow_long_context_model', True):
                return None
            return self.llm_config.get('long_context_model')
        return None

    def _apply_context_cap(self, prompt: str) -> str:
        max_chars = int(self.llm_config.get('max_context_chars', 32000))
        if len(prompt) <= max_chars:
            return prompt
        head_len = min(2000, max_chars // 4)
        tail_len = max_chars - head_len - 40
        if tail_len <= 0:
            return prompt[-max_chars:]
        return prompt[:head_len] + "\n... [内容过长已截断] ...\n" + prompt[-tail_len:]

    def _truncate_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        max_chars = int(self.llm_config.get('max_context_chars', 32000))
        total = sum(len(m.get('content', '') or '') for m in messages)
        if total <= max_chars or not messages:
            return messages
        trimmed = [dict(m) for m in messages]
        overflow = total - max_chars
        # 只截断最后一条用户消息内容
        last = trimmed[-1]
        content = last.get('content', '') or ''
        if overflow < len(content):
            last['content'] = content[overflow:]
        else:
            last['content'] = content[-max_chars:]
        trimmed[-1] = last
        return trimmed

    def _parse_json_from_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            curly_match = re.search(r'\{[\s\S]*\}', response)
            if curly_match:
                try:
                    return json.loads(curly_match.group())
                except json.JSONDecodeError:
                    pass
            raise Exception(f"无法解析JSON响应: {response[:200]}...")
    
    # ==================== 增强功能：意图识别 ====================
    
    def analyze_conversation_intent(self, messages: List[Dict], 
                                     window_size: int = 15) -> List[Dict]:
        """
        批量分析对话意图（核心功能）
        
        使用LLM理解对话上下文，识别：
        - 每个消息的意图类型
        - 话题边界
        - 问题-回答关系
        
        Args:
            messages: 消息列表
            window_size: 分析窗口大小
            
        Returns:
            意图分析结果列表
        """
        if not messages:
            return []
        
        results = []
        
        # 分批处理（避免超出LLM上下文窗口）
        batch_size = window_size
        total_batches = (len(messages) + batch_size - 1) // batch_size
        
        async_enabled = bool(self.llm_config.get('async_enabled', False))
        max_workers = int(self.llm_config.get('async_max_workers', 2))
        stall_warning_seconds = int(self.llm_config.get('stall_warning_seconds', 45))

        # 说明：
        # - max_workers<=1 时不启用异步（否则 as_completed 反而容易“看起来卡住”且没有每批耗时）
        # - 异步模式下使用 wait(timeout=...) 周期性输出 pending 数，便于定位卡在哪个批次
        if async_enabled and total_batches > 1 and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
            futures = {}
            start_times: Dict[int, float] = {}
            ordered: Dict[int, List[Dict]] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(messages))
                    batch_messages = messages[start_idx:end_idx]
                    print(f"    分析批次 {batch_idx + 1}/{total_batches} (消息 {start_idx + 1}-{end_idx})...")
                    fut = executor.submit(self._analyze_batch_intent, batch_messages, start_idx)
                    futures[fut] = (batch_idx, batch_messages, start_idx)
                    start_times[batch_idx] = time.time()

                pending = set(futures.keys())
                while pending:
                    done, pending = wait(pending, timeout=stall_warning_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        # 一段时间没有任何批次完成，输出可视化提示
                        pending_indices = sorted(futures[f][0] for f in pending)
                        print(f"      ⏳ {stall_warning_seconds}s 内无批次完成，pending: {len(pending)}，批次: {pending_indices[:10]}{'...' if len(pending_indices) > 10 else ''}")
                        continue

                    for fut in done:
                        batch_idx, batch_messages, start_idx = futures[fut]
                        elapsed = time.time() - start_times.get(batch_idx, time.time())
                        try:
                            ordered[batch_idx] = fut.result()
                            print(f"      ✅ 批次 {batch_idx + 1}/{total_batches} 完成，用时 {elapsed:.1f}s")
                        except Exception as e:
                            print(f"      ⚠️ 批次 {batch_idx + 1}/{total_batches} 失败({elapsed:.1f}s): {e}，使用备选方案")
                            ordered[batch_idx] = self._fallback_intent_analysis(batch_messages, start_idx)

            for batch_idx in range(total_batches):
                results.extend(ordered.get(batch_idx, []))

        else:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(messages))
                batch_messages = messages[start_idx:end_idx]

                print(f"    分析批次 {batch_idx + 1}/{total_batches} (消息 {start_idx + 1}-{end_idx})...")
                batch_start = time.time()
                try:
                    batch_results = self._analyze_batch_intent(batch_messages, start_idx)
                    results.extend(batch_results)
                    elapsed = time.time() - batch_start
                    print(f"      ✅ 批次完成，用时 {elapsed:.1f}s")
                except Exception as e:
                    elapsed = time.time() - batch_start
                    print(f"      ⚠️ 批次分析失败({elapsed:.1f}s): {e}，使用备选方案")
                    batch_results = self._fallback_intent_analysis(batch_messages, start_idx)
                    results.extend(batch_results)

                if elapsed > stall_warning_seconds:
                    print(f"      ⏳ 警告: 批次耗时超过 {stall_warning_seconds}s，可能存在卡顿或网络问题")
        
        return results
    
    def _analyze_batch_intent(self, batch_messages: List[Dict], 
                              start_offset: int = 0) -> List[Dict]:
        """
        使用LLM分析一批消息的意图
        
        Args:
            batch_messages: 消息批次
            start_offset: 起始索引偏移
            
        Returns:
            意图分析结果
        """
        # 构建消息文本
        messages_text = []
        for i, msg in enumerate(batch_messages):
            time_str = msg.get('time', '')[:8]  # 只取时间部分
            sender = msg.get('sender_name', 'Unknown')
            content = msg.get('content', '')[:150]  # 限制长度
            msg_type = msg.get('message_type', 0)
            
            # 标记特殊类型
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
你是一个专业的对话分析专家。请分析以下微信群聊片段的对话意图和结构：

{dialogue}

请为每条消息分析以下内容：
1. 消息意图类型（question/answer/share/chat/system/other）
2. 是否开启新话题
3. 是否回答了前面的问题
4. 话题相关性（与上下文的关联程度0-10）

输出JSON格式（每条消息一行，不要用代码块包裹）：
[
  {{
    "index": 消息序号（从1开始）,
    "intent_type": "意图类型",
    "is_new_topic": true/false,
    "is_answer": true/false,
    "question_index": 对应问题序号（如果是回答）,
    "topic_relevance": 0-10,
    "confidence": 0.0-1.0,
    "reason": "分析理由"
  }}
]

意图类型说明：
- question: 提问、求助、咨询
- answer: 回答、解答、回应
- share: 分享链接、视频、文章
- chat: 闲聊、讨论、评论
- system: 系统通知、群管理消息
- other: 其他

重要规则：
1. 如果一条消息回答了前面的问题，标记is_answer=true，并指明回答了哪个问题
2. 如果话题发生明显切换（如从闲聊到技术讨论），标记is_new_topic=true
3. 每条消息都要给出分析和理由
4. 如果不确定，confidence设置低一些
"""
        prompt = self._apply_context_cap(prompt)
        
        if self.llm_config.get('debug', {}).get('print_prompts', False):
            print("\n[Prompt][llm_client][intent]\n" + prompt.strip() + "\n")
        if self.debug_config.get('log_to_file', False):
            self._append_log("Prompt.llm_client.intent", prompt.strip())
        
        response = self.chat([
            {"role": "system", "content": "你是一个专业的对话分析专家，擅长理解对话上下文和意图。"},
            {"role": "user", "content": prompt}
        ], temperature=0.2)
        
        # 解析JSON结果
        import re
        
        # 提取JSON数组
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                results = json.loads(json_match.group())
                # 添加全局索引
                for r in results:
                    r['global_index'] = start_offset + r['index'] - 1
                return results
            except json.JSONDecodeError:
                pass
        
        # 备选：简单解析
        print(f"      ⚠️ JSON解析失败，使用简单解析")
        return self._fallback_intent_analysis(batch_messages, start_offset)
    
    def _fallback_intent_analysis(self, batch_messages: List[Dict], 
                                   start_offset: int = 0) -> List[Dict]:
        """
        备选意图分析（基于简单规则）
        当LLM分析失败时使用
        
        Args:
            batch_messages: 消息批次
            start_offset: 起始索引偏移
            
        Returns:
            意图分析结果
        """
        results = []
        
        # 简单规则判断
        question_markers = ['?', '怎么', '如何', '请问', '为什么', '?', 
                          '求助', '求教', '报错', '问题']
        
        share_markers = [99]  # type=99是分享链接
        system_markers = [80]  # type=80是系统消息
        
        for i, msg in enumerate(batch_messages):
            content = msg.get('content', '')
            msg_type = msg.get('message_type', 0)
            
            # 判断意图类型
            if msg_type in system_markers:
                intent = 'system'
            elif msg_type in share_markers or content.startswith('<?xml'):
                intent = 'share'
            elif any(marker in content for marker in question_markers):
                intent = 'question'
            else:
                intent = 'chat'
            
            result = {
                'index': i + 1,
                'intent_type': intent,
                'is_new_topic': False,
                'is_answer': False,
                'question_index': None,
                'topic_relevance': 7.0,
                'confidence': 0.6,  # 备选方案置信度较低
                'reason': f"基于规则分析: {intent}"
            }
            
            result['global_index'] = start_offset + i
            results.append(result)
        
        return results
    
    def detect_topic_boundaries_llm(self, messages: List[Dict], window_size: int = 15) -> List[Dict]:
        """
        使用LLM检测话题边界（主要方法）
        
        Args:
            messages: 消息列表
            
        Returns:
            话题边界信息列表
        """
        if not messages:
            return []
        
        # 先进行意图分析
        intent_results = self.analyze_conversation_intent(messages, window_size=window_size)
        
        # 根据意图分析结果确定话题边界
        topic_boundaries = []
        current_topic_start = 0
        
        for i, intent in enumerate(intent_results):
            # 如果标记为新话题，或者相关性很低
            is_new_topic = bool(intent.get('is_new_topic', False))
            relevance_raw = intent.get('topic_relevance', 10)
            try:
                relevance = float(relevance_raw) if relevance_raw is not None else 10.0
            except (TypeError, ValueError):
                relevance = 10.0

            if is_new_topic or relevance < 4:
                if i > current_topic_start:
                    # 保存前一个话题的边界
                    topic_boundaries.append({
                        'start_index': current_topic_start,
                        'end_index': i - 1,
                        'transition_reason': intent.get('reason', '话题切换')
                    })
                current_topic_start = i
        
        # 添加最后一个话题
        if current_topic_start < len(messages):
            topic_boundaries.append({
                'start_index': current_topic_start,
                'end_index': len(messages) - 1,
                'transition_reason': '对话结束'
            })
        
        return topic_boundaries
    
    def analyze_qa_pairs_llm(self, messages: List[Dict]) -> List[Dict]:
        """
        使用LLM分析问答对（主要方法）
        
        Args:
            messages: 消息列表
            
        Returns:
            QA对列表
        """
        if not messages:
            return []
        
        # 先进行意图分析
        intent_results = self.analyze_conversation_intent(messages)
        
        # 根据意图分析结果提取QA对
        qa_pairs = []
        pending_questions = []  # 待回答的问题
        
        for i, (msg, intent) in enumerate(zip(messages, intent_results)):
            # 如果是问题
            if intent.get('intent_type') == 'question':
                pending_questions.append({
                    'index': i,
                    'message': msg,
                    'intent': intent
                })
            
            # 如果是回答
            elif intent.get('intent_type') == 'answer' and intent.get('question_index'):
                q_idx = intent['question_index'] - 1  # 转为0基索引
                
                # 找到对应的提问
                for q in pending_questions:
                    if q['index'] == q_idx:
                        # 构建QA对
                        qa_pair = {
                            'question_index': q_idx,
                            'answer_index': i,
                            'question': msg,
                            'answer': messages[i],
                            'confidence': intent.get('confidence', 0.5),
                            'reason': f"LLM识别: {intent.get('reason', '')}"
                        }
                        qa_pairs.append(qa_pair)
                        
                        # 从待回答列表中移除
                        pending_questions.remove(q)
                        break
        
        # 添加没有回答的问题（可选）
        for q in pending_questions:
            qa_pair = {
                'question_index': q['index'],
                'answer_index': None,
                'question': q['message'],
                'answer': None,
                'confidence': q['intent'].get('confidence', 0.5),
                'reason': f"问题未得到回答: {q['intent'].get('reason', '')}"
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    # ==================== 原有功能（保持兼容） ====================
    
    def analyze_topic(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        分析话题（包装后的方法，保持原有接口）
        
        Args:
            messages: 消息列表
            
        Returns:
            分析结果字典
        """
        prompt = self._build_topic_analysis_prompt(messages)
        return self.extract_json(prompt)
    
    def _build_topic_analysis_prompt(self, messages: List[Dict]) -> str:
        """构建话题分析提示词"""
        display_messages = messages[:30]
        
        messages_text = "\n".join([
            f"[{msg.get('time', '')}] {msg.get('sender', '')}: {msg.get('content', '')}"
            for msg in display_messages
        ])
        
        prompt = f"""
你是一个技术群聊话题检测专家。请分析以下微信群聊片段：

{messages_text}

任务：
1. 判断这段是否是一个独立完整的话题。如果前后明显切换了主题，请指出切换点。
2. 指出本段的"话题启动消息"（通常是提问、分享链接/视频、突然引入新主题的消息），并说明理由。
3. 给话题起一个简短标题（10字以内）。
4. 简要总结话题内容（50字以内）。

输出JSON格式：
{{
  "is_new_topic": true/false,
  "start_index": X,
  "title": "话题标题",
  "summary": "话题简要总结",
  "reason": "为什么这是新话题/不是新话题的理由"
}}
"""
        return prompt
    
    def score_importance(self, topic_info: Dict, **kwargs) -> Dict[str, Any]:
        """
        评估话题重要度
        
        Args:
            topic_info: 话题信息
            
        Returns:
            评分结果
        """
        prompt = self._build_importance_prompt(topic_info)
        return self.extract_json(prompt)
    
    def _build_importance_prompt(self, topic_info: Dict) -> str:
        """构建重要度评估提示词"""
        prompt = f"""
你是技术群话题重要度评估专家。请评估以下话题的重要程度（0-10分）：

话题标题：{topic_info.get('title', '未知')}
话题摘要：{topic_info.get('summary', '无')}
参与人数：{topic_info.get('participant_count', 0)}
消息数量：{topic_info.get('message_count', 0)}
是否有链接/视频分享：{topic_info.get('has_media', '未知')}
是否有问答互动：{topic_info.get('has_qa', '未知')}

评分维度：
- 参与度（30%）：话题内回复数、参与人数
- 内容深度（25%）：是否有代码、链接、详细解释、技术关键词
- QA质量（20%）：是否形成清晰问答，回答是否有实际解决方案
- 分享价值（15%）：是否包含视频/链接/文件，且被多人讨论
- 时效性/热度（10%）：近期话题加分

请给出0-10分的评分，并说明理由。输出JSON格式：
{{
  "score": X.X,
  "engagement_score": X.X,
  "depth_score": X.X,
  "qa_score": X.X,
  "share_score": X.X,
  "timeliness_score": X.X,
  "reason": "评分理由...",
  "highlights": ["亮点1", "亮点2"],
  "suggestions": ["改进建议1"]
}}
"""
        return prompt
    
    def generate_note(self, topic_info: Dict, messages: List[Dict], **kwargs) -> str:
        """
        生成话题笔记
        
        Args:
            topic_info: 话题信息
            messages: 消息列表
            
        Returns:
            Markdown格式的笔记内容
        """
        prompt = self._build_note_prompt(topic_info, messages)
        return self.chat([
            {"role": "system", "content": "你是一个技术文档整理专家。请将群聊内容整理成结构化笔记。"},
            {"role": "user", "content": prompt}
        ], temperature=0.5)
    
    def _build_note_prompt(self, topic_info: Dict, messages: List[Dict]) -> str:
        """构建笔记生成提示词"""
        core_messages = []
        for msg in messages[:50]:
            if msg.get('type') in [0, 1]:
                core_messages.append(f"- [{msg.get('time', '')}] {msg.get('sender', '')}: {msg.get('content', '')[:200]}")
        
        messages_text = "\n".join(core_messages)
        
        prompt = f"""
请将以下技术群话题总结成一份结构化笔记：

话题标题：{topic_info.get('title', '未知话题')}
话题摘要：{topic_info.get('summary', '无')}

核心消息：
{messages_text}

要求：
1. 核心问题/分享：一句话描述触发点
2. 关键讨论点：列出3-8条要点
3. 推荐资源：列出所有链接、视频描述
4. 结论/最佳实践：如果有共识，总结出来
5. 相关关键词：5-10个标签

输出Markdown格式，适合存入知识库。
"""
        return prompt


# ==================== 便捷函数 ====================

def get_llm_client(config_path: str = "config.yaml") -> LLMManager:
    """
    获取LLM客户端实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LLMManager实例
    """
    return LLMManager(config_path)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("测试 LLM 客户端（增强版）...")
    
    try:
        llm = get_llm_client()
        print("✅ LLM客户端初始化成功")
        
        # 测试意图分析
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
            }
        ]
        
        print("\n测试意图分析...")
        intents = llm.analyze_conversation_intent(test_messages)
        print(f"✅ 意图分析完成: {len(intents)} 条结果")
        
        for intent in intents:
            print(f"  {intent['intent_type']}: {intent.get('reason', '')[:50]}...")
        
        # 测试QA对检测
        print("\n测试QA对检测...")
        qa_pairs = llm.analyze_qa_pairs_llm(test_messages)
        print(f"✅ QA对检测完成: {len(qa_pairs)} 对")
        
        for qa in qa_pairs:
            print(f"  Q: {qa['question_index']} -> A: {qa['answer_index']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
