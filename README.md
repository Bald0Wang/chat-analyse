# 微信聊天记录分析系统

基于分层处理 + LLM 辅助的聊天记录分析方案。【注意：数据来源请从weflow或其他项目获取。本项目不提供数据】

## ✨ 功能特性

- **数据预处理**: 清洗JSONL格式，解析XML内容
- **话题分割**: 基于时间间隔 + LLM细分割
- **QA识别**: 自动识别问答对
- **重要度评估**: 多维度评分 (0-10分)
- **笔记生成**: 自动生成Markdown笔记

## 📁 项目结构

```
chat-analysis/
├── config.yaml              # 配置文件
├── main.py                  # 主程序入口
├── README.md                # 本文件
├── USAGE.md                 # 使用说明
│
├── modules/                 # 核心模块
│   ├── __init__.py
│   ├── llm_client.py       # LLM客户端
│   │
│   ├── preprocessor/       # 数据预处理
│   │   ├── __init__.py
│   │   └── cleaner.py
│   │
│   ├── topic_segment/      # 话题分割
│   │   ├── __init__.py
│   │   └── segmenter.py
│   │
│   ├── qa_detector/        # QA检测
│   │   ├── __init__.py
│   │   └── detector.py
│   │
│   ├── importance_scorer/  # 重要度评估
│   │   ├── __init__.py
│   │   └── scorer.py
│   │
│   └── note_generator/     # 笔记生成
│       ├── __init__.py
│       └── generator.py
│
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后的数据
│   └── chat_analysis.db     # SQLite数据库
│
└── notes/                   # 生成的笔记
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包: `requests`, `pyyaml`

### 安装依赖

```bash
pip install requests pyyaml
```

### 基本使用

```bash
# 默认：全量LLM汇总模式
python main.py chat.jsonl

# 使用标准分步流水线（非全量汇总）
python main.py chat.jsonl --pipeline

# 标准流水线下不使用LLM（仅规则）
python main.py chat.jsonl --pipeline --no-llm

# 查看文件信息
python main.py chat.jsonl --info

# 分步运行
python main.py chat.jsonl -s 1
```

## 📄 先生成报告再生成HTML

下面流程会先产出 `notes/` 下的 Markdown 笔记（report），再基于这些 MD 生成 HTML 报告。

### 一次性完成：提取日期 + 生成笔记 + 输出 HTML

```bash
python main.py "chat/怎么用AI辅助编程？.jsonl" --extract-date 2026-01-14 --html-report
```

说明：
- 提取后的 JSONL 默认命名为：`<原文件前缀>_<日期>.jsonl`（例如 `怎么用AI辅助编程？_2026-01-14.jsonl`）。
- HTML 输出默认文件名为：`<jsonl前缀>_<日期>_output.html`（例如 `怎么用AI辅助编程？_2026-01-14_output.html`）。
- 如需指定输出文件名，可加 `--extract-output ...` 或 `--html-output ...`。
-- `--no-llm` 仅对 `--pipeline` 生效，用于仅规则生成笔记。

### 配置修改

编辑 `config.yaml`：

```yaml
llm:
  provider: "zhipu"
  api_key: "your-api-key"
  model: "glm-4.7"

importance_scorer:
  threshold: 6.0  # 重要度阈值

note_generator:
  output_dir: "./notes"
```

## 📖 详细使用

查看 [USAGE.md](USAGE.md) 获取详细使用说明。

## 🔧 模块说明

### 1. 数据预处理 (A)

```python
from modules.preprocessor import DataCleaner

cleaner = DataCleaner()
messages = cleaner.clean_file("chat.jsonl")
```

功能：
- 解析JSONL格式
- 提取XML中的文本、链接、视频
- 识别问题消息
- 保存到SQLite

### 2. 话题分割 (B)

```python
from modules.topic_segment import TopicSegmenter

segmenter = TopicSegmenter()
topics = segmenter.segment(messages)
```

功能：
- 时间间隔 > 30分钟切分
- 连续发言 > 5条切分
- LLM细分割和标题生成

### 3. QA检测 (C)

```python
from modules.qa_detector import QADetector

detector = QADetector()
qa_pairs = detector.detect(messages)
```

功能：
- 问题关键词检测
- 回答窗口分析
- LLM验证和质量评分

### 4. 重要度评估 (D)

```python
from modules.importance_scorer import ImportanceScorer

scorer = ImportanceScorer()
result = scorer.score(topic, qa_pairs)
```

评分维度：
- 参与度 (30%)
- 内容深度 (25%)
- QA质量 (20%)
- 分享价值 (15%)
- 时效性 (10%)

### 5. 笔记生成 (E)

```python
from modules.note_generator import NoteGenerator

generator = NoteGenerator()
result = generator.generate(topic, score, qa_pairs)
```

输出格式：
- Markdown文件
- 保存到SQLite

## 📊 评分体系

重要度分数范围 0-10分：

| 分数段 | 含义 |
|--------|------|
| 8-10 | 高价值，必读 |
| 6-8 | 有价值，建议看 |
| 4-6 | 一般，可跳过 |
| 0-4 | 低价值，忽略 |

## 🔌 API密钥配置

系统使用智谱AI的GLM-4.7模型。在`config.yaml`中配置：

```yaml
llm:
  api_key: "your-api-key-here"
```

## 📝 输出示例

生成的Markdown笔记包含：

```markdown
# AI编程助手使用技巧

> 生成时间: 2026-01-13 | 重要度: 7.5分

## 核心问题/分享

讨论Claude、Cursor等AI编程工具的使用技巧

## 关键讨论点

- Claude Code的安装方法
- 使用技巧和最佳实践
- 与其他工具的对比

## 推荐资源

- [Claude官网](https://claude.com)
- 官方文档链接...

## 相关关键词

AI, 编程, Claude, Python, ...
```

## ⚠️ 注意事项

1. **API配额**: LLM调用会消耗API配额
2. **处理时间**: 大文件可能需要几分钟
3. **网络要求**: 需要访问智谱AI API
4. **存储空间**: 笔记会保存在`notes/`目录

## 📄 许可证

MIT License

## 👨‍💻 作者

AI Assistant
