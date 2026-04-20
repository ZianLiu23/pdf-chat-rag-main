# 📄 PDF Chat （基于 RAG 的问答系统）

把任意 PDF 变成可交互对话。上传文档后，即可用自然语言提问，基于 RAG 与大模型获取答案。

- 项目唯一标识：`pdf-chat-rag-qa-rag-system-zianliu-2026`
- GitHub 项目名：`PDF Chat （基于 RAG 的问答系统）`

## 🌟 在线演示

- 演示地址：[https://pdf-chat-rag-fx5nczbrwczzpou6qyczmj.streamlit.app/](https://pdf-chat-rag-fx5nczbrwczzpou6qyczmj.streamlit.app/)
- 注意：需要一个免费的 DeepSeek API Key，可在 [https://platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) 创建。

## ✨ 功能特性

- 📤 支持上传任意 PDF（论文、书籍、报告、手册等）
- 💬 自然语言问答，交互体验接近聊天
- ⚡ 基于 DeepSeek 的高速推理
- 🎯 RAG 检索增强，答案更贴合文档内容
- 🎨 使用 Streamlit 构建的简洁界面
- 🔒 注重隐私，文档处理在本地进行，不做数据存储

## 🛠️ 技术栈

| 技术 | 作用 | 选择理由 |
| --- | --- | --- |
| LangChain | RAG 框架 | LLM 文档检索应用的主流方案 |
| DeepSeek | 大模型推理 | OpenAI 兼容接口，接入简单 |
| FAISS | 向量检索 | 高效相似度搜索库 |
| Streamlit | Web UI 框架 | Python 生态下快速搭建交互页面 |
| HuggingFace Embeddings | 文本向量化 | 开源句向量模型，效果稳定 |

## 🚀 快速开始

### 1) 环境要求

- Python 3.8 或更高版本
- DeepSeek API Key

### 2) 安装与运行

```bash
cd pdf-chat-rag-main

python -m venv venv

# 在 Windows 上激活
venv\Scripts\activate

# 在 macOS / Linux 上激活
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

浏览器访问：`http://localhost:8501`

## 📖 工作原理

### RAG 流程

```text
PDF 上传 -> 文本提取 -> 文本切分 -> 向量化 -> 向量库
                                            |
用户问题 -> 向量化 -> 相似度检索 -> 上下文召回 -> LLM 生成答案
```

### 分步骤说明

1. 文档处理
- 使用 `PyPDF2` 提取 PDF 文本
- 按固定大小切分（默认 `1000` 字符，重叠 `200`）
- 使用 HuggingFace Embeddings 生成向量

2. 向量存储
- 将向量写入 `FAISS`
- 在问答时快速检索相关片段

3. 问题处理
- 将用户问题向量化
- 召回最相关的 Top-K 文本片段（默认 `k=3`）
- 将上下文与问题一起发送给 LLM

4. 答案生成
- 由 DeepSeek 模型生成基于上下文的回答
- 回答严格依赖文档内容，减少幻觉

## 💡 示例使用场景

- 学术研究：这篇论文的核心结论是什么？
- 法律文档：第 5 条关于责任如何定义？
- 技术手册：错误码 E404 应该如何排查？
- 商业报告：总结一下 Q3 财务表现。

## 🔧 可调参数

### 文本切分参数

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 可按文档长度调整
    chunk_overlap=200  # 平衡上下文连续性与片段独立性
)
```

### 模型参数

```python
llm = ChatOpenAI(
    model="deepseek-chat",  # 可替换为其他 DeepSeek 模型
    base_url="https://api.deepseek.com",
    api_key="YOUR_DEEPSEEK_API_KEY",
    temperature=0  # 0 更偏事实，调高可增加创造性
)
```

### 检索参数

```python
docs = vector_store.similarity_search(
    question,
    k=3  # 每次召回的文档片段数量
)
```

## 📊 性能参考

- 文档处理：典型 PDF 约 10-30 秒
- 单次问答：约 1-3 秒
- 内存占用：随文档体量增长
- 准确性：事实类问题表现较好，依赖文本质量

## 🎯 路线图

- [ ] 支持更多文件格式（DOCX、TXT 等）
- [ ] 会话历史持久化
- [ ] 多文档联合问答
- [ ] 对话导出
- [ ] 更高级的过滤与检索策略
- [ ] 提供 API 接口便于系统集成

## 📝 许可证

本项目基于 MIT License 开源，详见 `LICENSE` 文件。

## 🙏 致谢

- LangChain：提供强大的 RAG 框架
- DeepSeek：提供 OpenAI 兼容的 LLM 推理能力
- Streamlit：提供便捷的 UI 构建体验
- Facebook AI：提供 FAISS 向量检索库
- HuggingFace：提供开源嵌入模型

## 📧 联系方式

- 姓名：刘子安
- 手机号：13607717765
- 邮箱：867012264@qq.com

