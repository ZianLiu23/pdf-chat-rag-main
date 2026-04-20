import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from PyPDF2 import PdfReader

# 页面配置
st.set_page_config(
    page_title="PDF 对话助手", 
    page_icon="📄",
    layout="wide"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">📄 与你的 PDF 对话</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">上传 PDF 后即可提问，由 RAG 与 LangChain 驱动</p>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")
    env_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    input_deepseek_api_key = st.text_input(
        "DeepSeek API 密钥",
        type="password",
        help="在 platform.deepseek.com/api_keys 获取你的 API 密钥"
    )
    deepseek_api_key = input_deepseek_api_key.strip() or env_deepseek_api_key
    
    if deepseek_api_key:
        st.success("✅ API 密钥已配置")
        if env_deepseek_api_key and not input_deepseek_api_key.strip():
            st.caption("正在使用环境变量中的 API 密钥：DEEPSEEK_API_KEY")
    
    st.markdown("---")
    st.markdown("### 📚 使用方法：")
    st.markdown("""
    1. 输入 DeepSeek API 密钥
    2. 上传 PDF 文件
    3. 等待处理完成
    4. 开始提问
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 快速链接：")
    st.markdown("[获取 DeepSeek API 密钥](https://platform.deepseek.com/api_keys)")
    
    st.markdown("---")
    st.markdown("### 🛠️ 技术栈：")
    st.markdown("""
    • **LangChain** - RAG 框架
    • **DeepSeek** - 大模型推理
    • **FAISS** - 向量检索
    • **Streamlit** - 界面框架
    """)

    st.markdown("---")
    st.markdown("### 👤 个人信息")
    st.markdown("姓名：刘子安")
    st.markdown("手机号：13607717765")
    st.markdown("邮箱：867012264@qq.com")
    
    st.markdown("---")
    st.markdown("*© 2025 - 开源且免费使用*")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# 主内容区
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "📎 选择一个 PDF 文件", 
        type=['pdf'],
        help="上传任意 PDF 文档后即可开始问答"
    )

# 处理 PDF
if uploaded_file and deepseek_api_key:
    file_name = uploaded_file.name
    
    # 检查是否需要重新处理
    if st.session_state.processed_file != file_name:
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.processed_file = file_name
    
    if st.session_state.vector_store is None:
        with st.spinner("🔄 正在处理 PDF，请稍候..."):
            try:
                # 从 PDF 中提取文本
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    st.error("❌ 无法从 PDF 提取文本，请确认文件不是扫描件或纯图片 PDF。")
                    st.stop()
                
                # 将文本切分为分块
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # 创建向量嵌入与向量库
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.success(f"✅ 已成功处理 **{file_name}**！（共生成 {len(chunks)} 个文本分块）")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ 处理 PDF 时出错：{str(e)}")
                st.stop()
    
    # 聊天界面
    st.markdown("---")
    st.markdown("### 💬 与 PDF 对话")
    
    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入
    if question := st.chat_input("请输入你想问的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                try:
                    # 检索相关文档
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # 配置 LLM（直接调用 OpenAI 兼容接口，避免链式调用中的编码兼容问题）
                    client = OpenAI(
                        api_key=deepseek_api_key,
                        base_url="https://api.deepseek.com/v1"
                    )

                    user_prompt = (
                        "Context from the PDF:\n"
                        f"{context}\n\n"
                        f"Question: {question}\n\n"
                        "Please provide a detailed and accurate answer based only on the context above. "
                        "If the answer cannot be found in the context, say so."
                    )

                    # 获取回答
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        temperature=0,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that answers questions based on the provided context."
                            },
                            {"role": "user", "content": user_prompt}
                        ],
                    )
                    answer = (response.choices[0].message.content or "").strip()
                    if not answer:
                        answer = "未生成有效回答，请重试。"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except UnicodeEncodeError:
                    error_msg = "❌ 出错了：请求编码失败。请重试，或改用更短的问题。"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"❌ 出错了：{str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif not deepseek_api_key:
    st.warning("⚠️ 请先在侧边栏填写 DeepSeek API 密钥后再继续")
    st.info("👉 你可以在 [platform.deepseek.com](https://platform.deepseek.com/api_keys) 获取 API 密钥")
    st.caption("提示：也可以通过环境变量 `DEEPSEEK_API_KEY` 配置")
else:
    st.info("👆 上传一个 PDF 文件后即可开始对话")
    
    # 展示示例
    with st.expander("📖 查看示例用法"):
        st.markdown("""
        **你可以这样用：**
        - 📊 分析研究论文
        - 📝 总结长文档
        - 🔍 快速定位关键信息
        - ❓ 针对内容提问
        - 💡 获取复杂概念解释
        
        **示例问题：**
        - "这份文档的主题是什么？"
        - "请总结关键要点"
        - "文档里是如何描述[某个主题]的？"
        - "请列出主要结论"
        """)
