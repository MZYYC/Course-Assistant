import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 硅基流动API配置
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    
    # 文档路径配置
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data/documents")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    
    # 模型配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # 多模态模型配置
    MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "THUDM/GLM-4.1V-9B-Thinking")
    
    # 生成模型配置
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "Qwen/Qwen3-8B")
    
    # 重排序模型配置
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    
    # 文本分割配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # 父子切块配置
    PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
    PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
    CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
    CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
    
    # 检索配置
    TOP_K = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # 混合检索配置
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
    VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.7"))
    KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.3"))
    USE_BM25 = os.getenv("USE_BM25", "true").lower() == "true"
    
    # 文本分割策略配置
    TEXT_SPLIT_STRATEGY = os.getenv("TEXT_SPLIT_STRATEGY", "parent_child")  # standard, parent_child, semantic, hybrid
    
    # 生成配置
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # LangChain配置
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
    
    # 多模态配置
    IMAGE_SUMMARY_PROMPT = """
    请分析这张图片的内容，并提供详细的文本描述。描述应该包括：
    1. 图片中的主要对象和场景
    2. 任何可见的文本内容
    3. 图片的整体主题和目的
    4. 重要的细节和特征
    
    请用中文回答，描述要详细且准确。
    """
    
    # RAG提示模板
    RAG_PROMPT_TEMPLATE = """基于以下检索到的相关文档，回答用户的问题。

相关文档：
{context}

用户问题：{question}

请根据检索到的文档内容，提供准确、详细的回答。如果文档中没有足够的信息来回答问题，请诚实地说明。

回答："""
