#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain组件管理器
统一管理所有LangChain组件的创建和配置
支持持久化集成、性能监控和企业级功能
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config

# LangChain导入
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 自定义组件导入
from .siliconflow_embeddings import SiliconFlowEmbeddings
from .siliconflow_llm import SiliconFlowLLM
from .siliconflow_multimodal import SiliconFlowMultiModal
from .siliconflow_reranker import SiliconFlowReranker

# 使用条件导入以避免循环导入
try:
    from .text_splitter import ParentChildTextSplitter, HybridTextSplitter
    from .hybrid_retriever import HybridRetriever, ParentChildRetriever
    from .persistence import persistence_manager
except ImportError:
    # 如果导入失败，设置为None，稍后动态导入
    ParentChildTextSplitter = None
    HybridTextSplitter = None
    HybridRetriever = None
    ParentChildRetriever = None
    persistence_manager = None


class LangChainComponentManager:
    """LangChain组件管理器 - 企业级功能版本"""
    
    def __init__(self):
        # 核心组件
        self._embeddings: Optional[Embeddings] = None
        self._llm: Optional[BaseLLM] = None
        self._multimodal_llm: Optional[BaseLLM] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._vector_store: Optional[VectorStore] = None
        self._reranker: Optional[SiliconFlowReranker] = None
        self._parent_child_splitter = None
        self._hybrid_splitter = None
        self._hybrid_retriever = None
        self._parent_child_retriever = None
        
        # 性能监控
        self.component_stats = {
            'initialization_time': datetime.now().isoformat(),
            'components_created': {},
            'component_usage': {},
            'performance_metrics': {}
        }
        
        # 持久化管理器
        self._persistence_manager = None
        self._init_persistence()
        
        print("LangChain组件管理器初始化完成 (企业级功能版本)")
    
    def _init_persistence(self):
        """初始化持久化管理器"""
        try:
            if persistence_manager is None:
                # 动态导入
                from .persistence import persistence_manager as pm
                self._persistence_manager = pm
            else:
                self._persistence_manager = persistence_manager
                
            # 记录初始化
            if self._persistence_manager:
                self._log_operation("component_manager_init", {"status": "success"})
                
        except Exception as e:
            print(f"持久化管理器初始化警告: {e}")
            self._persistence_manager = None
    
    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """记录操作到持久化系统"""
        try:
            if self._persistence_manager:
                # 保存操作记录
                record = {
                    'operation': operation,
                    'timestamp': datetime.now().isoformat(),
                    'details': details
                }
                
                # 更新统计
                if operation not in self.component_stats['component_usage']:
                    self.component_stats['component_usage'][operation] = 0
                self.component_stats['component_usage'][operation] += 1
                
        except Exception as e:
            print(f"记录操作时出错: {e}")
    
    def _update_performance_metrics(self, component_name: str, operation_time: float):
        """更新性能指标"""
        if component_name not in self.component_stats['performance_metrics']:
            self.component_stats['performance_metrics'][component_name] = {
                'total_operations': 0,
                'total_time': 0,
                'avg_time': 0,
                'last_operation': None
            }
        
        metrics = self.component_stats['performance_metrics'][component_name]
        metrics['total_operations'] += 1
        metrics['total_time'] += operation_time
        metrics['avg_time'] = metrics['total_time'] / metrics['total_operations']
        metrics['last_operation'] = datetime.now().isoformat()
    
    @property
    def embeddings(self) -> Embeddings:
        """获取嵌入模型"""
        start_time = time.time()
        
        if self._embeddings is None:
            self._embeddings = SiliconFlowEmbeddings(
                api_key=Config.SILICONFLOW_API_KEY,
                base_url=Config.SILICONFLOW_BASE_URL,
                model=Config.EMBEDDING_MODEL
            )
            self.component_stats['components_created']['embeddings'] = datetime.now().isoformat()
            print(f"创建嵌入模型: {Config.EMBEDDING_MODEL}")
            
            # 记录创建操作
            self._log_operation("create_embeddings", {
                "model": Config.EMBEDDING_MODEL,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("embeddings", operation_time)
        
        return self._embeddings
    
    @property
    def llm(self) -> BaseLLM:
        """获取生成模型"""
        start_time = time.time()
        
        if self._llm is None:
            self._llm = SiliconFlowLLM(
                api_key=Config.SILICONFLOW_API_KEY,
                base_url=Config.SILICONFLOW_BASE_URL,
                model=Config.GENERATION_MODEL,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            self.component_stats['components_created']['llm'] = datetime.now().isoformat()
            print(f"创建生成模型: {Config.GENERATION_MODEL}")
            
            # 记录创建操作
            self._log_operation("create_llm", {
                "model": Config.GENERATION_MODEL,
                "max_tokens": Config.MAX_TOKENS,
                "temperature": Config.TEMPERATURE,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("llm", operation_time)
        
        return self._llm
    
    @property
    def multimodal_llm(self) -> BaseLLM:
        """获取多模态模型"""
        start_time = time.time()
        
        if self._multimodal_llm is None:
            self._multimodal_llm = SiliconFlowMultiModal(
                api_key=Config.SILICONFLOW_API_KEY,
                base_url=Config.SILICONFLOW_BASE_URL,
                model=Config.MULTIMODAL_MODEL,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            self.component_stats['components_created']['multimodal_llm'] = datetime.now().isoformat()
            print(f"创建多模态模型: {Config.MULTIMODAL_MODEL}")
            
            # 记录创建操作
            self._log_operation("create_multimodal_llm", {
                "model": Config.MULTIMODAL_MODEL,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("multimodal_llm", operation_time)
        
        return self._multimodal_llm
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """获取文本分割器"""
        start_time = time.time()
        
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.component_stats['components_created']['text_splitter'] = datetime.now().isoformat()
            print(f"创建文本分割器: chunk_size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}")
            
            # 记录创建操作
            self._log_operation("create_text_splitter", {
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("text_splitter", operation_time)
        
        return self._text_splitter
    
    @property
    def parent_child_splitter(self):
        """获取父子文本分割器"""
        start_time = time.time()
        
        if self._parent_child_splitter is None:
            # 动态导入
            from .text_splitter import ParentChildTextSplitter
            self._parent_child_splitter = ParentChildTextSplitter(
                parent_chunk_size=Config.PARENT_CHUNK_SIZE,
                parent_chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
                child_chunk_size=Config.CHILD_CHUNK_SIZE,
                child_chunk_overlap=Config.CHILD_CHUNK_OVERLAP
            )
            self.component_stats['components_created']['parent_child_splitter'] = datetime.now().isoformat()
            print("创建父子文本分割器")
            
            # 记录创建操作
            self._log_operation("create_parent_child_splitter", {
                "parent_chunk_size": Config.PARENT_CHUNK_SIZE,
                "child_chunk_size": Config.CHILD_CHUNK_SIZE,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("parent_child_splitter", operation_time)
        
        return self._parent_child_splitter
    
    @property
    def hybrid_splitter(self):
        """获取混合文本分割器"""
        start_time = time.time()
        
        if self._hybrid_splitter is None:
            # 动态导入
            from .text_splitter import HybridTextSplitter
            self._hybrid_splitter = HybridTextSplitter()
            self.component_stats['components_created']['hybrid_splitter'] = datetime.now().isoformat()
            print("创建混合文本分割器")
            
            # 记录创建操作
            self._log_operation("create_hybrid_splitter", {"status": "success"})
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("hybrid_splitter", operation_time)
        
        return self._hybrid_splitter
    
    @property
    def reranker(self) -> SiliconFlowReranker:
        """获取重排序器"""
        start_time = time.time()
        
        if self._reranker is None:
            self._reranker = SiliconFlowReranker(
                api_key=Config.SILICONFLOW_API_KEY,
                base_url=Config.SILICONFLOW_BASE_URL,
                model=Config.RERANKER_MODEL
            )
            self.component_stats['components_created']['reranker'] = datetime.now().isoformat()
            print(f"创建重排序器: {Config.RERANKER_MODEL}")
            
            # 记录创建操作
            self._log_operation("create_reranker", {
                "model": Config.RERANKER_MODEL,
                "status": "success"
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("reranker", operation_time)
        
        return self._reranker
    
    def get_vector_store(self, force_reload: bool = False) -> VectorStore:
        """获取向量存储"""
        start_time = time.time()
        
        if self._vector_store is None or force_reload:
            vector_store_path = Path(Config.VECTOR_STORE_PATH)
            
            try:
                if vector_store_path.exists() and not force_reload:
                    # 加载现有向量存储
                    self._vector_store = FAISS.load_local(
                        str(vector_store_path), 
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print("✅ 加载现有向量存储")
                    
                    # 记录加载操作
                    self._log_operation("load_vector_store", {
                        "path": str(vector_store_path),
                        "status": "success"
                    })
                else:
                    raise FileNotFoundError("需要创建新的向量存储")
                    
            except Exception as e:
                if "No such file or directory" in str(e) or isinstance(e, FileNotFoundError):
                    print("📋 首次运行，向量存储文件不存在，将创建新的向量存储")
                else:
                    print(f"⚠️ 向量存储加载异常: {type(e).__name__}, 将创建新的向量存储")
                
                # 创建新的向量存储
                self._vector_store = FAISS.from_texts(
                    texts=["初始化向量存储"],
                    embedding=self.embeddings,
                    metadatas=[{"source": "initialization"}]
                )
                print("✅ 创建新的向量存储完成")
                
                # 记录创建操作
                self._log_operation("create_vector_store", {
                    "reason": str(e),
                    "status": "success"
                })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("vector_store", operation_time)
        
        return self._vector_store
    
    def save_vector_store(self) -> bool:
        """保存向量存储"""
        start_time = time.time()
        success = False
        
        try:
            if self._vector_store is not None:
                vector_store_path = Path(Config.VECTOR_STORE_PATH)
                vector_store_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 检查是否为FAISS向量存储并保存
                if isinstance(self._vector_store, FAISS):
                    self._vector_store.save_local(str(vector_store_path))
                    print(f"✅ 向量存储已保存到: {vector_store_path}")
                    success = True
                    
                    # 记录保存操作
                    self._log_operation("save_vector_store", {
                        "path": str(vector_store_path),
                        "status": "success"
                    })
                else:
                    print("⚠️ 当前向量存储类型不支持本地保存功能")
                    
                    # 记录保存失败
                    self._log_operation("save_vector_store", {
                        "status": "failed",
                        "reason": "unsupported_vector_store_type"
                    })
            else:
                print("⚠️ 没有向量存储可保存")
                
        except Exception as e:
            print(f"❌ 保存向量存储时出错: {e}")
            self._log_operation("save_vector_store", {
                "status": "error",
                "error": str(e)
            })
        
        # 更新性能指标
        operation_time = time.time() - start_time
        self._update_performance_metrics("save_vector_store", operation_time)
        
        return success
    
    def create_hybrid_retriever(self, documents: List[Document]):
        """创建混合检索器"""
        start_time = time.time()
        
        try:
            if self._hybrid_retriever is None:
                # 动态导入
                from .hybrid_retriever import HybridRetriever
                vector_store = self.get_vector_store()
                self._hybrid_retriever = HybridRetriever(
                    vector_store=vector_store,
                    documents=documents,
                    vector_weight=Config.VECTOR_SEARCH_WEIGHT,
                    keyword_weight=Config.KEYWORD_SEARCH_WEIGHT,
                    use_bm25=Config.USE_BM25
                )
                self.component_stats['components_created']['hybrid_retriever'] = datetime.now().isoformat()
                print("创建混合检索器")
                
                # 记录创建操作
                self._log_operation("create_hybrid_retriever", {
                    "documents_count": len(documents),
                    "vector_weight": Config.VECTOR_SEARCH_WEIGHT,
                    "keyword_weight": Config.KEYWORD_SEARCH_WEIGHT,
                    "use_bm25": Config.USE_BM25,
                    "status": "success"
                })
            
            # 更新性能指标
            operation_time = time.time() - start_time
            self._update_performance_metrics("hybrid_retriever", operation_time)
            
            return self._hybrid_retriever
            
        except Exception as e:
            print(f"❌ 创建混合检索器时出错: {e}")
            self._log_operation("create_hybrid_retriever", {
                "status": "error",
                "error": str(e)
            })
            raise
    
    def create_parent_child_retriever(
        self, 
        child_vector_store: VectorStore,
        parent_docs: List[Document],
        child_docs: List[Document],
        parent_to_children: Dict[str, List[str]]
    ):
        """创建父子检索器"""
        start_time = time.time()
        
        try:
            # 动态导入
            from .hybrid_retriever import ParentChildRetriever
            self._parent_child_retriever = ParentChildRetriever(
                child_vector_store=child_vector_store,
                parent_docs=parent_docs,
                child_docs=child_docs,
                parent_to_children=parent_to_children
            )
            self.component_stats['components_created']['parent_child_retriever'] = datetime.now().isoformat()
            print("创建父子检索器")
            
            # 记录创建操作
            self._log_operation("create_parent_child_retriever", {
                "parent_docs_count": len(parent_docs),
                "child_docs_count": len(child_docs),
                "parent_child_mappings": len(parent_to_children),
                "status": "success"
            })
            
            # 更新性能指标
            operation_time = time.time() - start_time
            self._update_performance_metrics("parent_child_retriever", operation_time)
            
            return self._parent_child_retriever
            
        except Exception as e:
            print(f"❌ 创建父子检索器时出错: {e}")
            self._log_operation("create_parent_child_retriever", {
                "status": "error",
                "error": str(e)
            })
            raise
    
    def create_rag_chain(self):
        """创建RAG链"""
        start_time = time.time()
        
        try:
            # 创建提示模板
            prompt = PromptTemplate.from_template(Config.RAG_PROMPT_TEMPLATE)
            
            # 创建检索器
            retriever = self.get_vector_store().as_retriever(
                search_kwargs={"k": Config.TOP_K}
            )
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # 创建RAG链
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 记录创建操作
            self._log_operation("create_rag_chain", {
                "top_k": Config.TOP_K,
                "status": "success"
            })
            
            # 更新性能指标
            operation_time = time.time() - start_time
            self._update_performance_metrics("rag_chain", operation_time)
            
            return rag_chain
            
        except Exception as e:
            print(f"❌ 创建RAG链时出错: {e}")
            self._log_operation("create_rag_chain", {
                "status": "error",
                "error": str(e)
            })
            raise
    
    def create_multimodal_chain(self):
        """创建多模态处理链"""
        start_time = time.time()
        
        try:
            prompt = PromptTemplate.from_template(Config.IMAGE_SUMMARY_PROMPT)
            
            multimodal_chain = (
                prompt
                | self.multimodal_llm
                | StrOutputParser()
            )
            
            # 记录创建操作
            self._log_operation("create_multimodal_chain", {"status": "success"})
            
            # 更新性能指标
            operation_time = time.time() - start_time
            self._update_performance_metrics("multimodal_chain", operation_time)
            
            return multimodal_chain
            
        except Exception as e:
            print(f"❌ 创建多模态链时出错: {e}")
            self._log_operation("create_multimodal_chain", {
                "status": "error",
                "error": str(e)
            })
            raise
    
    def get_component_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "embedding_model": Config.EMBEDDING_MODEL,
            "generation_model": Config.GENERATION_MODEL,
            "multimodal_model": Config.MULTIMODAL_MODEL,
            "reranker_model": Config.RERANKER_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "top_k": Config.TOP_K,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "stats": self.component_stats
        }
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """获取组件统计信息"""
        return {
            "components_created": self.component_stats['components_created'],
            "component_usage": self.component_stats['component_usage'],
            "performance_metrics": self.component_stats['performance_metrics'],
            "initialization_time": self.component_stats['initialization_time'],
            "current_time": datetime.now().isoformat()
        }
    
    def reset_component(self, component_name: str) -> bool:
        """重置指定组件"""
        try:
            component_map = {
                'embeddings': '_embeddings',
                'llm': '_llm',
                'multimodal_llm': '_multimodal_llm',
                'text_splitter': '_text_splitter',
                'vector_store': '_vector_store',
                'reranker': '_reranker',
                'parent_child_splitter': '_parent_child_splitter',
                'hybrid_splitter': '_hybrid_splitter',
                'hybrid_retriever': '_hybrid_retriever',
                'parent_child_retriever': '_parent_child_retriever'
            }
            
            if component_name in component_map:
                setattr(self, component_map[component_name], None)
                print(f"✅ 组件 {component_name} 已重置")
                
                # 记录重置操作
                self._log_operation("reset_component", {
                    "component": component_name,
                    "status": "success"
                })
                return True
            else:
                print(f"⚠️ 未知组件: {component_name}")
                return False
                
        except Exception as e:
            print(f"❌ 重置组件时出错: {e}")
            self._log_operation("reset_component", {
                "component": component_name,
                "status": "error",
                "error": str(e)
            })
            return False
    
    def cleanup_resources(self):
        """清理资源"""
        try:
            # 保存当前向量存储
            if self._vector_store:
                self.save_vector_store()
            
            # 保存统计信息到持久化系统
            if self._persistence_manager:
                try:
                    # 保存系统状态
                    self._persistence_manager.save_system_state({
                        "component_manager_stats": self.component_stats,
                        "cleanup_time": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"保存状态时出错: {e}")
            
            print("✅ 资源清理完成")
            
            # 记录清理操作
            self._log_operation("cleanup_resources", {"status": "success"})
            
        except Exception as e:
            print(f"❌ 清理资源时出错: {e}")
            self._log_operation("cleanup_resources", {
                "status": "error",
                "error": str(e)
            })

# 全局组件管理器实例
component_manager = LangChainComponentManager()

# 为了向后兼容性，提供别名
ComponentManager = LangChainComponentManager

# 模块清理函数
def cleanup_component_manager():
    """模块清理函数"""
    global component_manager
    if component_manager:
        component_manager.cleanup_resources()

# 注册清理函数
import atexit
atexit.register(cleanup_component_manager)
