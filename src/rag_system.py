#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangChain的RAG系统 - 重构版本
修复父子检索器和文档检索问题，优化系统架构
"""

import sys
import time
import threading
import queue
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import logging
from contextlib import contextmanager

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .components import component_manager
from .document_loader import LangChainDocumentLoader
from .multimodal_processor import LangChainMultimodalProcessor
from .persistence import persistence_manager
from config import Config


class LangChainRAGSystem:
    """重构的RAG系统 - 修复版本
    
    主要改进：
    1. 修复父子检索器创建时机问题
    2. 优化检索策略选择逻辑
    3. 简化系统初始化流程
    4. 增强错误处理和恢复机制
    5. 改进性能监控
    """
    
    def __init__(self):
        """初始化RAG系统"""
        # 核心组件
        self.document_loader = LangChainDocumentLoader(Config.DOCUMENTS_PATH)
        self.multimodal_processor = LangChainMultimodalProcessor(persistence_manager)
        
        # 系统状态
        self.is_initialized = False
        self._initialization_lock = threading.Lock()
        self._current_strategy = Config.TEXT_SPLIT_STRATEGY
        
        # 数据存储
        self._all_documents = []
        self._split_result = {}
        self._rag_chain = None
        
        # 检索器状态跟踪
        self._retriever_status = {
            'parent_child': False,
            'hybrid': False,
            'vector': False
        }
        
        # 性能监控
        self._stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0,
            'initialization_time': None,
            'last_query_time': None
        }
        
        # 查询缓存
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag_v2")
        
        # 设置日志
        self._setup_logging()
        
        print("RAG系统V2初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        log_path = Path("logs") / "rag_system_v2.log"
        log_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def _performance_monitor(self, operation: str):
        """性能监控上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(f"{operation} 耗时: {duration:.3f}秒")
    
    def initialize_system(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """初始化系统
        
        Args:
            force_rebuild: 是否强制重建索引
            
        Returns:
            初始化结果
        """
        with self._initialization_lock:
            if self.is_initialized and not force_rebuild:
                return {
                    "status": "success",
                    "message": "系统已初始化",
                    "retriever_status": self._retriever_status
                }
            
            start_time = time.time()
            
            try:
                self.logger.info("开始初始化RAG系统V2...")
                
                # 1. 检查是否需要重建
                need_rebuild = force_rebuild or not self._check_existing_index()
                
                if need_rebuild:
                    result = self._full_rebuild()
                else:
                    result = self._load_existing_system()
                
                if result["status"] != "success":
                    return result
                
                # 2. 预热系统
                self._warmup_system()
                
                # 3. 验证检索器状态
                self._validate_retrievers()
                
                self.is_initialized = True
                initialization_time = time.time() - start_time
                self._stats['initialization_time'] = initialization_time
                
                return {
                    "status": "success",
                    "message": "RAG系统V2初始化成功",
                    "initialization_time": initialization_time,
                    "num_documents": len(self._all_documents),
                    "split_strategy": self._current_strategy,
                    "split_stats": self._get_split_stats(),
                    "retriever_status": self._retriever_status,
                    "stats": self._get_system_stats()
                }
                
            except Exception as e:
                self.logger.error(f"RAG系统初始化失败: {e}")
                return {
                    "status": "error",
                    "message": f"初始化失败: {str(e)}",
                    "initialization_time": time.time() - start_time
                }
    
    def _full_rebuild(self) -> Dict[str, Any]:
        """完整重建系统"""
        try:
            self.logger.info("开始完整重建...")
            
            # 1. 加载文档
            with self._performance_monitor("文档加载"):
                documents = self.document_loader.load_documents()
                if not documents:
                    return {
                        "status": "error",
                        "message": "没有找到任何文档"
                    }
                self._all_documents = documents
            
            # 2. 多模态处理
            with self._performance_monitor("多模态处理"):
                processed_documents = self.multimodal_processor.process_documents(documents)
            
            # 3. 文本分割
            with self._performance_monitor("文本分割"):
                self._split_result = self._split_documents_with_strategy(processed_documents)
            
            # 4. 构建向量存储
            with self._performance_monitor("向量存储构建"):
                self._build_vector_store_with_strategy()
            
            # 5. 创建检索器
            with self._performance_monitor("检索器创建"):
                self._create_retrievers()
            
            return {"status": "success", "message": "完整重建成功"}
            
        except Exception as e:
            self.logger.error(f"完整重建失败: {e}")
            return {"status": "error", "message": f"重建失败: {str(e)}"}
    
    def _load_existing_system(self) -> Dict[str, Any]:
        """加载现有系统"""
        try:
            self.logger.info("加载现有系统...")
            
            # 对于父子策略，总是重新构建以确保一致性
            if self._current_strategy == "parent_child":
                self.logger.info("父子策略需要重新构建以确保父子块ID一致性")
                return self._full_rebuild()
            
            # 1. 加载向量存储
            component_manager.get_vector_store()
            
            # 2. 重新加载文档以获取分割结果（关键修复）
            documents = self.document_loader.load_documents()
            if not documents:
                return {"status": "error", "message": "无法加载文档"}
            
            self._all_documents = documents
            
            # 3. 重新进行多模态处理和分割（获得内存中的分割结果）
            processed_documents = self.multimodal_processor.process_documents(documents)
            self._split_result = self._split_documents_with_strategy(processed_documents)
            
            # 4. 重新创建检索器（基于内存中的分割结果）
            self._create_retrievers()
            
            return {"status": "success", "message": "现有系统加载成功"}
            
        except Exception as e:
            self.logger.warning(f"加载现有系统失败: {e}，将进行完整重建")
            return self._full_rebuild()
    
    def _split_documents_with_strategy(self, documents: List[Document]) -> Dict[str, Any]:
        """根据策略分割文档"""
        if self._current_strategy == "parent_child":
            splitter = component_manager.parent_child_splitter
            result = splitter.split_documents(documents)
            self.logger.info(f"父子切块完成: {len(result['parent_docs'])} 个父块, {len(result['child_docs'])} 个子块")
            return result
        
        elif self._current_strategy == "hybrid":
            splitter = component_manager.hybrid_splitter
            result = splitter.split_documents_with_strategy(documents, "hybrid")
            self.logger.info("混合切块完成")
            return result
        
        else:
            text_splitter = component_manager.text_splitter
            chunks = text_splitter.split_documents(documents)
            self.logger.info(f"标准切块完成: {len(chunks)} 个文本块")
            return {"chunks": chunks, "strategy": "standard"}
    
    def _build_vector_store_with_strategy(self) -> None:
        """根据策略构建向量存储"""
        strategy = self._current_strategy
        embeddings = component_manager.embeddings
        
        from langchain_community.vectorstores import FAISS
        
        if strategy == "parent_child":
            # 父子策略：为子块建立向量索引
            child_docs = self._split_result["child_docs"]
            if not child_docs:
                raise ValueError("没有子块可以构建向量存储")
            
            # 为子块创建向量存储
            child_vector_store = FAISS.from_documents(child_docs, embeddings)
            
            # 保存子块向量存储
            child_vector_path = Path(Config.VECTOR_STORE_PATH) / "child_chunks"
            child_vector_path.parent.mkdir(parents=True, exist_ok=True)
            child_vector_store.save_local(str(child_vector_path))
            
            # 为父块创建单独的向量存储
            parent_docs = self._split_result["parent_docs"]
            if parent_docs:
                parent_vector_store = FAISS.from_documents(parent_docs, embeddings)
                parent_vector_path = Path(Config.VECTOR_STORE_PATH) / "parent_chunks"
                parent_vector_store.save_local(str(parent_vector_path))
            
            # 更新组件管理器中的向量存储为子块存储
            component_manager._vector_store = child_vector_store
            
        elif strategy == "hybrid":
            # 混合策略
            child_docs = self._split_result.get("child_docs", [])
            standard_chunks = self._split_result.get("standard_chunks", [])
            
            all_chunks = []
            if child_docs:
                all_chunks.extend(child_docs)
            if standard_chunks:
                all_chunks.extend(standard_chunks)
            
            if not all_chunks:
                raise ValueError("没有文档块可以构建向量存储")
            
            vector_store = FAISS.from_documents(all_chunks, embeddings)
            vector_store_path = Path(Config.VECTOR_STORE_PATH)
            vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(vector_store_path))
            
            component_manager._vector_store = vector_store
        
        else:
            # 标准策略
            chunks = self._split_result["chunks"]
            if not chunks:
                raise ValueError("没有文档块可以构建向量存储")
            
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store_path = Path(Config.VECTOR_STORE_PATH)
            vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(vector_store_path))
            
            component_manager._vector_store = vector_store
        
        self.logger.info(f"向量存储已保存到: {Config.VECTOR_STORE_PATH}")
    
    def _create_retrievers(self) -> None:
        """创建检索器"""
        try:
            # 重置检索器状态
            self._retriever_status = {
                'parent_child': False,
                'hybrid': False,
                'vector': False
            }
            
            # 创建父子检索器（如果是父子策略）
            if self._current_strategy == "parent_child":
                child_vector_store = component_manager.get_vector_store()
                parent_docs = self._split_result.get("parent_docs", [])
                child_docs = self._split_result.get("child_docs", [])
                parent_to_children = self._split_result.get("parent_to_children", {})
                
                if parent_docs and child_docs and parent_to_children:
                    component_manager.create_parent_child_retriever(
                        child_vector_store, parent_docs, child_docs, parent_to_children
                    )
                    self._retriever_status['parent_child'] = True
                    self.logger.info("✅ 父子检索器创建成功")
                else:
                    self.logger.warning("⚠️ 父子检索器创建失败：缺少必要数据")
            
            # 创建混合检索器（如果启用混合搜索）
            if Config.USE_HYBRID_SEARCH and self._all_documents:
                component_manager.create_hybrid_retriever(self._all_documents)
                self._retriever_status['hybrid'] = True
                self.logger.info("✅ 混合检索器创建成功")
            
            # 向量检索器总是可用
            if component_manager.get_vector_store():
                self._retriever_status['vector'] = True
                self.logger.info("✅ 向量检索器可用")
            
        except Exception as e:
            self.logger.error(f"创建检索器失败: {e}")
            raise
    
    def _validate_retrievers(self) -> None:
        """验证检索器状态"""
        self.logger.info("验证检索器状态...")
        
        # 检查父子检索器
        if (self._current_strategy == "parent_child" and 
            hasattr(component_manager, '_parent_child_retriever')):
            retriever = component_manager._parent_child_retriever
            if retriever:
                stats = retriever._stats
                self.logger.info(f"父子检索器: 父块数={stats.get('parent_docs_count', 0)}, "
                               f"子块数={stats.get('child_docs_count', 0)}, "
                               f"映射数={stats.get('parent_child_mappings', 0)}")
                
                if stats.get('parent_docs_count', 0) > 0 and stats.get('child_docs_count', 0) > 0:
                    self._retriever_status['parent_child'] = True
                else:
                    self.logger.warning("⚠️ 父子检索器数据为空")
        
        # 检查混合检索器
        if (Config.USE_HYBRID_SEARCH and 
            hasattr(component_manager, '_hybrid_retriever')):
            if component_manager._hybrid_retriever:
                self._retriever_status['hybrid'] = True
        
        # 检查向量检索器
        try:
            vector_store = component_manager.get_vector_store()
            if vector_store:
                # 测试向量检索
                test_docs = vector_store.similarity_search("测试", k=1)
                if test_docs:
                    self._retriever_status['vector'] = True
                    self.logger.info("✅ 向量检索器验证成功")
                else:
                    self.logger.warning("⚠️ 向量检索器返回空结果")
        except Exception as e:
            self.logger.error(f"向量检索器验证失败: {e}")
    
    def _warmup_system(self) -> None:
        """系统预热"""
        try:
            self.logger.info("系统预热中...")
            
            # 预热嵌入模型
            embeddings = component_manager.embeddings
            embeddings.embed_query("系统预热测试")
            
            # 预热生成模型
            llm = component_manager.llm
            llm.invoke("系统预热")
            
            self.logger.info("✅ 系统预热完成")
            
        except Exception as e:
            self.logger.warning(f"系统预热失败: {e}")
    
    def _check_existing_index(self) -> bool:
        """检查是否存在现有索引"""
        vector_store_path = Path(Config.VECTOR_STORE_PATH)
        return (vector_store_path.exists() and 
                (vector_store_path / "index.faiss").exists() and
                (vector_store_path / "index.pkl").exists())
    
    def query(
        self, 
        question: str, 
        use_reranking: bool = True,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """查询系统
        
        Args:
            question: 查询问题
            use_reranking: 是否使用重排序
            top_k: 返回文档数量
            use_cache: 是否使用缓存
            timeout: 超时时间
            
        Returns:
            查询结果
        """
        start_time = time.time()
        query_id = f"query_{int(start_time)}_{hash(question) % 10000}"
        
        # 参数验证
        if not question or not question.strip():
            return self._create_error_result("查询问题不能为空", start_time)
        
        if not self.is_initialized:
            return self._create_error_result("系统未初始化", start_time)
        
        try:
            self.logger.info(f"处理查询 [{query_id}]: {question[:50]}...")
            
            top_k = top_k or Config.TOP_K
            
            # 检查缓存
            cache_key = None
            if use_cache:
                cache_key = self._get_cache_key(question, use_reranking, top_k)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.logger.info(f"查询 [{query_id}] 命中缓存")
                    cached_result['query_id'] = query_id
                    cached_result['cached'] = True
                    return cached_result
            
            # 执行查询
            result = self._execute_query(question, use_reranking, top_k, timeout)
            result['query_id'] = query_id
            result['cached'] = False
            
            # 缓存结果
            if use_cache and cache_key and result.get("status") == "success":
                self._cache_result(cache_key, result)
            
            # 更新统计
            response_time = time.time() - start_time
            self._update_stats(response_time, result.get("status") == "success")
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询 [{query_id}] 失败: {e}")
            response_time = time.time() - start_time
            self._update_stats(response_time, False)
            return self._create_error_result(f"查询失败: {str(e)}", start_time)
    
    def _execute_query(
        self, 
        question: str, 
        use_reranking: bool, 
        top_k: int, 
        timeout: float
    ) -> Dict[str, Any]:
        """执行查询"""
        retrieval_start = time.time()
        
        # 1. 检索文档
        docs = self._retrieve_documents(question, top_k, use_reranking)
        retrieval_time = time.time() - retrieval_start
        
        if not docs:
            return {
                "status": "warning",
                "message": "未找到相关文档",
                "answer": "抱歉，我没有找到与您的问题相关的信息。",
                "sources": [],
                "retrieval_time": retrieval_time,
                "generation_time": 0.0,
                "total_docs_found": 0,
                "search_strategy": self._get_current_search_strategy()
            }
        
        # 2. 生成答案
        generation_start = time.time()
        try:
            # 创建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 使用超时控制
            answer = self._generate_answer_with_timeout(question, context, timeout)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"生成答案失败: {str(e)}",
                "answer": "",
                "sources": self._format_sources(docs),
                "retrieval_time": retrieval_time,
                "generation_time": time.time() - generation_start
            }
        
        generation_time = time.time() - generation_start
        
        # 3. 构建结果
        return {
            "status": "success",
            "message": "查询成功",
            "answer": answer,
            "sources": self._format_sources(docs),
            "context": context,
            "search_strategy": self._get_current_search_strategy(),
            "split_strategy": self._current_strategy,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_docs_found": len(docs),
            "reranking_used": use_reranking,
            "retriever_status": self._retriever_status
        }
    
    def _retrieve_documents(
        self, 
        question: str, 
        top_k: int, 
        use_reranking: bool
    ) -> List[Document]:
        """检索文档"""
        try:
            # 选择检索策略
            if (self._current_strategy == "parent_child" and 
                self._retriever_status.get('parent_child', False)):
                return self._retrieve_parent_child(question, top_k, use_reranking)
            
            elif (Config.USE_HYBRID_SEARCH and 
                  self._retriever_status.get('hybrid', False)):
                return self._retrieve_hybrid(question, top_k, use_reranking)
            
            elif self._retriever_status.get('vector', False):
                return self._retrieve_vector(question, top_k, use_reranking)
            
            else:
                self.logger.error("没有可用的检索器")
                return []
                
        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            return []
    
    def _retrieve_parent_child(
        self, 
        question: str, 
        top_k: int, 
        use_reranking: bool
    ) -> List[Document]:
        """父子检索"""
        if (hasattr(component_manager, '_parent_child_retriever') and 
            component_manager._parent_child_retriever):
            
            try:
                docs = component_manager._parent_child_retriever.search(question, top_k)
                self.logger.info(f"父子检索返回 {len(docs)} 个文档")
                return docs
            except Exception as e:
                self.logger.error(f"父子检索失败: {e}")
                # 回退到向量检索
                return self._retrieve_vector(question, top_k, use_reranking)
        else:
            self.logger.warning("父子检索器不可用，回退到向量检索")
            return self._retrieve_vector(question, top_k, use_reranking)
    
    def _retrieve_hybrid(
        self, 
        question: str, 
        top_k: int, 
        use_reranking: bool
    ) -> List[Document]:
        """混合检索"""
        if (hasattr(component_manager, '_hybrid_retriever') and 
            component_manager._hybrid_retriever):
            
            try:
                docs = component_manager._hybrid_retriever.search(question, top_k*2)
                
                # 重排序
                if use_reranking and docs:
                    reranker = component_manager.reranker
                    docs = reranker.rerank(question, docs, top_k)
                
                return docs[:top_k]
            except Exception as e:
                self.logger.error(f"混合检索失败: {e}")
                return self._retrieve_vector(question, top_k, use_reranking)
        else:
            return self._retrieve_vector(question, top_k, use_reranking)
    
    def _retrieve_vector(
        self, 
        question: str, 
        top_k: int, 
        use_reranking: bool
    ) -> List[Document]:
        """向量检索"""
        try:
            vector_store = component_manager.get_vector_store()
            docs = vector_store.similarity_search(question, k=top_k*2)
            
            # 重排序
            if use_reranking and docs:
                reranker = component_manager.reranker
                docs = reranker.rerank(question, docs, top_k)
            
            return docs[:top_k]
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []
    
    def _generate_answer_with_timeout(
        self, 
        question: str, 
        context: str, 
        timeout: float
    ) -> str:
        """带超时的答案生成"""
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def generate():
            try:
                # 创建提示
                prompt_template = Config.RAG_PROMPT_TEMPLATE
                prompt = prompt_template.format(context=context, question=question)
                
                # 生成答案
                llm = component_manager.llm
                answer = llm.invoke(prompt)
                result_queue.put(answer)
            except Exception as e:
                exception_queue.put(e)
        
        # 启动生成线程
        gen_thread = threading.Thread(target=generate)
        gen_thread.daemon = True
        gen_thread.start()
        gen_thread.join(timeout)
        
        # 检查结果
        if gen_thread.is_alive():
            raise TimeoutError(f"答案生成超时（{timeout}秒）")
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        if result_queue.empty():
            raise Exception("生成器未返回结果")
        
        return result_queue.get()
    
    def _format_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """格式化源文档"""
        sources = []
        for i, doc in enumerate(docs):
            source_info = {
                "index": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "file_name": doc.metadata.get("file_name", "unknown"),
                "chunk_type": doc.metadata.get("chunk_type", "standard"),
                "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
                "content_preview": (
                    doc.page_content[:200] + "..." 
                    if len(doc.page_content) > 200 
                    else doc.page_content
                ),
                "content_length": len(doc.page_content)
            }
            sources.append(source_info)
        return sources
    
    def _get_current_search_strategy(self) -> str:
        """获取当前搜索策略"""
        if (self._current_strategy == "parent_child" and 
            self._retriever_status.get('parent_child', False)):
            return "parent_child"
        elif (Config.USE_HYBRID_SEARCH and 
              self._retriever_status.get('hybrid', False)):
            return "hybrid"
        else:
            return "vector"
    
    def _create_error_result(self, message: str, start_time: float) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "status": "error",
            "message": message,
            "answer": "",
            "sources": [],
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_cache_key(self, question: str, use_reranking: bool, top_k: int) -> str:
        """生成缓存键"""
        cache_input = f"{question}_{self._current_strategy}_{use_reranking}_{top_k}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        with self._cache_lock:
            cached = self._query_cache.get(cache_key)
            if cached:
                # 检查缓存是否过期（1小时）
                if time.time() - cached['timestamp'] < 3600:
                    return cached['result']
                else:
                    del self._query_cache[cache_key]
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """缓存结果"""
        with self._cache_lock:
            # 限制缓存大小
            if len(self._query_cache) > 100:
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k]['timestamp'])
                del self._query_cache[oldest_key]
            
            self._query_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def _update_stats(self, response_time: float, success: bool):
        """更新统计信息"""
        self._stats['total_queries'] += 1
        self._stats['total_response_time'] += response_time
        self._stats['average_response_time'] = (
            self._stats['total_response_time'] / self._stats['total_queries']
        )
        
        if success:
            self._stats['successful_queries'] += 1
        else:
            self._stats['failed_queries'] += 1
        
        self._stats['last_query_time'] = datetime.now().isoformat()
    
    def _get_split_stats(self) -> Dict[str, Any]:
        """获取分割统计信息"""
        stats: Dict[str, Any] = {"strategy": self._current_strategy}
        
        if self._current_strategy == "parent_child":
            stats["parent_chunks"] = len(self._split_result.get("parent_docs", []))
            stats["child_chunks"] = len(self._split_result.get("child_docs", []))
            stats["parent_to_children_mapping"] = len(self._split_result.get("parent_to_children", {}))
        elif self._current_strategy == "hybrid":
            stats["standard_chunks"] = len(self._split_result.get("standard_chunks", []))
            stats["parent_chunks"] = len(self._split_result.get("parent_docs", []))
            stats["child_chunks"] = len(self._split_result.get("child_docs", []))
            stats["semantic_chunks"] = len(self._split_result.get("semantic_chunks", []))
        else:
            stats["total_chunks"] = len(self._split_result.get("chunks", []))
        
        return stats
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            vector_store = component_manager.get_vector_store()
            num_vectors = 0
            if hasattr(vector_store, 'index'):
                index = getattr(vector_store, 'index')
                if hasattr(index, 'ntotal'):
                    num_vectors = getattr(index, 'ntotal')
            
            return {
                "num_vectors": num_vectors,
                "embedding_model": Config.EMBEDDING_MODEL,
                "generation_model": Config.GENERATION_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "top_k": Config.TOP_K,
                "retriever_status": self._retriever_status,
                "performance": self._stats
            }
        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self.is_initialized,
            "strategy": self._current_strategy,
            "retriever_status": self._retriever_status,
            "stats": self._stats,
            "documents_loaded": len(self._all_documents),
            "cache_size": len(self._query_cache)
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """清除缓存"""
        with self._cache_lock:
            cache_size = len(self._query_cache)
            self._query_cache.clear()
        
        return {
            "status": "success",
            "message": f"已清除 {cache_size} 个缓存项"
        }
    
    def switch_strategy(self, new_strategy: str, force_rebuild: bool = True) -> Dict[str, Any]:
        """切换策略"""
        if new_strategy not in ["standard", "parent_child", "hybrid"]:
            return {
                "status": "error",
                "message": f"不支持的策略: {new_strategy}"
            }
        
        old_strategy = self._current_strategy
        self._current_strategy = new_strategy
        
        # 重置RAG链和检索器状态
        self._rag_chain = None
        self._retriever_status = {'parent_child': False, 'hybrid': False, 'vector': False}
        
        if force_rebuild:
            result = self.initialize_system(force_rebuild=True)
            if result["status"] == "success":
                return {
                    "status": "success",
                    "message": f"成功切换策略从 {old_strategy} 到 {new_strategy}",
                    "old_strategy": old_strategy,
                    "new_strategy": new_strategy
                }
            else:
                # 回滚
                self._current_strategy = old_strategy
                return {
                    "status": "error",
                    "message": f"策略切换失败，已回滚到 {old_strategy}"
                }
        else:
            return {
                "status": "success",
                "message": f"策略已更新到 {new_strategy}，下次初始化时生效"
            }
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# 创建全局实例，保持向后兼容
rag_system_v2 = LangChainRAGSystem()
