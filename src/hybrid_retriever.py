#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索器
支持关键词+向量混合检索，包含持久化集成和性能监控
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import time
from collections import Counter
from datetime import datetime
import math

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.vectorstores import FAISS
import jieba
from config import Config


class KeywordRetriever:
    """关键词检索器，基于TF-IDF和BM25算法"""
    
    def __init__(self, documents: List[Document]):
        """
        初始化关键词检索器
        
        Args:
            documents: 文档列表
        """
        self.documents = documents
        self.doc_frequencies = {}  # 词频统计
        self.idf_scores = {}       # IDF分数
        self.doc_vectors = []      # 文档向量
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        # 移除标点符号和特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 使用jieba分词
        tokens = list(jieba.cut(text.lower()))
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1 and token.strip() not in stop_words]
        return tokens
    
    def _build_index(self):
        """构建倒排索引和TF-IDF"""
        print("正在构建关键词索引...")
        
        # 统计词频
        doc_word_counts = []
        all_words = set()
        
        for doc in self.documents:
            words = self._tokenize(doc.page_content)
            word_count = Counter(words)
            doc_word_counts.append(word_count)
            all_words.update(words)
        
        # 计算文档频率
        for word in all_words:
            self.doc_frequencies[word] = sum(1 for doc_words in doc_word_counts if word in doc_words)
        
        # 计算IDF
        total_docs = len(self.documents)
        for word in all_words:
            self.idf_scores[word] = math.log(total_docs / (self.doc_frequencies[word] + 1))
        
        # 构建文档向量
        for word_count in doc_word_counts:
            doc_vector = {}
            total_words = sum(word_count.values())
            for word, count in word_count.items():
                tf = count / total_words
                doc_vector[word] = tf * self.idf_scores[word]
            self.doc_vectors.append(doc_vector)
        
        print(f"关键词索引构建完成，词汇表大小: {len(all_words)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        基于关键词搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            文档和相关性分数的列表
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        # 计算查询向量
        query_word_count = Counter(query_words)
        total_query_words = len(query_words)
        query_vector = {}
        for word, count in query_word_count.items():
            if word in self.idf_scores:
                tf = count / total_query_words
                query_vector[word] = tf * self.idf_scores[word]
        
        # 计算相似度
        doc_scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            # 余弦相似度
            dot_product = sum(query_vector.get(word, 0) * doc_vector.get(word, 0) for word in query_vector)
            query_norm = math.sqrt(sum(score ** 2 for score in query_vector.values()))
            doc_norm = math.sqrt(sum(score ** 2 for score in doc_vector.values()))
            
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
                doc_scores.append((self.documents[i], similarity))
        
        # 排序并返回top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表
            k1: BM25参数k1
            b: BM25参数b
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_frequencies = {}
        self.doc_word_counts = []
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = list(jieba.cut(text.lower()))
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        tokens = [token.strip() for token in tokens if len(token.strip()) > 1 and token.strip() not in stop_words]
        return tokens
    
    def _build_index(self):
        """构建BM25索引"""
        print("正在构建BM25索引...")
        
        if not self.documents:
            print("⚠️ 没有文档，跳过BM25索引构建")
            self.avg_doc_length = 1.0  # 避免除零错误
            return
        
        # 统计文档长度和词频
        all_words = set()
        for doc in self.documents:
            words = self._tokenize(doc.page_content)
            word_count = Counter(words)
            self.doc_word_counts.append(word_count)
            self.doc_lengths.append(len(words))
            all_words.update(words)
        
        # 避免除零错误
        if len(self.doc_lengths) == 0:
            self.avg_doc_length = 1.0
        else:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
            if self.avg_doc_length == 0:
                self.avg_doc_length = 1.0  # 避免除零
        
        # 计算文档频率
        for word in all_words:
            self.doc_frequencies[word] = sum(1 for doc_words in self.doc_word_counts if word in doc_words)
        
        print(f"BM25索引构建完成，平均文档长度: {self.avg_doc_length:.2f}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        使用BM25算法搜索文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            文档和BM25分数的列表
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        doc_scores = []
        N = len(self.documents)
        
        for i, doc in enumerate(self.documents):
            score = 0
            doc_length = self.doc_lengths[i]
            doc_word_count = self.doc_word_counts[i]
            
            for word in query_words:
                if word not in self.doc_frequencies:
                    continue
                
                # 计算IDF
                df = self.doc_frequencies[word]
                idf = math.log((N - df + 0.5) / (df + 0.5))
                
                # 计算TF
                tf = doc_word_count.get(word, 0)
                
                # BM25分数
                numerator = tf * (self.k1 + 1)
                # 避免除零错误
                avg_length = max(self.avg_doc_length, 1.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_length))
                if denominator == 0:
                    denominator = 1.0  # 避免除零
                score += idf * (numerator / denominator)
            
            doc_scores.append((doc, score))
        
        # 排序并返回top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


class HybridRetriever(BaseRetriever):
    """混合检索器，结合向量检索和关键词检索，包含持久化集成和性能监控"""
    
    # 声明Pydantic字段
    _vector_store: VectorStore
    _documents: List[Document]
    _vector_weight: float
    _keyword_weight: float
    _enable_persistence: bool
    _algorithm: str
    _keyword_retriever: Any
    _stats: Dict[str, Any]
    _persistence: Any
    
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    def __init__(
        self,
        vector_store: VectorStore,
        documents: List[Document],
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_bm25: bool = True,
        enable_persistence: bool = True
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            documents: 文档列表
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            use_bm25: 是否使用BM25（否则使用TF-IDF）
            enable_persistence: 是否启用持久化
        """
        super().__init__()
        # 使用私有属性避免LangChain的字段限制
        self._vector_store = vector_store
        self._documents = documents
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._enable_persistence = enable_persistence
        
        # 初始化关键词检索器
        try:
            if use_bm25:
                if not documents:
                    print("⚠️ 文档列表为空，无法创建BM25检索器")
                    # 创建一个空的检索器作为占位符
                    self._keyword_retriever = None
                    self._algorithm = "BM25 (Empty)"
                else:
                    self._keyword_retriever = BM25Retriever(documents)
                    self._algorithm = "BM25"
            else:
                if not documents:
                    print("⚠️ 文档列表为空，无法创建关键词检索器")
                    self._keyword_retriever = None
                    self._algorithm = "TF-IDF (Empty)"
                else:
                    self._keyword_retriever = KeywordRetriever(documents)
                    self._algorithm = "TF-IDF"
        except Exception as e:
            print(f"❌ 创建混合检索器时出错: {e}")
            self._keyword_retriever = None
            self._algorithm = f"Error: {str(e)}"
        
        # 统计信息
        self._stats = {
            'total_queries': 0,
            'keyword_queries': 0,
            'vector_queries': 0,
            'hybrid_queries': 0,
            'average_retrieval_time': 0.0,
            'total_retrieval_time': 0.0,
            'algorithm_used': self._algorithm,
            'vector_weight': vector_weight,
            'keyword_weight': keyword_weight,
            'query_distribution': Counter(),
            'performance_metrics': {
                'fastest_query': float('inf'),
                'slowest_query': 0.0,
                'total_documents_retrieved': 0,
                'average_documents_per_query': 0.0
            },
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # 如果启用持久化，尝试导入持久化系统
        if self._enable_persistence:
            try:
                from .persistence import PersistenceManager
                self._persistence = PersistenceManager()
                print("已启用持久化系统集成")
            except ImportError:
                print("警告: 无法导入持久化系统，禁用持久化功能")
                self._enable_persistence = False
                self._persistence = None
        else:
            self._persistence = None
        
        print(f"初始化混合检索器: 向量权重={vector_weight}, 关键词权重={keyword_weight}, 算法={self._algorithm}")
        
    @property
    def stats(self):
        """获取统计信息的属性访问器"""
        return self._stats
        
    def _update_stats(self, query: str, retrieval_time: float, num_results: int, query_type: str = "hybrid"):
        """更新统计信息"""
        self._stats['total_queries'] += 1
        self._stats['total_retrieval_time'] += retrieval_time
        self._stats['average_retrieval_time'] = self._stats['total_retrieval_time'] / self._stats['total_queries']
        
        if query_type == "keyword":
            self._stats['keyword_queries'] += 1
        elif query_type == "vector":
            self._stats['vector_queries'] += 1
        else:
            self._stats['hybrid_queries'] += 1
        
        # 更新查询分布
        query_length = len(query.split())
        self._stats['query_distribution'][f"{query_length}_words"] += 1
        
        # 更新性能指标
        self._stats['performance_metrics']['fastest_query'] = min(
            self._stats['performance_metrics']['fastest_query'], retrieval_time
        )
        self._stats['performance_metrics']['slowest_query'] = max(
            self._stats['performance_metrics']['slowest_query'], retrieval_time
        )
        self._stats['performance_metrics']['total_documents_retrieved'] += num_results
        self._stats['performance_metrics']['average_documents_per_query'] = (
            self._stats['performance_metrics']['total_documents_retrieved'] / self._stats['total_queries']
        )
        
        self._stats['last_updated'] = datetime.now().isoformat()
        
        # 如果启用持久化，记录检索历史
        if self._enable_persistence and self._persistence:
            try:
                # 使用save_query_history方法记录查询历史
                self._persistence.save_query_history(
                    query=query,
                    answer=f"检索到{num_results}个相关文档",
                    sources=[{"type": "hybrid_retrieval", "count": num_results}],
                    strategy=f"{self._algorithm}_hybrid",
                    response_time=retrieval_time,
                    success=True
                )
            except Exception as e:
                print(f"警告: 记录检索历史失败: {e}")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """获取相关文档"""
        return self.search(query, top_k=Config.TOP_K)
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        混合搜索，包含性能监控
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            相关文档列表
        """
        start_time = time.time()
        
        try:
            # 向量检索
            vector_results = self._vector_store.similarity_search_with_score(query, k=top_k * 2)
            
            # 关键词检索 - 检查检索器是否可用
            keyword_results = []
            if self._keyword_retriever is not None:
                try:
                    keyword_results = self._keyword_retriever.search(query, top_k * 2)
                except Exception as e:
                    print(f"⚠️ 关键词检索失败: {e}")
                    keyword_results = []
            else:
                print("⚠️ 关键词检索器未初始化，仅使用向量检索")
            
            # 归一化分数
            vector_scores = self._normalize_scores([score for _, score in vector_results])
            keyword_scores = self._normalize_scores([score for _, score in keyword_results]) if keyword_results else []
            
            # 合并分数
            doc_scores = {}
            
            # 处理向量检索结果
            for i, (doc, _) in enumerate(vector_results):
                doc_id = self._get_doc_id(doc)
                doc_scores[doc_id] = {
                    'doc': doc,
                    'vector_score': vector_scores[i],
                    'keyword_score': 0
                }
        
            # 处理关键词检索结果
            for i, (doc, _) in enumerate(keyword_results):
                doc_id = self._get_doc_id(doc)
                if doc_id in doc_scores:
                    doc_scores[doc_id]['keyword_score'] = keyword_scores[i]
                else:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_score': 0,
                        'keyword_score': keyword_scores[i]
                    }
            
            # 计算综合分数
            final_results = []
            for doc_info in doc_scores.values():
                combined_score = (
                    self._vector_weight * doc_info['vector_score'] + 
                    self._keyword_weight * doc_info['keyword_score']
                )
                final_results.append((doc_info['doc'], combined_score))
            
            # 排序并返回top_k
            final_results.sort(key=lambda x: x[1], reverse=True)
            result_docs = [doc for doc, _ in final_results[:top_k]]
            
            # 计算检索时间并更新统计信息
            retrieval_time = time.time() - start_time
            self._update_stats(query, retrieval_time, len(result_docs), "hybrid")
            
            print(f"混合检索完成: 查询='{query[:50]}...', 结果数={len(result_docs)}, 耗时={retrieval_time:.3f}秒")
            
            return result_docs
            
        except Exception as e:
            retrieval_time = time.time() - start_time
            print(f"混合检索失败: {e}")
            
            # 记录失败的查询
            if self._enable_persistence and self._persistence:
                try:
                    self._persistence.save_query_history(
                        query=query,
                        strategy=f"{self._algorithm}_hybrid",
                        response_time=retrieval_time,
                        success=False,
                        error_message=str(e)
                    )
                except Exception as persist_error:
                    print(f"记录失败查询时出错: {persist_error}")
            
            # 返回空结果
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数到[0,1]区间"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _get_doc_id(self, doc: Document) -> str:
        """获取文档唯一标识"""
        # 使用文档内容的哈希作为ID
        return str(hash(doc.page_content))
    
    def update_weights(self, vector_weight: float, keyword_weight: float):
        """更新权重"""
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._stats['vector_weight'] = vector_weight
        self._stats['keyword_weight'] = keyword_weight
        self._stats['last_updated'] = datetime.now().isoformat()
        print(f"更新混合检索权重: 向量={vector_weight}, 关键词={keyword_weight}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            **self._stats,
            'algorithm': self._algorithm,
            'total_documents': len(self._documents),
            'vector_store_type': type(self._vector_store).__name__
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            'total_queries': 0,
            'keyword_queries': 0,
            'vector_queries': 0,
            'hybrid_queries': 0,
            'average_retrieval_time': 0.0,
            'total_retrieval_time': 0.0,
            'algorithm_used': self._algorithm,
            'vector_weight': self._vector_weight,
            'keyword_weight': self._keyword_weight,
            'query_distribution': Counter(),
            'performance_metrics': {
                'fastest_query': float('inf'),
                'slowest_query': 0.0,
                'total_documents_retrieved': 0,
                'average_documents_per_query': 0.0
            },
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        print("已重置检索统计信息")
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n=== 混合检索器统计信息 ===")
        print(f"算法: {stats['algorithm']}")
        print(f"文档总数: {stats['total_documents']}")
        print(f"向量存储类型: {stats['vector_store_type']}")
        print(f"当前权重: 向量={stats['vector_weight']:.2f}, 关键词={stats['keyword_weight']:.2f}")
        print(f"\n查询统计:")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  混合查询: {stats['hybrid_queries']}")
        print(f"  关键词查询: {stats['keyword_queries']}")
        print(f"  向量查询: {stats['vector_queries']}")
        print(f"\n性能统计:")
        print(f"  平均检索时间: {stats['average_retrieval_time']:.3f}秒")
        print(f"  最快查询: {stats['performance_metrics']['fastest_query']:.3f}秒")
        print(f"  最慢查询: {stats['performance_metrics']['slowest_query']:.3f}秒")
        print(f"  平均每次检索文档数: {stats['performance_metrics']['average_documents_per_query']:.1f}")
        print(f"\n创建时间: {stats['created_at']}")
        print(f"最后更新: {stats['last_updated']}")
        print("=" * 40)


class ParentChildRetriever(BaseRetriever):
    """父子检索器，用于父子切块策略，包含持久化集成"""
    
    # 声明Pydantic字段
    _child_vector_store: VectorStore
    _parent_docs: Dict[str, Document]
    _child_docs: Dict[str, Document]
    _parent_to_children: Dict[str, List[str]]
    _enable_persistence: bool
    _stats: Dict[str, Any]
    _persistence: Any
    _parent_map: Dict[str, Document]
    
    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
    
    def __init__(
        self,
        child_vector_store: VectorStore,
        parent_docs: List[Document],
        child_docs: List[Document],
        parent_to_children: Dict[str, List[str]],
        enable_persistence: bool = True
    ):
        """
        初始化父子检索器
        
        Args:
            child_vector_store: 子块向量存储
            parent_docs: 父块列表
            child_docs: 子块列表
            parent_to_children: 父块到子块的映射
            enable_persistence: 是否启用持久化
        """
        super().__init__()
        self._child_vector_store = child_vector_store
        self._parent_docs = {doc.metadata.get('id', str(hash(doc.page_content))): doc for doc in parent_docs}
        self._child_docs = {doc.metadata.get('id', str(hash(doc.page_content))): doc for doc in child_docs}
        self._parent_to_children = parent_to_children
        self._enable_persistence = enable_persistence
        
        # 统计信息
        self._stats = {
            'total_queries': 0,
            'parent_docs_count': len(parent_docs),
            'child_docs_count': len(child_docs),
            'parent_child_mappings': len(parent_to_children),
            'average_retrieval_time': 0.0,
            'total_retrieval_time': 0.0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'average_results_per_query': 0.0,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # 如果启用持久化，尝试导入持久化系统
        if self._enable_persistence:
            try:
                from .persistence import PersistenceManager
                self._persistence = PersistenceManager()
                print("父子检索器已启用持久化系统集成")
            except ImportError:
                print("警告: 无法导入持久化系统，禁用持久化功能")
                self._enable_persistence = False
                self._persistence = None
        else:
            self._persistence = None
        
        # 创建父块映射（使用chunk_id或生成ID）
        self._parent_map = {}
        for doc in parent_docs:
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id") or str(hash(doc.page_content))
            self._parent_map[chunk_id] = doc
        
        print(f"初始化父子检索器: 父块数={len(parent_docs)}, 子块数={len(child_docs)}, 映射数={len(parent_to_children)}")
    
    def _update_stats(self, query: str, retrieval_time: float, num_results: int, success: bool = True):
        """更新统计信息"""
        self._stats['total_queries'] += 1
        self._stats['total_retrieval_time'] += retrieval_time
        self._stats['average_retrieval_time'] = self._stats['total_retrieval_time'] / self._stats['total_queries']
        
        if success:
            self._stats['successful_retrievals'] += 1
        else:
            self._stats['failed_retrievals'] += 1
        
        if self._stats['total_queries'] > 0:
            total_results = self._stats.get('total_results_retrieved', 0) + num_results
            self._stats['total_results_retrieved'] = total_results
            self._stats['average_results_per_query'] = total_results / self._stats['total_queries']
        
        self._stats['last_updated'] = datetime.now().isoformat()
        
        # 如果启用持久化，记录检索历史
        if self._enable_persistence and self._persistence:
            try:
                self._persistence.save_query_history(
                    query=query,
                    answer=f"父子检索到{num_results}个相关文档",
                    sources=[{"type": "parent_child_retrieval", "count": num_results}],
                    strategy="parent_child",
                    response_time=retrieval_time,
                    success=success
                )
            except Exception as e:
                print(f"警告: 记录父子检索历史失败: {e}")
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """获取相关文档"""
        return self.search(query, top_k=Config.TOP_K)
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        父子检索，包含性能监控
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            相关的父块文档列表
        """
        start_time = time.time()
        
        try:
            # 首先在子块中检索
            child_results = self._child_vector_store.similarity_search_with_score(query, k=top_k * 2)
            
            # 获取对应的父块
            parent_scores = {}
            for child_doc, score in child_results:
                parent_id = child_doc.metadata.get("parent_id") or child_doc.metadata.get("chunk_id")
                if parent_id and parent_id in self._parent_map:
                    if parent_id not in parent_scores:
                        parent_scores[parent_id] = []
                    parent_scores[parent_id].append(score)
            
            # 计算父块的综合分数（取子块分数的最大值）
            parent_results = []
            for parent_id, scores in parent_scores.items():
                # 使用最大分数作为父块分数
                max_score = max(scores)
                parent_doc = self._parent_map[parent_id]
                parent_results.append((parent_doc, max_score))
            
            # 排序并返回top_k
            parent_results.sort(key=lambda x: x[1], reverse=True)
            result_docs = [doc for doc, _ in parent_results[:top_k]]
            
            # 更新统计信息
            retrieval_time = time.time() - start_time
            self._update_stats(query, retrieval_time, len(result_docs), True)
            
            print(f"父子检索完成: 查询='{query[:50]}...', 结果数={len(result_docs)}, 耗时={retrieval_time:.3f}秒")
            
            return result_docs
            
        except Exception as e:
            retrieval_time = time.time() - start_time
            print(f"父子检索失败: {e}")
            
            # 更新失败统计
            self._update_stats(query, retrieval_time, 0, False)
            
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取父子检索统计信息"""
        return self._stats.copy()
    
    def print_stats(self):
        """打印统计信息"""
        print("\n=== 父子检索器统计信息 ===")
        print(f"父块数: {self._stats['parent_docs_count']}")
        print(f"子块数: {self._stats['child_docs_count']}")
        print(f"父子映射数: {self._stats['parent_child_mappings']}")
        print(f"\n检索统计:")
        print(f"  总查询数: {self._stats['total_queries']}")
        print(f"  成功检索: {self._stats['successful_retrievals']}")
        print(f"  失败检索: {self._stats['failed_retrievals']}")
        print(f"  平均检索时间: {self._stats['average_retrieval_time']:.3f}秒")
        print(f"  平均每次检索结果数: {self._stats['average_results_per_query']:.1f}")
        print(f"\n创建时间: {self._stats['created_at']}")
        print(f"最后更新: {self._stats['last_updated']}")
        print("=" * 40)
