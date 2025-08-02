#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的文本分割器
支持父子切块策略、持久化集成和性能监控
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import time
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


class ParentChildTextSplitter:
    """父子切块文本分割器"""
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        persistence_manager=None
    ):
        """
        初始化父子文本分割器
        
        Args:
            parent_chunk_size: 父块大小
            parent_chunk_overlap: 父块重叠
            child_chunk_size: 子块大小
            child_chunk_overlap: 子块重叠
            separators: 分割符列表
            persistence_manager: 持久化管理器
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.persistence_manager = persistence_manager
        
        # 分割统计
        self.split_stats = {
            'total_documents': 0,
            'total_parent_chunks': 0,
            'total_child_chunks': 0,
            'processing_time': 0,
            'last_split_time': None
        }
        
        # 默认分割符
        if separators is None:
            separators = [
                "\n\n\n",  # 段落分割
                "\n\n",    # 双换行
                "\n",      # 单换行
                "。",      # 中文句号
                "！",      # 中文感叹号
                "？",      # 中文问号
                ".",       # 英文句号
                "!",       # 英文感叹号
                "?",       # 英文问号
                ";",       # 分号
                ",",       # 逗号
                " ",       # 空格
                ""         # 字符级别
            ]
        
        # 创建父块分割器
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        # 创建子块分割器
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        print(f"父子文本分割器初始化完成: 父块={parent_chunk_size}, 子块={child_chunk_size}")
    
    def split_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        分割文档为父子块
        
        Args:
            documents: 输入文档列表
            
        Returns:
            包含parent_docs和child_docs的字典
        """
        start_time = time.time()
        parent_docs = []
        child_docs = []
        parent_to_children = {}  # 父块ID到子块ID列表的映射
        
        print(f"🔧 开始父子切块，输入文档: {len(documents)}")
        
        for doc_idx, doc in enumerate(documents):
            print(f"   处理文档 [{doc_idx+1}/{len(documents)}]: {doc.metadata.get('source', 'unknown')}")
            
            # 分割为父块
            parent_chunks = self.parent_splitter.split_documents([doc])
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # 为父块生成唯一ID
                parent_id = str(uuid.uuid4())
                parent_chunk.metadata.update({
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                    "parent_index": parent_idx,
                    "original_source": doc.metadata.get("source", "unknown"),
                    "split_timestamp": datetime.now().isoformat(),
                    "chunk_size": len(parent_chunk.page_content)
                })
                parent_docs.append(parent_chunk)
                
                # 将父块进一步分割为子块
                child_chunks = self.child_splitter.split_documents([parent_chunk])
                child_ids = []
                
                for i, child_chunk in enumerate(child_chunks):
                    # 为子块生成唯一ID
                    child_id = f"{parent_id}_child_{i}"
                    child_chunk.metadata.update({
                        "chunk_id": child_id,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                        "child_index": i,
                        "parent_index": parent_idx,
                        "original_source": doc.metadata.get("source", "unknown"),
                        "split_timestamp": datetime.now().isoformat(),
                        "chunk_size": len(child_chunk.page_content)
                    })
                    child_docs.append(child_chunk)
                    child_ids.append(child_id)
                
                parent_to_children[parent_id] = child_ids
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self.split_stats.update({
            'total_documents': len(documents),
            'total_parent_chunks': len(parent_docs),
            'total_child_chunks': len(child_docs),
            'processing_time': processing_time,
            'last_split_time': datetime.now().isoformat()
        })
        
        print(f"✅ 父子切块完成: {len(parent_docs)} 个父块, {len(child_docs)} 个子块, 耗时 {processing_time:.2f}s")
        
        # 记录到持久化系统
        if self.persistence_manager:
            try:
                self.persistence_manager.save_document_history(
                    file_path="text_splitting",
                    file_name="parent_child_split",
                    file_size=sum(len(doc.page_content) for doc in documents),
                    processing_time=processing_time,
                    success=True,
                    chunk_count=len(parent_docs) + len(child_docs),
                    strategy="parent_child"
                )
            except Exception as e:
                print(f"⚠️ 持久化记录失败: {e}")
        
        return {
            "parent_docs": parent_docs,
            "child_docs": child_docs,
            "parent_to_children": parent_to_children,
            "stats": self.split_stats
        }
    
    def get_parent_for_child(self, child_doc: Document) -> Optional[str]:
        """获取子块对应的父块ID"""
        return child_doc.metadata.get("parent_id")
    
    def get_context_for_child(
        self, 
        child_doc: Document, 
        parent_docs: List[Document]
    ) -> Optional[Document]:
        """为子块获取完整的父块上下文"""
        parent_id = self.get_parent_for_child(child_doc)
        if not parent_id:
            return None
        
        for parent_doc in parent_docs:
            if parent_doc.metadata.get("chunk_id") == parent_id:
                return parent_doc
        
        return None
    
    def get_split_statistics(self) -> Dict[str, Any]:
        """获取分割统计信息"""
        return self.split_stats.copy()
    
    def analyze_chunk_distribution(self, chunks: List[Document]) -> Dict[str, Any]:
        """分析块分布统计"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes),
            'chunk_types': {
                'parent': len([c for c in chunks if c.metadata.get('chunk_type') == 'parent']),
                'child': len([c for c in chunks if c.metadata.get('chunk_type') == 'child'])
            }
        }


class HybridTextSplitter:
    """混合文本分割器，结合多种分割策略"""
    
    def __init__(self, persistence_manager=None):
        """初始化混合文本分割器"""
        self.persistence_manager = persistence_manager
        
        # 标准分割器
        self.standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", ",", " ", ""]
        )
        
        # 父子分割器
        self.parent_child_splitter = ParentChildTextSplitter(
            parent_chunk_size=Config.PARENT_CHUNK_SIZE,
            parent_chunk_overlap=Config.PARENT_CHUNK_OVERLAP,
            child_chunk_size=Config.CHILD_CHUNK_SIZE,
            child_chunk_overlap=Config.CHILD_CHUNK_OVERLAP,
            persistence_manager=persistence_manager
        )
        
        # 语义分割器（基于句子）
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP // 2,
            length_function=len,
            separators=["\n\n\n", "\n\n", "。", "！", "？", ".", "!", "?", "\n", " "]
        )
        
        # 分割统计
        self.hybrid_stats = {
            'strategies_used': [],
            'total_processing_time': 0,
            'last_strategy': None,
            'performance_comparison': {}
        }
        
        print(f"混合文本分割器初始化完成，支持策略: standard, parent_child, semantic, hybrid")
    
    def split_documents_with_strategy(
        self, 
        documents: List[Document], 
        strategy: str = "parent_child"
    ) -> Dict[str, Any]:
        """
        根据策略分割文档
        
        Args:
            documents: 输入文档列表
            strategy: 分割策略 ("standard", "parent_child", "semantic", "hybrid")
            
        Returns:
            分割结果字典
        """
        start_time = time.time()
        print(f"🔧 使用 '{strategy}' 策略分割 {len(documents)} 个文档")
        
        result = {}
        
        if strategy == "standard":
            chunks = self.standard_splitter.split_documents(documents)
            # 添加分割元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "split_strategy": "standard",
                    "split_timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk.page_content)
                })
            result = {"chunks": chunks, "strategy": "standard"}
        
        elif strategy == "parent_child":
            result = self.parent_child_splitter.split_documents(documents)
            result["strategy"] = "parent_child"
        
        elif strategy == "semantic":
            chunks = self.semantic_splitter.split_documents(documents)
            # 添加分割元数据
            for chunk in chunks:
                chunk.metadata.update({
                    "split_strategy": "semantic",
                    "split_timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk.page_content)
                })
            result = {"chunks": chunks, "strategy": "semantic"}
        
        elif strategy == "hybrid":
            # 混合策略：同时使用多种分割方法
            print("   🔄 执行混合分割策略...")
            
            # 标准分割
            standard_start = time.time()
            standard_chunks = self.standard_splitter.split_documents(documents)
            standard_time = time.time() - standard_start
            
            # 父子分割
            parent_child_start = time.time()
            parent_child_result = self.parent_child_splitter.split_documents(documents)
            parent_child_time = time.time() - parent_child_start
            
            # 语义分割
            semantic_start = time.time()
            semantic_chunks = self.semantic_splitter.split_documents(documents)
            semantic_time = time.time() - semantic_start
            
            # 为不同策略的块添加标识
            for chunk in standard_chunks:
                chunk.metadata.update({
                    "split_strategy": "standard",
                    "split_timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk.page_content)
                })
            
            for chunk in parent_child_result["parent_docs"]:
                chunk.metadata["split_strategy"] = "parent"
            
            for chunk in parent_child_result["child_docs"]:
                chunk.metadata["split_strategy"] = "child"
            
            for chunk in semantic_chunks:
                chunk.metadata.update({
                    "split_strategy": "semantic",
                    "split_timestamp": datetime.now().isoformat(),
                    "chunk_size": len(chunk.page_content)
                })
            
            # 性能对比
            performance_comparison = {
                "standard": {"time": standard_time, "chunks": len(standard_chunks)},
                "parent_child": {"time": parent_child_time, "chunks": len(parent_child_result["parent_docs"]) + len(parent_child_result["child_docs"])},
                "semantic": {"time": semantic_time, "chunks": len(semantic_chunks)}
            }
            
            result = {
                "standard_chunks": standard_chunks,
                "parent_docs": parent_child_result["parent_docs"],
                "child_docs": parent_child_result["child_docs"],
                "semantic_chunks": semantic_chunks,
                "parent_to_children": parent_child_result["parent_to_children"],
                "strategy": "hybrid",
                "performance_comparison": performance_comparison
            }
        
        else:
            raise ValueError(f"不支持的分割策略: {strategy}")
        
        processing_time = time.time() - start_time
        
        # 更新统计信息
        self.hybrid_stats['strategies_used'].append(strategy)
        self.hybrid_stats['total_processing_time'] += processing_time
        self.hybrid_stats['last_strategy'] = strategy
        if strategy == "hybrid" and "performance_comparison" in result:
            self.hybrid_stats['performance_comparison'] = result["performance_comparison"]
        
        # 记录到持久化系统
        if self.persistence_manager:
            try:
                total_chunks = 0
                if strategy == "standard" or strategy == "semantic":
                    total_chunks = len(result.get("chunks", []))
                elif strategy == "parent_child":
                    total_chunks = len(result.get("parent_docs", [])) + len(result.get("child_docs", []))
                elif strategy == "hybrid":
                    total_chunks = (len(result.get("standard_chunks", [])) + 
                                  len(result.get("parent_docs", [])) + 
                                  len(result.get("child_docs", [])) + 
                                  len(result.get("semantic_chunks", [])))
                
                self.persistence_manager.save_document_history(
                    file_path="text_splitting_hybrid",
                    file_name=f"split_{strategy}",
                    file_size=sum(len(doc.page_content) for doc in documents),
                    processing_time=processing_time,
                    success=True,
                    chunk_count=total_chunks,
                    strategy=strategy
                )
            except Exception as e:
                print(f"⚠️ 持久化记录失败: {e}")
        
        print(f"✅ '{strategy}' 策略分割完成，耗时 {processing_time:.2f}s")
        
        # 添加处理时间到结果
        result["processing_time"] = processing_time
        result["hybrid_stats"] = self.hybrid_stats.copy()
        
        return result
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """获取混合分割器统计信息"""
        return self.hybrid_stats.copy()
    
    def compare_strategies(self, documents: List[Document]) -> Dict[str, Any]:
        """比较不同分割策略的性能"""
        print(f"🔍 开始策略性能比较...")
        
        strategies = ["standard", "parent_child", "semantic"]
        comparison_results = {}
        
        for strategy in strategies:
            print(f"   测试策略: {strategy}")
            start_time = time.time()
            result = self.split_documents_with_strategy(documents, strategy)
            end_time = time.time()
            
            # 计算统计信息
            total_chunks = 0
            total_chars = 0
            
            if strategy == "parent_child":
                chunks = result.get("parent_docs", []) + result.get("child_docs", [])
            else:
                chunks = result.get("chunks", [])
            
            total_chunks = len(chunks)
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
            
            comparison_results[strategy] = {
                "processing_time": end_time - start_time,
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "avg_chunk_size": avg_chunk_size,
                "chunks_per_second": total_chunks / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }
        
        # 找出最佳策略
        fastest_strategy = min(comparison_results.keys(), 
                             key=lambda x: comparison_results[x]["processing_time"])
        most_efficient = min(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]["avg_chunk_size"])
        
        comparison_results["summary"] = {
            "fastest_strategy": fastest_strategy,
            "most_efficient_strategy": most_efficient,
            "total_documents": len(documents),
            "comparison_timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ 策略比较完成，最快策略: {fastest_strategy}")
        
        return comparison_results
    
    def recommend_strategy(self, documents: List[Document]) -> str:
        """根据文档特征推荐最佳分割策略"""
        if not documents:
            return "standard"
        
        # 分析文档特征
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_doc_length = total_length / len(documents)
        max_doc_length = max(len(doc.page_content) for doc in documents)
        
        # 检查文档结构特征
        has_structured_content = any(
            any(marker in doc.page_content for marker in ["\n\n", "##", "###", "。", "！", "？"])
            for doc in documents
        )
        
        # 推荐策略
        if avg_doc_length > 5000 and has_structured_content:
            return "parent_child"
        elif avg_doc_length > 2000 and has_structured_content:
            return "semantic"
        elif max_doc_length > 10000:
            return "hybrid"
        else:
            return "standard"


# 辅助函数
def create_text_splitter(strategy: str = "parent_child", persistence_manager=None) -> HybridTextSplitter:
    """创建文本分割器的工厂函数"""
    return HybridTextSplitter(persistence_manager=persistence_manager)


def analyze_document_structure(documents: List[Document]) -> Dict[str, Any]:
    """分析文档结构特征"""
    if not documents:
        return {}
    
    total_docs = len(documents)
    total_length = sum(len(doc.page_content) for doc in documents)
    
    # 统计各种结构特征
    structure_analysis = {
        "total_documents": total_docs,
        "total_characters": total_length,
        "avg_document_length": total_length / total_docs,
        "min_document_length": min(len(doc.page_content) for doc in documents),
        "max_document_length": max(len(doc.page_content) for doc in documents),
        "has_paragraphs": sum(1 for doc in documents if "\n\n" in doc.page_content),
        "has_chinese_sentences": sum(1 for doc in documents if any(punct in doc.page_content for punct in ["。", "！", "？"])),
        "has_english_sentences": sum(1 for doc in documents if any(punct in doc.page_content for punct in [".", "!", "?"])),
        "has_lists": sum(1 for doc in documents if any(marker in doc.page_content for marker in ["- ", "* ", "1. ", "2. "])),
        "has_headers": sum(1 for doc in documents if any(header in doc.page_content for header in ["#", "##", "###"]))
    }
    
    return structure_analysis
