#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiliconFlow重排序器的LangChain集成
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from config import Config


class SiliconFlowReranker:
    """SiliconFlow重排序器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        self.api_key = api_key or Config.SILICONFLOW_API_KEY
        self.base_url = base_url or Config.SILICONFLOW_BASE_URL
        self.model = model or Config.RERANKER_MODEL
        self.top_k = top_k or Config.TOP_K
        
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Document]:
        """重排序文档"""
        if not documents:
            return documents
        
        top_k = top_k or self.top_k
        
        try:
            # 准备文档文本
            texts = [doc.page_content for doc in documents]
            
            # 调用重排序API
            scores = self._call_rerank_api(query, texts)
            
            # 创建(文档, 分数)对并排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个文档
            reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
            
            print(f"重排序完成，返回前 {len(reranked_docs)} 个文档")
            return reranked_docs
            
        except Exception as e:
            print(f"重排序时出错: {e}")
            # 如果重排序失败，返回原始文档的前top_k个
            return documents[:top_k]
    
    def _call_rerank_api(self, query: str, texts: List[str]) -> List[float]:
        """调用重排序API"""
        url = f"{self.base_url}/rerank"
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_k": len(texts),  # 获取所有文档的分数
            "return_documents": False  # 只返回分数
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "results" not in result:
                raise ValueError("API响应格式错误：缺少results字段")
            
            # 提取分数并按原始顺序排列
            scores = [0.0] * len(texts)
            for item in result["results"]:
                index = item["index"]
                score = item["relevance_score"]
                scores[index] = score
            
            return scores
            
        except requests.exceptions.RequestException as e:
            print(f"请求重排序API时出错: {e}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            print(f"解析重排序API响应时出错: {e}")
            raise
        except Exception as e:
            print(f"调用重排序API时发生未知错误: {e}")
            raise
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """重排序文档并返回分数"""
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        try:
            # 准备文档文本
            texts = [doc.page_content for doc in documents]
            
            # 调用重排序API
            scores = self._call_rerank_api(query, texts)
            
            # 创建(文档, 分数)对并排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个文档和分数
            return doc_score_pairs[:top_k]
            
        except Exception as e:
            print(f"重排序时出错: {e}")
            # 如果重排序失败，返回原始文档和默认分数
            return [(doc, 0.5) for doc in documents[:top_k]]
