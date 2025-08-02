#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiliconFlow嵌入模型的LangChain集成
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.embeddings import Embeddings
from config import Config


class SiliconFlowEmbeddings(Embeddings):
    """SiliconFlow嵌入模型的LangChain接口"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key or Config.SILICONFLOW_API_KEY
        self.base_url = base_url or Config.SILICONFLOW_BASE_URL
        self.model = model or Config.EMBEDDING_MODEL
        
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"初始化SiliconFlow嵌入模型: {self.model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = []
        
        for text in texts:
            try:
                embedding = self._embed_single_text(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"嵌入文档时出错: {e}")
                # 返回零向量作为后备
                embeddings.append([0.0] * Config.EMBEDDING_DIMENSION)
        
        print(f"成功嵌入 {len(embeddings)} 个文档")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        try:
            embedding = self._embed_single_text(text)
            print(f"成功嵌入查询文本")
            return embedding
        except Exception as e:
            print(f"嵌入查询时出错: {e}")
            # 返回零向量作为后备
            return [0.0] * Config.EMBEDDING_DIMENSION
    
    def _embed_single_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        if not text.strip():
            return [0.0] * Config.EMBEDDING_DIMENSION
        
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "input": text,
            "model": self.model
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
            
            if "data" not in result or not result["data"]:
                raise ValueError("API响应格式错误：缺少data字段")
            
            embedding = result["data"][0]["embedding"]
            
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise ValueError("API返回的嵌入向量格式错误")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            print(f"请求SiliconFlow API时出错: {e}")
            raise
        except (KeyError, IndexError, ValueError) as e:
            print(f"解析API响应时出错: {e}")
            raise
        except Exception as e:
            print(f"嵌入文本时发生未知错误: {e}")
            raise
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档列表（目前使用同步实现）"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询文本（目前使用同步实现）"""
        return self.embed_query(text)
