#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiliconFlow多模态模型的LangChain集成
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import json
import base64
from io import BytesIO
from PIL import Image

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field
from typing import Dict, Any
from config import Config


class SiliconFlowMultiModal(BaseLLM):
    """SiliconFlow多模态模型的LangChain接口"""
    
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    model: str = Field(default="")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    headers: Dict[str, str] = Field(default_factory=dict)
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ):
        # 设置字段值
        api_key_val = api_key or Config.SILICONFLOW_API_KEY
        kwargs.update({
            'api_key': api_key_val,
            'base_url': base_url or Config.SILICONFLOW_BASE_URL,
            'model': model or Config.MULTIMODAL_MODEL,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'headers': {
                "Authorization": f"Bearer {api_key_val}",
                "Content-Type": "application/json"
            }
        })
        
        # 调用父类初始化
        super().__init__(**kwargs)
        
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required")
    
    @property
    def _llm_type(self) -> str:
        return "siliconflow_multimodal"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> LLMResult:
        """生成文本"""
        generations = []
        
        for prompt in prompts:
            try:
                response_text = self._call_api(prompt)
                generations.append([Generation(text=response_text)])
            except Exception as e:
                print(f"生成文本时出错: {e}")
                generations.append([Generation(text=f"错误: {str(e)}")])
        
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """调用模型生成文本"""
        return self._call_api(prompt)
    
    def _call_api(self, prompt: str, image_path: Optional[str] = None) -> str:
        """调用SiliconFlow多模态API"""
        url = f"{self.base_url}/chat/completions"
        
        # 构建消息内容
        if image_path and Path(image_path).exists():
            # 包含图像的多模态请求
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._encode_image(image_path)
                    }
                }
            ]
        else:
            # 纯文本请求
            content = prompt
        
        messages = [
            {"role": "user", "content": content}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                raise ValueError("API响应格式错误：缺少choices字段")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"请求SiliconFlow API时出错: {e}")
            raise
        except (KeyError, IndexError) as e:
            print(f"解析API响应时出错: {e}")
            raise
        except Exception as e:
            print(f"调用多模态LLM时发生未知错误: {e}")
            raise
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为base64字符串"""
        try:
            with Image.open(image_path) as img:
                # 调整图像大小以减少token使用
                if img.size[0] > 1024 or img.size[1] > 1024:
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 编码为base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_data = buffer.getvalue()
                
                return f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
        
        except Exception as e:
            print(f"编码图像时出错: {e}")
            raise
    
    def process_image_with_text(self, image_path: str, prompt: str) -> str:
        """处理图像和文本"""
        return self._call_api(prompt, image_path)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取标识参数"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
