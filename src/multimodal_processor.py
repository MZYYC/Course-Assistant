#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangChain的多模态处理器
支持图像处理、持久化集成和性能监控
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime
from PIL import Image, ImageFile

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from .components import component_manager
from config import Config

# 允许加载不完整的图像文件
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LangChainMultimodalProcessor:
    """基于LangChain的多模态处理器"""
    
    def __init__(self, persistence_manager=None):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
        self.multimodal_chain = component_manager.create_multimodal_chain()
        self.persistence_manager = persistence_manager
        
        # 处理统计
        self.processing_stats = {
            'total_documents': 0,
            'documents_with_images': 0,
            'total_images_processed': 0,
            'processing_time': 0,
            'successful_descriptions': 0,
            'failed_descriptions': 0,
            'last_processing_time': None
        }
        
        print(f"多模态处理器初始化完成，支持格式: {', '.join(sorted(self.supported_image_formats))}")
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """处理文档，提取图像并生成描述"""
        start_time = time.time()
        processed_documents = []
        
        print(f"🖼️ 开始多模态处理，文档数量: {len(documents)}")
        
        # 重置统计
        self.processing_stats.update({
            'total_documents': len(documents),
            'documents_with_images': 0,
            'total_images_processed': 0,
            'successful_descriptions': 0,
            'failed_descriptions': 0
        })
        
        for i, doc in enumerate(documents):
            try:
                print(f"   处理文档 [{i+1}/{len(documents)}]: {doc.metadata.get('source', 'unknown')}")
                
                # 复制原文档
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                )
                
                # 查找相关图像文件
                image_descriptions = self._process_related_images(doc)
                
                # 将图像描述添加到文档内容
                if image_descriptions:
                    processed_doc.page_content += "\n\n📷 图像内容描述:\n" + "\n".join(image_descriptions)
                    processed_doc.metadata['has_images'] = True
                    processed_doc.metadata['image_count'] = len(image_descriptions)
                    processed_doc.metadata['multimodal_processed'] = True
                    processed_doc.metadata['processing_timestamp'] = datetime.now().isoformat()
                    
                    self.processing_stats['documents_with_images'] += 1
                    self.processing_stats['total_images_processed'] += len(image_descriptions)
                else:
                    processed_doc.metadata['has_images'] = False
                    processed_doc.metadata['image_count'] = 0
                    processed_doc.metadata['multimodal_processed'] = True
                    processed_doc.metadata['processing_timestamp'] = datetime.now().isoformat()
                
                processed_documents.append(processed_doc)
                
            except Exception as e:
                print(f"   ❌ 处理文档时出错: {e}")
                # 如果处理失败，保留原文档并添加错误标记
                doc.metadata['multimodal_error'] = str(e)
                doc.metadata['multimodal_processed'] = False
                processed_documents.append(doc)
                continue
        
        processing_time = time.time() - start_time
        self.processing_stats['processing_time'] = processing_time
        self.processing_stats['last_processing_time'] = datetime.now().isoformat()
        
        # 记录到持久化系统
        if self.persistence_manager:
            try:
                self.persistence_manager.save_document_history(
                    file_path="multimodal_processing",
                    file_name=f"multimodal-{len(documents)}-docs",
                    file_size=sum(len(doc.page_content.encode('utf-8')) for doc in processed_documents),
                    processing_time=processing_time,
                    success=True,
                    chunk_count=len(processed_documents),
                    strategy="multimodal_enhancement"
                )
            except Exception as e:
                print(f"⚠️ 持久化记录失败: {e}")
        
        print(f"✅ 多模态处理完成: {len(processed_documents)} 文档, {self.processing_stats['total_images_processed']} 图像, 耗时 {processing_time:.2f}s")
        return processed_documents
    
    def _process_related_images(self, document: Document) -> List[str]:
        """处理与文档相关的图像"""
        image_descriptions = []
        
        # 获取文档所在目录
        source_path = document.metadata.get('source')
        if not source_path:
            return image_descriptions
        
        source_dir = Path(source_path).parent
        
        # 查找同目录下的图像文件
        image_files = []
        for ext in self.supported_image_formats:
            image_files.extend(source_dir.glob(f"*{ext}"))
            image_files.extend(source_dir.glob(f"*{ext.upper()}"))
        
        # 移除重复文件并排序
        image_files = sorted(set(image_files))
        
        print(f"     发现 {len(image_files)} 个图像文件")
        
        # 处理找到的图像（限制数量以避免性能问题）
        max_images = 5
        for i, image_path in enumerate(image_files[:max_images]):
            try:
                print(f"       处理图像 [{i+1}/{min(len(image_files), max_images)}]: {image_path.name}")
                description = self._describe_image(image_path)
                if description:
                    image_descriptions.append(
                        f"📸 {image_path.name}: {description}"
                    )
                    self.processing_stats['successful_descriptions'] += 1
                else:
                    self.processing_stats['failed_descriptions'] += 1
            except Exception as e:
                print(f"       ❌ 处理图像 {image_path} 失败: {e}")
                self.processing_stats['failed_descriptions'] += 1
                continue
        
        if len(image_files) > max_images:
            image_descriptions.append(f"📝 注意: 发现 {len(image_files)} 个图像文件，仅处理了前 {max_images} 个")
        
        return image_descriptions
    
    def _describe_image(self, image_path: Path) -> str:
        """使用多模态模型描述图像"""
        try:
            # 检查图像文件是否存在且可读
            if not image_path.exists():
                return ""
            
            # 获取文件大小信息
            file_size = image_path.stat().st_size
            if file_size > 10 * 1024 * 1024:  # 10MB限制
                return f"图像文件过大 ({file_size / 1024 / 1024:.1f}MB)，跳过处理"
            
            # 验证图像文件
            try:
                with Image.open(image_path) as img:
                    # 确保图像有效
                    img.verify()
                    # 重新打开获取信息（verify会关闭文件）
                    with Image.open(image_path) as img2:
                        width, height = img2.size
                        format_info = img2.format or "Unknown"
                        mode = img2.mode
            except Exception as e:
                return f"无法读取图像文件: {str(e)}"
            
            # 使用多模态链处理图像
            try:
                multimodal_llm = component_manager.multimodal_llm
                # 检查是否有自定义的处理方法
                if hasattr(multimodal_llm, 'process_image_with_text'):
                    description = getattr(multimodal_llm, 'process_image_with_text')(
                        str(image_path), 
                        Config.IMAGE_SUMMARY_PROMPT
                    )
                    if description and description.strip():
                        return f"{description.strip()} (尺寸: {width}x{height}, 格式: {format_info})"
                    else:
                        return f"图像处理完成但未生成描述 (尺寸: {width}x{height}, 格式: {format_info})"
                else:
                    # 后备方案：返回基本图像信息
                    return f"图像信息 - 尺寸: {width}x{height}, 格式: {format_info}, 模式: {mode}, 大小: {file_size/1024:.1f}KB"
            except Exception as e:
                # 如果多模态处理失败，至少返回基本信息
                return f"多模态处理失败，基本信息 - 尺寸: {width}x{height}, 格式: {format_info} (错误: {str(e)})"
            
        except Exception as e:
            print(f"         ⚠️ 描述图像 {image_path} 时出错: {e}")
            return f"处理图像时出错: {str(e)}"
    
    def process_single_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """处理单个图像"""
        try:
            prompt = prompt or Config.IMAGE_SUMMARY_PROMPT
            
            multimodal_llm = component_manager.multimodal_llm
            # 检查是否有自定义的处理方法
            if hasattr(multimodal_llm, 'process_image_with_text'):
                return getattr(multimodal_llm, 'process_image_with_text')(image_path, prompt)
            else:
                return f"无法处理图像: {image_path} (多模态功能不可用)"
                
        except Exception as e:
            print(f"处理单个图像时出错: {e}")
            return f"处理图像时出错: {str(e)}"
    
    def get_supported_image_formats(self) -> List[str]:
        """获取支持的图像格式"""
        return list(self.supported_image_formats)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()
    
    def analyze_image_distribution(self, documents: List[Document]) -> Dict[str, Any]:
        """分析文档中的图像分布"""
        total_docs = len(documents)
        docs_with_images = sum(1 for doc in documents if doc.metadata.get('has_images', False))
        total_images = sum(doc.metadata.get('image_count', 0) for doc in documents)
        
        # 按文件类型统计
        type_stats = {}
        for doc in documents:
            if doc.metadata.get('has_images', False):
                file_type = doc.metadata.get('file_type', 'unknown')
                if file_type not in type_stats:
                    type_stats[file_type] = {'docs': 0, 'images': 0}
                type_stats[file_type]['docs'] += 1
                type_stats[file_type]['images'] += doc.metadata.get('image_count', 0)
        
        return {
            'total_documents': total_docs,
            'documents_with_images': docs_with_images,
            'documents_without_images': total_docs - docs_with_images,
            'total_images_found': total_images,
            'average_images_per_doc': total_images / docs_with_images if docs_with_images > 0 else 0,
            'image_coverage_rate': docs_with_images / total_docs if total_docs > 0 else 0,
            'by_file_type': type_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def batch_process_images(self, image_paths: List[str], prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量处理图像"""
        results = []
        prompt = prompt or Config.IMAGE_SUMMARY_PROMPT
        
        print(f"🖼️ 批量处理 {len(image_paths)} 个图像...")
        
        for i, image_path in enumerate(image_paths):
            print(f"   处理图像 [{i+1}/{len(image_paths)}]: {Path(image_path).name}")
            
            start_time = time.time()
            try:
                description = self.process_single_image(image_path, prompt)
                processing_time = time.time() - start_time
                
                results.append({
                    'image_path': image_path,
                    'description': description,
                    'success': True,
                    'processing_time': processing_time,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                processing_time = time.time() - start_time
                results.append({
                    'image_path': image_path,
                    'description': f"处理失败: {str(e)}",
                    'success': False,
                    'processing_time': processing_time,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['processing_time'] for r in results)
        
        print(f"✅ 批量处理完成: {successful}/{len(image_paths)} 成功, 总耗时 {total_time:.2f}s")
        
        return results
