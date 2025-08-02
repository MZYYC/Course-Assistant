#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangChain的多格式文档加载器
支持多种文档格式的加载，包含持久化和进度跟踪功能
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import time
import mimetypes
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader
)
from config import Config


class LangChainDocumentLoader:
    """基于LangChain的多格式文档加载器"""
    
    def __init__(self, documents_path: str, persistence_manager=None):
        self.documents_path = Path(documents_path)
        self.persistence_manager = persistence_manager
        self.supported_extensions = {
            # 文档格式
            '.pdf': self._load_pdf,
            '.doc': self._load_docx,
            '.docx': self._load_docx,
            '.ppt': self._load_pptx,
            '.pptx': self._load_pptx,
            
            # 文本格式
            '.txt': self._load_text,
            '.md': self._load_markdown,
            '.markdown': self._load_markdown,
            '.rst': self._load_text,
            
            # 结构化数据
            '.json': self._load_json,
            '.jsonl': self._load_jsonl,
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            
            # 网页和标记语言
            '.html': self._load_html,
            '.htm': self._load_html,
            '.xml': self._load_xml,
            
            # 代码文件
            '.py': self._load_text,
            '.js': self._load_text,
            '.ts': self._load_text,
            '.java': self._load_text,
            '.cpp': self._load_text,
            '.c': self._load_text,
            '.cs': self._load_text,
            '.go': self._load_text,
            '.rs': self._load_text,
            '.php': self._load_text,
            '.rb': self._load_text,
            '.sh': self._load_text,
            '.sql': self._load_text,
            '.yml': self._load_text,
            '.yaml': self._load_text,
            '.toml': self._load_text,
            '.ini': self._load_text,
            '.cfg': self._load_text,
            '.conf': self._load_text,
            '.log': self._load_text
        }
        
        # 加载统计
        self.load_stats = {
            'total_files': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'total_documents': 0,
            'by_type': {},
            'errors': []
        }
        
        print(f"文档加载器初始化完成，支持 {len(self.supported_extensions)} 种文件格式")
    
    def load_documents(self) -> List[Document]:
        """递归加载所有支持的文档"""
        documents = []
        start_time = time.time()
        
        # 重置统计
        self.load_stats = {
            'total_files': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'total_documents': 0,
            'by_type': {},
            'errors': [],
            'start_time': datetime.now().isoformat(),
            'processing_time': 0
        }
        
        if not self.documents_path.exists():
            print(f"文档路径不存在: {self.documents_path}")
            return documents
        
        print(f"开始加载文档，路径: {self.documents_path}")
        
        # 收集所有支持的文件
        all_files = []
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                if file_extension in self.supported_extensions:
                    all_files.append(file_path)
        
        self.load_stats['total_files'] = len(all_files)
        print(f"找到 {len(all_files)} 个支持的文件")
        
        # 递归遍历所有文件
        for i, file_path in enumerate(all_files):
            file_extension = file_path.suffix.lower()
            file_type = self._get_file_type(file_extension)
            
            try:
                print(f"正在加载 [{i+1}/{len(all_files)}]: {file_path.name}")
                
                loader_func = self.supported_extensions[file_extension]
                docs = loader_func(file_path)
                
                if docs:
                    documents.extend(docs)
                    self.load_stats['successful_loads'] += 1
                    self.load_stats['total_documents'] += len(docs)
                    
                    # 按类型统计
                    if file_type not in self.load_stats['by_type']:
                        self.load_stats['by_type'][file_type] = {'files': 0, 'documents': 0}
                    self.load_stats['by_type'][file_type]['files'] += 1
                    self.load_stats['by_type'][file_type]['documents'] += len(docs)
                    
                    print(f"成功加载: {file_path.name}, 获得 {len(docs)} 个文档")
                else:
                    self.load_stats['failed_loads'] += 1
                    error_msg = f"文件为空或无内容: {file_path}"
                    self.load_stats['errors'].append(error_msg)
                    print(f"⚠️ {error_msg}")
                    
            except Exception as e:
                self.load_stats['failed_loads'] += 1
                error_msg = f"加载文件失败 {file_path}: {str(e)}"
                self.load_stats['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                continue
        
        # 计算处理时间
        processing_time = time.time() - start_time
        self.load_stats['processing_time'] = processing_time
        self.load_stats['end_time'] = datetime.now().isoformat()
        
        # 打印统计信息
        self._print_load_statistics()
        
        # 记录文档处理历史到持久化系统
        if self.persistence_manager and documents:
            try:
                self.persistence_manager.save_document_history(
                    file_path=str(self.documents_path),
                    file_name=f"multi-format-{self.load_stats['successful_loads']}-files",
                    file_size=sum(len(doc.page_content.encode('utf-8')) for doc in documents),
                    processing_time=processing_time,
                    success=True,
                    chunk_count=len(documents),
                    strategy="multi-format-loading"
                )
            except Exception as e:
                print(f"⚠️ 持久化记录失败: {e}")
        
        return documents
    
    def _get_file_type(self, extension: str) -> str:
        """根据扩展名获取文件类型"""
        type_mapping = {
            '.pdf': 'PDF',
            '.doc': 'Word', '.docx': 'Word',
            '.ppt': 'PowerPoint', '.pptx': 'PowerPoint',
            '.txt': 'Text', '.md': 'Markdown', '.markdown': 'Markdown', '.rst': 'Text',
            '.json': 'JSON', '.jsonl': 'JSONL',
            '.csv': 'CSV', '.xlsx': 'Excel', '.xls': 'Excel',
            '.html': 'HTML', '.htm': 'HTML', '.xml': 'XML',
            '.py': 'Code', '.js': 'Code', '.ts': 'Code', '.java': 'Code',
            '.cpp': 'Code', '.c': 'Code', '.cs': 'Code', '.go': 'Code',
            '.rs': 'Code', '.php': 'Code', '.rb': 'Code', '.sh': 'Code',
            '.sql': 'Code', '.yml': 'Config', '.yaml': 'Config',
            '.toml': 'Config', '.ini': 'Config', '.cfg': 'Config',
            '.conf': 'Config', '.log': 'Log'
        }
        return type_mapping.get(extension, 'Unknown')
    
    def _print_load_statistics(self):
        """打印加载统计信息"""
        stats = self.load_stats
        print(f"\n📊 文档加载统计:")
        print(f"   总文件数: {stats['total_files']}")
        print(f"   成功加载: {stats['successful_loads']}")
        print(f"   加载失败: {stats['failed_loads']}")
        print(f"   总文档数: {stats['total_documents']}")
        print(f"   处理时间: {stats['processing_time']:.2f}s")
        
        if stats['by_type']:
            print(f"   按类型统计:")
            for file_type, counts in stats['by_type'].items():
                print(f"     {file_type}: {counts['files']} 文件, {counts['documents']} 文档")
        
        if stats['errors']:
            print(f"   错误数: {len(stats['errors'])}")
            if len(stats['errors']) <= 5:
                for error in stats['errors']:
                    print(f"     - {error}")
            else:
                for error in stats['errors'][:3]:
                    print(f"     - {error}")
                print(f"     ... 还有 {len(stats['errors']) - 3} 个错误")
        
        print(f"文档加载完成，共加载 {stats['total_documents']} 个文档\n")
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文档"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'pdf',
                    'file_name': file_path.name
                })
            
            return documents
        except Exception as e:
            print(f"加载PDF文件失败 {file_path}: {e}")
            return []
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文档"""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'markdown',
                    'file_name': file_path.name
                })
            
            return documents
        except Exception as e:
            print(f"加载Markdown文件失败 {file_path}: {e}")
            return []
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """加载Word文档"""
        try:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'docx',
                    'file_name': file_path.name
                })
            
            return documents
        except Exception as e:
            print(f"加载Word文件失败 {file_path}: {e}")
            return []
    
    def _load_pptx(self, file_path: Path) -> List[Document]:
        """加载PowerPoint文档"""
        try:
            loader = UnstructuredPowerPointLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'pptx',
                    'file_name': file_path.name
                })
            
            return documents
        except Exception as e:
            print(f"加载PowerPoint文件失败 {file_path}: {e}")
            return []
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """加载文本文档"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'txt',
                    'file_name': file_path.name
                })
            
            return documents
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ['gbk', 'gb2312', 'latin-1']:
                try:
                    loader = TextLoader(str(file_path), encoding=encoding)
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata.update({
                            'source': str(file_path),
                            'file_type': 'txt',
                            'file_name': file_path.name,
                            'encoding': encoding
                        })
                    return documents
                except:
                    continue
            print(f"加载文本文件失败 {file_path}: 无法识别编码")
            return []
        except Exception as e:
            print(f"加载文本文件失败 {file_path}: {e}")
            return []
    
    def _load_jsonl(self, file_path: Path) -> List[Document]:
        """加载JSONL文件 (每行一个JSON对象)"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # 提取文本内容
                            text = ""
                            if isinstance(data, dict):
                                # 尝试常见的文本字段
                                for key in ['text', 'content', 'body', 'message', 'description']:
                                    if key in data and data[key]:
                                        text = str(data[key])
                                        break
                                if not text:
                                    text = json.dumps(data, ensure_ascii=False)
                            else:
                                text = str(data)
                            
                            if text:
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'source': str(file_path),
                                        'line_number': i + 1,
                                        'file_type': 'jsonl',
                                        'file_name': file_path.name
                                    }
                                )
                                documents.append(doc)
                        except json.JSONDecodeError:
                            continue
            return documents
        except Exception as e:
            print(f"JSONL文件加载失败 {file_path}: {e}")
            return []
    
    def _load_csv(self, file_path: Path) -> List[Document]:
        """加载CSV文件"""
        try:
            loader = CSVLoader(file_path=str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'csv',
                    'file_name': file_path.name
                })
            
            return documents
        except Exception as e:
            print(f"CSV加载失败，尝试手动解析: {e}")
            # 手动解析CSV
            documents = []
            try:
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if row:
                            # 将所有字段组合成文本
                            text_parts = []
                            for key, value in row.items():
                                if value:
                                    text_parts.append(f"{key}: {value}")
                            
                            if text_parts:
                                text = "\n".join(text_parts)
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        'source': str(file_path),
                                        'row_number': i + 1,
                                        'file_type': 'csv',
                                        'file_name': file_path.name
                                    }
                                )
                                documents.append(doc)
                return documents
            except Exception:
                return []
    
    def _load_excel(self, file_path: Path) -> List[Document]:
        """加载Excel文件"""
        try:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(str(file_path), mode="elements")
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'excel',
                    'file_name': file_path.name
                })
            
            return documents
        except ImportError:
            print("未安装unstructured库，尝试使用pandas解析Excel")
            # 使用pandas解析
            try:
                import pandas as pd
                documents = []
                
                # 读取所有工作表
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # 将每行转换为文档
                    for row_idx, (index, row) in enumerate(df.iterrows()):
                        text_parts = []
                        for col, value in row.items():
                            if pd.notna(value):
                                text_parts.append(f"{col}: {value}")
                        
                        if text_parts:
                            text = "\n".join(text_parts)
                            doc = Document(
                                page_content=text,
                                metadata={
                                    'source': str(file_path),
                                    'sheet_name': sheet_name,
                                    'row_number': row_idx + 1,
                                    'file_type': 'excel',
                                    'file_name': file_path.name
                                }
                            )
                            documents.append(doc)
                
                return documents
            except ImportError:
                print("未安装pandas库，无法解析Excel文件")
                return []
            except Exception:
                return []
        except Exception:
            return []
    
    def _load_html(self, file_path: Path) -> List[Document]:
        """加载HTML文件"""
        try:
            from langchain_community.document_loaders import UnstructuredHTMLLoader
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'html',
                    'file_name': file_path.name
                })
            
            return documents
        except ImportError:
            print("未安装unstructured库，尝试使用BeautifulSoup解析HTML")
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                
                # 移除脚本和样式标签
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 提取文本
                text = soup.get_text()
                # 清理空白
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': str(file_path),
                            'file_type': 'html',
                            'file_name': file_path.name,
                            'title': soup.title.string if soup.title else None
                        }
                    )
                    return [doc]
                return []
            except ImportError:
                print("未安装BeautifulSoup库，无法解析HTML文件")
                return []
            except Exception:
                return []
        except Exception:
            return []
    
    def _load_xml(self, file_path: Path) -> List[Document]:
        """加载XML文件"""
        try:
            from langchain_community.document_loaders import UnstructuredXMLLoader
            loader = UnstructuredXMLLoader(str(file_path))
            documents = loader.load()
            
            # 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': 'xml',
                    'file_name': file_path.name
                })
            
            return documents
        except ImportError:
            print("未安装unstructured库，尝试使用xml.etree解析XML")
            try:
                import xml.etree.ElementTree as ET
                
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                def extract_text(element):
                    """递归提取XML元素的文本"""
                    texts = []
                    if element.text and element.text.strip():
                        texts.append(element.text.strip())
                    
                    for child in element:
                        child_texts = extract_text(child)
                        texts.extend(child_texts)
                        
                        if child.tail and child.tail.strip():
                            texts.append(child.tail.strip())
                    
                    return texts
                
                all_texts = extract_text(root)
                
                if all_texts:
                    text = '\n'.join(all_texts)
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': str(file_path),
                            'file_type': 'xml',
                            'file_name': file_path.name,
                            'root_tag': root.tag
                        }
                    )
                    return [doc]
                return []
            except Exception:
                return []
        except Exception:
            return []
    
    def _load_json(self, file_path: Path) -> List[Document]:
        """加载JSON文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 将JSON内容转换为文本
            if isinstance(data, dict):
                content = json.dumps(data, ensure_ascii=False, indent=2)
            elif isinstance(data, list):
                content = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                content = str(data)
            
            document = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'file_type': 'json',
                    'file_name': file_path.name
                }
            )
            
            return [document]
            
        except Exception as e:
            print(f"加载JSON文件失败 {file_path}: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式"""
        return list(self.supported_extensions.keys())
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        start_time = time.time()
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {path_obj}")
        
        file_extension = path_obj.suffix.lower()
        if file_extension not in self.supported_extensions:
            raise ValueError(f"不支持的文件格式: {file_extension}")
        
        loader_func = self.supported_extensions[file_extension]
        documents = loader_func(path_obj)
        
        processing_time = time.time() - start_time
        
        # 记录单个文档处理历史
        if self.persistence_manager and documents:
            try:
                file_size = path_obj.stat().st_size if path_obj.exists() else 0
                self.persistence_manager.save_document_history(
                    file_path=str(path_obj),
                    file_name=path_obj.name,
                    file_size=file_size,
                    processing_time=processing_time,
                    success=True,
                    chunk_count=len(documents),
                    strategy=f"single-file-{self._get_file_type(file_extension)}"
                )
            except Exception as e:
                print(f"⚠️ 持久化记录失败: {e}")
        
        return documents
    
    def get_load_statistics(self) -> Dict:
        """获取最近一次加载的统计信息"""
        return getattr(self, 'load_stats', {})
    
    def get_supported_formats_info(self) -> Dict[str, List[str]]:
        """获取支持格式的详细信息"""
        format_groups = {
            'Document Formats': ['.pdf', '.doc', '.docx', '.ppt', '.pptx'],
            'Text Formats': ['.txt', '.md', '.markdown', '.rst'],
            'Data Formats': ['.json', '.jsonl', '.csv', '.xlsx', '.xls'],
            'Web Formats': ['.html', '.htm', '.xml'],
            'Code Formats': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.rb', '.sh', '.sql'],
            'Config Formats': ['.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf'],
            'Log Formats': ['.log']
        }
        
        # 验证支持的格式
        supported_formats = {}
        for category, extensions in format_groups.items():
            supported_formats[category] = [ext for ext in extensions if ext in self.supported_extensions]
        
        return supported_formats
