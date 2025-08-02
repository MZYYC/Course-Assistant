#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程助手测试套件
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """测试基本导入功能"""
    
    def test_core_imports(self):
        """测试核心模块导入"""
        try:
            from src import rag_system
            from src import components
            from src import document_loader
            assert True
        except ImportError as e:
            pytest.fail(f"Core import failed: {e}")
    
    def test_config_import(self):
        """测试配置导入"""
        try:
            from config import Config
            assert hasattr(Config, 'SILICONFLOW_API_KEY')
            assert hasattr(Config, 'EMBEDDING_MODEL')
        except ImportError as e:
            pytest.fail(f"Config import failed: {e}")


class TestRAGSystem:
    """测试RAG系统基本功能"""
    
    def test_rag_system_creation(self):
        """测试RAG系统创建"""
        from src.rag_system import LangChainRAGSystem
        
        rag_system = LangChainRAGSystem()
        assert rag_system is not None
        assert hasattr(rag_system, 'initialize_system')
        assert hasattr(rag_system, 'query')


class TestDocumentLoader:
    """测试文档加载器"""
    
    def test_document_loader_creation(self):
        """测试文档加载器创建"""
        from src.document_loader import LangChainDocumentLoader
        
        loader = LangChainDocumentLoader(documents_path="./documents")
        assert loader is not None
        assert hasattr(loader, 'load_documents')


class TestWebInterface:
    """测试Web界面"""
    
    def test_app_creation(self):
        """测试应用创建"""
        from app import CourseAssistantApp, create_course_assistant_interface
        
        app = CourseAssistantApp()
        assert app is not None
        assert hasattr(app, 'initialize_system')
        assert hasattr(app, 'chat_with_system')
        
        interface = create_course_assistant_interface()
        assert interface is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
