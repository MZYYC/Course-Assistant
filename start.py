#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程助手启动脚本
"""

import os
import sys
import faiss
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    # 包名映射：(pip包名, 导入名)
    required_packages = [
        ("gradio", "gradio"),
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"), 
        ("faiss-cpu", "faiss"),
        ("pandas", "pandas")
    ]
    
    missing_packages = []
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"❌ 缺少以下依赖项: {', '.join(missing_packages)}")
        print("请运行以下命令安装依赖项:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖项已安装")
    return True

def setup_environment():
    """设置环境"""
    print("🔧 设置运行环境...")
    
    # 创建必要的目录
    directories = [
        "data/documents/notes",
        "data/documents/papers", 
        "data/persistence",
        "data/vector_store",
        "exports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ 环境设置完成")

def check_config():
    """检查配置文件"""
    print("📋 检查配置文件...")
    
    config_file = Path("config.py")
    if not config_file.exists():
        print("❌ 配置文件 config.py 不存在")
        return False
    
    try:
        from config import Config
        if not hasattr(Config, 'SILICONFLOW_API_KEY') or not Config.SILICONFLOW_API_KEY:
            print("⚠️ SiliconFlow API Key 未配置")
            print("请在 config.py 中设置 SILICONFLOW_API_KEY")
        else:
            print("✅ 配置文件检查通过")
    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        return False
    
    return True

def start_app():
    """启动应用"""
    print("🚀 启动课程助手...")
    print("=" * 50)
    
    # 运行应用
    try:
        from app import create_course_assistant_interface
        from src.rag_system import LangChainRAGSystem
        
        print("🔄 正在初始化RAG系统...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        # 预先初始化RAG系统
        rag_system = LangChainRAGSystem()
        print("📚 正在加载和处理文档...")
        
        init_result = rag_system.initialize_system(force_rebuild=False)
        
        if init_result["status"] == "success":
            print("✅ RAG系统初始化成功!")
            print(f"📊 已加载 {init_result.get('num_documents', 0)} 个文档")
            print(f"🔄 分割策略: {init_result.get('split_strategy', 'unknown')}")
            
            split_stats = init_result.get('split_stats', {})
            if split_stats:
                print(f"📄 父块数量: {split_stats.get('parent_chunks', 0)}")
                print(f"📄 子块数量: {split_stats.get('child_chunks', 0)}")
            
            print(f"⏱️ 初始化耗时: {init_result.get('initialization_time', 0):.2f}秒")
        else:
            print(f"⚠️ RAG系统初始化失败: {init_result.get('message', '未知错误')}")
            print("系统将以未初始化状态启动，您需要在Web界面中手动初始化")
        
        print("\n📱 创建Web界面...")
        interface = create_course_assistant_interface(rag_system)
        
        print("✅ 应用启动成功!")
        print("📱 访问地址: http://localhost:7860")
        print("🛑 按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 启动Gradio服务
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            inbrowser=True,
            favicon_path=None,
            auth=None,
            max_threads=10
        )
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎓 课程助手 - 智能问答系统")
    print("=" * 50)
    
    # 检查依赖项
    if not check_dependencies():
        return
    
    # 设置环境
    setup_environment()
    
    # 检查配置
    if not check_config():
        return
    
    # 启动应用
    start_app()

if __name__ == "__main__":
    main()
