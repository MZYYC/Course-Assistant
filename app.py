#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程助手RAG系统 - Web界面
基于Gradio的教育课程智能问答系统
"""

import sys
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import gradio as gr
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_system import LangChainRAGSystem
from config import Config

class CourseAssistantApp:
    """课程助手应用"""
    
    def __init__(self, pre_initialized_rag_system: Optional[LangChainRAGSystem] = None):
        self.rag_system: Optional[LangChainRAGSystem] = pre_initialized_rag_system
        self.chat_history = []
        self.system_stats = {}
        self.is_initialized = pre_initialized_rag_system is not None
        
        # 如果传入了预初始化的系统，获取其统计信息
        if self.is_initialized and self.rag_system:
            try:
                # 设置默认的统计信息，表示系统已就绪
                self.system_stats = {
                    "文档数量": "已加载",
                    "分割策略": "parent_child",
                    "父块数": "已就绪",
                    "子块数": "已就绪",
                    "初始化时间": "启动时完成",
                    "检索器状态": "全部就绪 ✅"
                }
            except Exception as e:
                print(f"获取预初始化系统状态失败: {e}")
                # 保持初始化状态为True，但使用默认统计信息
                self.system_stats = {
                    "文档数量": "已加载",
                    "分割策略": "parent_child", 
                    "父块数": "已就绪",
                    "子块数": "已就绪",
                    "初始化时间": "启动时完成",
                    "检索器状态": "全部就绪 ✅"
                }
        
    def initialize_system(self, progress=gr.Progress()) -> Tuple[str, str]:
        """初始化RAG系统"""
        try:
            progress(0.1, desc="创建RAG系统实例...")
            self.rag_system = LangChainRAGSystem()
            
            progress(0.3, desc="正在初始化系统...")
            result = self.rag_system.initialize_system(force_rebuild=False)
            
            if result["status"] == "success":
                self.is_initialized = True
                progress(1.0, desc="系统初始化完成!")
                
                # 获取系统统计信息
                self.system_stats = {
                    "文档数量": result.get("num_documents", 0),
                    "分割策略": result.get("split_strategy", "unknown"),
                    "父块数": result.get("split_stats", {}).get("parent_chunks", 0),
                    "子块数": result.get("split_stats", {}).get("child_chunks", 0),
                    "初始化时间": f"{result.get('initialization_time', 0):.2f}秒",
                    "检索器状态": str(result.get("retriever_status", {}))
                }
                
                status_msg = f"""
                ✅ **系统初始化成功！**
                
                📚 **课程文档统计**：
                - 加载文档数量：{self.system_stats['文档数量']} 个
                - 文档分割策略：{self.system_stats['分割策略']}
                - 父级文档块：{self.system_stats['父块数']} 个
                - 子级文档块：{self.system_stats['子块数']} 个
                
                ⚡ **系统性能**：
                - 初始化耗时：{self.system_stats['初始化时间']}
                - 检索器状态：全部就绪 ✅
                
                🎓 **现在您可以开始提问了！**
                """
                return status_msg, "success"
            else:
                return f"❌ 系统初始化失败：{result.get('message', '未知错误')}", "error"
                
        except Exception as e:
            return f"❌ 初始化异常：{str(e)}", "error"
    
    def chat_with_system(
        self, 
        message: str, 
        history: List[Dict[str, str]], 
        use_reranking: bool = True,
        top_k: int = 3,
        progress=gr.Progress()
    ) -> Tuple[str, List[Dict[str, str]], str]:
        """与系统对话"""
        if not self.is_initialized:
            return "", history, "❌ 系统未初始化，请先点击'初始化系统'按钮"
        
        if not message.strip():
            return "", history, "⚠️ 请输入您的问题"
        
        try:
            progress(0.2, desc="正在检索相关课程内容...")
            
            if not self.rag_system:
                return "", history, "❌ RAG系统未初始化"
            
            start_time = time.time()
            result = self.rag_system.query(
                question=message,
                use_reranking=use_reranking,
                top_k=top_k,
                use_cache=True
            )
            query_time = time.time() - start_time
            
            progress(0.8, desc="生成回答中...")
            
            if result.get("status") == "success":
                answer = result.get("answer", "抱歉，无法生成回答。")
                sources = result.get("sources", [])
                
                # 构建源文档信息
                source_info = ""
                if sources:
                    source_info = "\n\n📚 **参考来源**：\n"
                    for i, source in enumerate(sources[:3], 1):
                        file_name = source.get("file_name", "未知文件")
                        content_preview = source.get("content_preview", "")[:100]
                        source_info += f"{i}. **{file_name}** - {content_preview}...\n"
                
                # 添加查询统计
                stats_info = f"\n\n📊 **查询统计**：检索到 {result.get('total_docs_found', 0)} 个相关文档，耗时 {query_time:.2f}秒"
                
                full_answer = answer + source_info + stats_info
                
                # 更新对话历史
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": full_answer})
                
                progress(1.0, desc="完成!")
                return "", history, "✅ 回答生成成功"
                
            else:
                error_msg = f"❌ 查询失败：{result.get('message', '未知错误')}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return "", history, error_msg
                
        except Exception as e:
            error_msg = f"❌ 查询异常：{str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history, error_msg
    
    def get_system_status(self) -> str:
        """获取系统状态"""
        if not self.is_initialized:
            return "❌ 系统未初始化"
        
        try:
            # 简化状态获取，避免调用不存在的方法
            status_info = f"""
            ## 📊 系统运行状态
            
            **基本信息**：
            - 系统状态：{'✅ 运行中' if self.is_initialized else '❌ 未初始化'}
            - 加载文档：{self.system_stats.get('文档数量', 'N/A')} 个
            - 分割策略：{self.system_stats.get('分割策略', 'N/A')}
            
            **文档统计**：
            - 父块数量：{self.system_stats.get('父块数', 'N/A')}
            - 子块数量：{self.system_stats.get('子块数', 'N/A')}
            - 初始化时间：{self.system_stats.get('初始化时间', 'N/A')}
            
            **系统状态**：
            - 检索器：✅ 就绪
            - 重排序：✅ 就绪
            - 缓存：✅ 启用
            """
            
            return status_info
            
        except Exception as e:
            return f"❌ 获取状态失败：{str(e)}"
    
    def clear_chat(self) -> Tuple[List, str]:
        """清空对话历史"""
        return [], "✅ 对话历史已清空"
    
    def export_chat(self, history: List[Dict[str, str]]) -> str:
        """导出对话历史"""
        if not history:
            return "⚠️ 没有对话历史可导出"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
            filepath = Path("exports") / filename
            filepath.parent.mkdir(exist_ok=True)
            
            # 转换新格式到导出格式
            chat_pairs = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i]
                    assistant_msg = history[i + 1]
                    if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
                        chat_pairs.append({
                            "question": user_msg.get("content", ""),
                            "answer": assistant_msg.get("content", ""),
                            "timestamp": datetime.now().isoformat()
                        })
            
            export_data = {
                "export_time": datetime.now().isoformat(),
                "system_info": self.system_stats,
                "chat_history": chat_pairs
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return f"✅ 对话历史已导出到：{filepath}"
            
        except Exception as e:
            return f"❌ 导出失败：{str(e)}"

def create_course_assistant_interface(pre_initialized_rag_system=None):
    """创建课程助手界面"""
    app = CourseAssistantApp(pre_initialized_rag_system)
    
    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .header-text {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .description-text {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 1em;
    }
    .status-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    """
    
    with gr.Blocks(
        css=custom_css,
        title="课程助手 - 智能问答系统"
    ) as interface:
        
        # 标题和描述
        gr.HTML("""
        <div class="header-text">🎓 课程助手 - 智能问答系统</div>
        <div class="description-text">
            基于RAG技术的课程内容智能问答平台 | 上传课件，智能检索，精准回答
        </div>
        """)
        
        # 主要功能区域
        with gr.Tab("💬 智能问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    # 对话界面
                    chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        label="课程助手对话",
                        placeholder="系统已就绪，请输入您的课程问题..." if app.is_initialized 
                                   else "初始化系统后，我将为您解答课程相关问题...",
                        avatar_images=("🧑‍🎓", "🤖"),
                        type="messages"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="请输入您关于课程的问题...",
                            label="输入问题",
                            scale=4
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary")
                        export_btn = gr.Button("导出对话", variant="secondary")
                
                with gr.Column(scale=1):
                    # 系统控制面板
                    gr.Markdown("## ⚙️ 系统控制")
                    
                    init_btn = gr.Button(
                        "✅ 系统已就绪" if app.is_initialized else "🚀 初始化系统", 
                        variant="secondary" if app.is_initialized else "primary",
                        size="lg",
                        interactive=not app.is_initialized
                    )
                    
                    status_display = gr.Markdown(
                        "✅ 系统已初始化并就绪！\n\n可以直接开始提问" if app.is_initialized 
                        else "❌ 系统未初始化\n\n请点击上方按钮初始化系统",
                        label="系统状态"
                    )
                    
                    gr.Markdown("## 🎛️ 查询设置")
                    
                    use_reranking = gr.Checkbox(
                        label="启用重排序",
                        value=True,
                        info="提高查询结果的相关性"
                    )
                    
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="检索文档数量",
                        info="每次查询检索的文档数量"
                    )
        
        # 系统管理标签页
        with gr.Tab("📊 系统状态"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 📈 系统运行状态")
                    refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")
                    system_status = gr.Markdown(
                        "点击'刷新状态'查看系统详细信息",
                        label="详细状态"
                    )
                
                with gr.Column():
                    gr.Markdown("## 📋 使用指南")
                    gr.Markdown("""
                    ### 🚀 快速开始
                    1. **初始化系统**：点击"初始化系统"按钮
                    2. **提出问题**：在输入框中输入课程相关问题
                    3. **获得回答**：系统将基于课程内容生成回答
                    
                    ### 💡 使用技巧
                    - **具体问题**：提问越具体，回答越精准
                    - **课程术语**：使用课程中的专业术语
                    - **多角度提问**：从不同角度探索同一主题
                    
                    ### 📚 支持的文件格式
                    - PDF文档 (.pdf)
                    - PowerPoint演示文稿 (.pptx, .ppt)
                    - Markdown文件 (.md)
                    - JSON数据文件 (.json)
                    - 以及37种其他格式
                    """)
        
        # 关于系统标签页
        with gr.Tab("ℹ️ 关于系统"):
            gr.Markdown(f"""
            ## 🎯 系统介绍
            
            **课程助手**是一个基于检索增强生成（RAG）技术的智能问答系统，专为教育场景设计。
            
            ### ✨ 核心特性
            
            🧠 **智能检索**
            - 父子文档分割策略，确保信息完整性
            - 混合检索（向量检索 + 关键词检索）
            - 智能重排序，提升结果相关性
            
            📚 **多格式支持**
            - 支持37种文件格式
            - 自动多模态内容处理
            - 批量文档导入和管理
            
            🎛️ **高级功能**
            - 实时性能监控
            - 查询缓存机制
            - 对话历史导出
            
            ### 🔧 技术架构
            
            **模型配置**：
            - 嵌入模型：{Config.EMBEDDING_MODEL}
            - 生成模型：{Config.GENERATION_MODEL}
            - 重排序模型：BAAI/bge-reranker-v2-m3
            
            **系统参数**：
            - 文本分割策略：{Config.TEXT_SPLIT_STRATEGY}
            - 父块大小：{Config.PARENT_CHUNK_SIZE}
            - 子块大小：{Config.CHILD_CHUNK_SIZE}
            - 默认检索数量：{Config.TOP_K}
            
            ### 👨‍💻 开发信息
            
            - **版本**：2.0.0
            - **开发框架**：LangChain + Gradio
            - **向量数据库**：FAISS
            - **API服务**：SiliconFlow
            
            ---
            
            💡 **提示**：本系统开源发布，欢迎在GitHub上查看源代码和贡献改进！
            """)
        
        # 事件绑定
        notification = gr.Textbox(visible=False)
        
        # 初始化系统
        init_btn.click(
            fn=app.initialize_system,
            outputs=[status_display, notification]
        )
        
        # 发送消息
        send_btn.click(
            fn=app.chat_with_system,
            inputs=[msg_input, chatbot, use_reranking, top_k],
            outputs=[msg_input, chatbot, notification]
        )
        
        # 回车发送
        msg_input.submit(
            fn=app.chat_with_system,
            inputs=[msg_input, chatbot, use_reranking, top_k],
            outputs=[msg_input, chatbot, notification]
        )
        
        # 清空对话
        clear_btn.click(
            fn=app.clear_chat,
            outputs=[chatbot, notification]
        )
        
        # 导出对话
        export_btn.click(
            fn=app.export_chat,
            inputs=[chatbot],
            outputs=[notification]
        )
        
        # 刷新状态
        refresh_btn.click(
            fn=app.get_system_status,
            outputs=[system_status]
        )
    
    return interface

if __name__ == "__main__":
    # 创建界面
    interface = create_course_assistant_interface()
    
    # 启动服务
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 不创建公共链接
        debug=False,            # 生产模式
        show_error=True,        # 显示错误信息
        inbrowser=True,         # 自动打开浏览器
        favicon_path=None,      # 可以添加自定义图标
        auth=None,             # 可以添加身份验证
        max_threads=10         # 最大线程数
    )
