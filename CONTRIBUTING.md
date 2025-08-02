# 🤝 贡献指南

感谢您对课程助手项目的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 Bug报告
- ✨ 新功能建议
- 📚 文档改进
- 🔧 代码优化
- 🧪 测试用例

## 📋 贡献流程

### 1. 准备工作

1. **Fork 项目** 到您的 GitHub 账户
2. **克隆** Fork 的项目到本地：
   ```bash
   git clone https://github.com/yourusername/course-assistant-rag.git
   cd course-assistant-rag
   ```
3. **创建虚拟环境**：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```
4. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # 开发依赖
   ```

### 2. 开发流程

1. **创建分支**：
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

2. **进行开发**：
   - 遵循代码规范
   - 添加必要的测试
   - 更新相关文档

3. **运行测试**：
   ```bash
   pytest tests/
   black --check .
   flake8 .
   ```

4. **提交代码**：
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   # 或
   git commit -m "fix: 修复具体问题"
   ```

5. **推送分支**：
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建 Pull Request**

### 3. 提交信息规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug修复
- `docs:` 文档更新
- `style:` 代码格式调整
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建工具或辅助工具的变动

示例：
```
feat: 添加多语言支持功能
fix: 修复文档加载时的内存泄漏问题
docs: 更新API使用文档
```

## 📝 代码规范

### Python 代码规范

1. **遵循 PEP 8** 标准
2. **使用 Black** 进行代码格式化：
   ```bash
   black .
   ```
3. **使用 isort** 整理导入：
   ```bash
   isort .
   ```
4. **使用 flake8** 进行代码检查：
   ```bash
   flake8 .
   ```

### 代码结构

```python
"""
模块说明

详细的模块描述
"""

import os
import sys
from typing import Optional, List, Dict

from third_party_package import SomeClass

from .local_module import LocalClass


class ExampleClass:
    """类的说明
    
    Args:
        param1: 参数1的说明
        param2: 参数2的说明
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        self.param1 = param1
        self.param2 = param2
    
    def example_method(self) -> bool:
        """方法说明
        
        Returns:
            bool: 返回值说明
        """
        return True
```

## 🧪 测试指南

### 编写测试

1. **测试文件位置**：`tests/` 目录
2. **测试文件命名**：`test_*.py`
3. **测试类命名**：`Test*`
4. **测试函数命名**：`test_*`

### 测试示例

```python
import pytest
from src.your_module import YourClass


class TestYourClass:
    """测试 YourClass 类"""
    
    def setup_method(self):
        """每个测试方法运行前的设置"""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """测试基本功能"""
        result = self.instance.some_method()
        assert result is not None
        assert isinstance(result, str)
    
    def test_error_handling(self):
        """测试错误处理"""
        with pytest.raises(ValueError):
            self.instance.method_that_should_raise_error()
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/test_specific.py

# 运行特定测试
pytest tests/test_specific.py::TestClass::test_method

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

## 📚 文档指南

### 文档类型

1. **代码文档**：使用 docstring
2. **用户文档**：README.md 和 docs/ 目录
3. **API文档**：自动生成

### 文档规范

```python
def example_function(param1: str, param2: int = 0) -> Dict[str, Any]:
    """函数说明的简短描述
    
    详细的函数描述，包括使用场景、注意事项等。
    
    Args:
        param1: 参数1的详细说明
        param2: 参数2的详细说明，默认为0
    
    Returns:
        Dict[str, Any]: 返回值的详细说明
    
    Raises:
        ValueError: 在什么情况下抛出此异常
        TypeError: 在什么情况下抛出此异常
    
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        {'status': 'success', 'value': 5}
    """
    pass
```

## 🔍 Review 指南

### Pull Request 要求

1. **清晰的标题和描述**
2. **关联相关 Issue**
3. **通过所有测试**
4. **代码覆盖率不降低**
5. **遵循代码规范**

### Review 清单

- [ ] 代码逻辑正确
- [ ] 测试覆盖充分
- [ ] 文档更新完整
- [ ] 性能影响评估
- [ ] 安全性考虑
- [ ] 向后兼容性

## 🚀 发布流程

### 版本号规范

使用 [Semantic Versioning](https://semver.org/)：

- `MAJOR.MINOR.PATCH`
- `2.1.0` → `2.1.1` (patch)
- `2.1.0` → `2.2.0` (minor)
- `2.1.0` → `3.0.0` (major)

### 发布步骤

1. 更新版本号
2. 更新 CHANGELOG.md
3. 创建 Release Tag
4. 发布到 PyPI（如适用）

## 💬 社区规范

### 行为准则

我们致力于为每个人提供友好、安全和受欢迎的环境。请：

1. **尊重他人**：不同的观点和经验
2. **建设性反馈**：提供有用的建议
3. **保持专业**：专注于技术讨论
4. **帮助新手**：分享知识和经验

### 沟通渠道

- **GitHub Issues**：Bug报告和功能请求
- **GitHub Discussions**：一般讨论和问题
- **邮件**：私密或紧急问题

## 🙏 致谢

感谢所有贡献者的努力！您的贡献让这个项目变得更好。

---

**再次感谢您的贡献！** 🎉
