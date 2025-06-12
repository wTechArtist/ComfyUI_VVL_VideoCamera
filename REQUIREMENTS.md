# VVL VideoCamera 节点依赖说明

## 🚀 快速安装 (推荐uv)

### 一键安装所有依赖
```bash
# 1. 安装uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装系统依赖
sudo apt install colmap  # Ubuntu/Debian

# 3. 安装Python依赖
uv pip install -r requirements.txt open3d

# 4. 验证安装
python check_dependencies.py
```

### 为什么选择uv？
- ⚡ **极速安装**: 比pip快5-10倍
- 🔧 **完全兼容**: 使用相同的PyPI包源
- 🛡️ **依赖解析**: 更智能的冲突处理
- 🚀 **现代化**: Rust编写，性能卓越

## 详细安装步骤

### 1. 安装uv包管理器
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用pip安装
pip install uv
```

### 2. 安装COLMAP（必需）
用于相机标定和点云生成的核心工具。

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install colmap
```

**其他系统:** 请参考 [COLMAP_INSTALLATION.md](COLMAP_INSTALLATION.md)

### 3. 安装Python依赖
```bash
# 安装核心依赖
uv pip install -r requirements.txt

# 安装Open3D (可选，用于高质量TSDF融合)
uv pip install open3d
```

### 4. 验证安装
```bash
# 验证COLMAP
colmap help

# 验证Python依赖
python -c "
import torch, numpy, cv2, PIL, matplotlib, pycolmap
print('✅ 核心依赖正常')
"

# 验证Open3D (如果已安装)
python -c "
import open3d as o3d
print(f'✅ Open3D {o3d.__version__} 正常')
"

# 使用检查脚本
python check_dependencies.py
```

## 可选：其他安装方式

### 使用pip (如果没有uv)
```bash
pip install -r requirements.txt open3d
```

### 使用conda (解决复杂冲突时)
```bash
conda install torch numpy opencv pillow matplotlib pycolmap open3d -c conda-forge
```

## 功能对应表

| 融合方法 | 依赖要求 | 质量 | 速度 | 推荐场景 |
|---------|----------|------|------|----------|
| `tsdf` | Open3D | 高 | 中等 | 高质量点云，推荐 |
| `colmap_native` | COLMAP+CUDA | 最高 | 快 | 最高质量，有GPU |
| `weighted` | 无额外依赖 | 中等 | 中等 | 平衡选择 |
| `simple` | 无额外依赖 | 低 | 快 | 快速预览 |

## 故障排除

### Open3D导入错误
```bash
# 重新安装Open3D
uv pip uninstall open3d
uv pip install open3d

# 如果仍有问题，尝试pip
pip install open3d --force-reinstall
```

### COLMAP相关问题
请参考 [COLMAP_INSTALLATION.md](COLMAP_INSTALLATION.md) 获取详细的安装和故障排除指南。

### 依赖冲突
```bash
# uv提供更好的依赖解析
uv pip install --refresh-package open3d

# 极端情况下使用conda
conda install -c conda-forge open3d --force-reinstall
```

## 性能对比

| 包管理器 | 安装时间 | 依赖解析 | 错误处理 | 推荐度 |
|---------|----------|----------|----------|--------|
| **uv** | ⚡⚡⚡ | 智能 | 清晰 | ⭐⭐⭐⭐⭐ |
| **pip** | ⚡⚡ | 基础 | 一般 | ⭐⭐⭐ |
| **conda** | ⚡ | 全面 | 详细 | ⭐⭐⭐⭐ |

## 总结

**最推荐的安装流程**：
```bash
# 一次性完成所有安装
curl -LsSf https://astral.sh/uv/install.sh | sh && \
sudo apt install colmap && \
uv pip install -r requirements.txt open3d && \
python check_dependencies.py
```

🎯 **为什么选择uv**: 现代、快速、可靠的Python包管理器，是pip的完美升级版！ 