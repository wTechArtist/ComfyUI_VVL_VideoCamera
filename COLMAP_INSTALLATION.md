# COLMAP安装指南

## 概述

VVL COLMAP-MVS原生深度估计节点需要安装COLMAP软件。这是一个专业的3D重建软件，提供了最先进的多视图立体（MVS）算法。

## 🚀 快速安装 (推荐)

### 完整安装流程
```bash
# 1. 安装uv (现代Python包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装COLMAP系统依赖
sudo apt install colmap  # Ubuntu/Debian

# 3. 安装Python集成包
uv pip install pycolmap

# 4. 验证安装
colmap --help
python -c "import pycolmap; print('✅ COLMAP集成正常')"
```

## 详细安装方法

### Linux (Ubuntu/Debian) - 推荐

#### 方法1: APT安装 (最简单)
```bash
# 安装COLMAP
sudo apt update
sudo apt install colmap

# 安装Python集成
uv pip install pycolmap  # 推荐，最快
# 或: pip install pycolmap
```

#### 方法2: 从源码编译 (获取最新功能)
```bash
# 安装编译依赖
sudo apt install \
    git cmake build-essential \
    libboost-program-options-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-system-dev libboost-test-dev \
    libeigen3-dev libsuitesparse-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev libgflags-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev

# 克隆并编译
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# 安装Python集成
uv pip install pycolmap
```

### Windows

#### 预编译版本 (推荐)
```bash
# 1. 下载预编译版本
# 访问: https://github.com/colmap/colmap/releases
# 下载: COLMAP-3.x-windows.zip

# 2. 解压并添加到PATH
# 解压到: C:\Program Files\COLMAP
# 添加到PATH: C:\Program Files\COLMAP\bin

# 3. 安装Python集成
# 在PowerShell中安装uv:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv pip install pycolmap
```

### macOS

```bash
# 使用Homebrew安装
brew install colmap

# 安装Python集成
uv pip install pycolmap  # 或: pip install pycolmap
```

## GPU支持配置

### NVIDIA GPU (CUDA) - 强烈推荐

COLMAP可以使用CUDA显著加速MVS操作：

#### Linux CUDA安装
```bash
# 1. 安装NVIDIA驱动
sudo ubuntu-drivers autoinstall

# 2. 安装CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda

# 3. 重新编译COLMAP (如果从源码安装)
cd colmap/build
cmake .. -DCUDA_ENABLED=ON
make -j$(nproc)
sudo make install
```

### 验证GPU支持
```bash
# 检查COLMAP是否支持CUDA
colmap help | grep -i cuda
colmap help | grep -i gpu

# 检查GPU设备
nvidia-smi
```

## 验证安装

### 基础验证
```bash
# 验证COLMAP可执行文件
colmap --help

# 验证版本信息
colmap --version

# 验证Python集成
python -c "
import pycolmap
print(f'✅ PyColmap版本: {pycolmap.__version__}')
"
```

### 功能测试
```bash
# 创建测试脚本
cat > test_colmap.py << 'EOF'
import subprocess
import pycolmap

def test_colmap():
    try:
        # 测试COLMAP可执行文件
        result = subprocess.run(['colmap', '--help'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ COLMAP可执行文件正常")
        else:
            print("❌ COLMAP可执行文件异常")
            
        # 测试GPU支持
        gpu_result = subprocess.run(['colmap', 'help'], 
                                   capture_output=True, text=True)
        if 'gpu' in gpu_result.stdout.lower():
            print("✅ COLMAP支持GPU加速")
        else:
            print("⚠️ COLMAP可能不支持GPU加速")
            
        # 测试Python集成
        print(f"✅ PyColmap版本: {pycolmap.__version__}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_colmap()
EOF

# 运行测试
python test_colmap.py
```

## 常见问题解决

### 1. "colmap: command not found"

**原因**: COLMAP未正确添加到系统PATH

**解决方案**:
```bash
# Linux: 添加到~/.bashrc
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Windows: 手动添加到环境变量PATH
# C:\Program Files\COLMAP\bin
```

### 2. GPU未被检测或使用

**解决方案**:
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA安装
nvcc --version

# 重新编译COLMAP (启用CUDA)
cmake .. -DCUDA_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
```

### 3. PyColmap导入失败

**解决方案**:
```bash
# 重新安装PyColmap
uv pip uninstall pycolmap
uv pip install pycolmap

# 或使用pip
pip install pycolmap --force-reinstall
```

### 4. 内存不足错误

**解决方案**:
- 降低图像分辨率 (resize到1024x768)
- 使用较低的质量设置 (`low`或`medium`)
- 减少输入图像数量 (3-10张)
- 增加系统虚拟内存

## 性能优化建议

### 1. GPU设置
```python
# 在VVL节点中启用GPU
use_gpu = True
gpu_index = 0  # 使用第一块GPU
```

### 2. 质量 vs 速度权衡
| 质量设置 | 处理时间 | 结果质量 | 推荐场景 |
|---------|----------|----------|----------|
| `low` | 快 | 基础 | 快速预览 |
| `medium` | 中等 | 良好 | 平衡选择 |
| `high` | 慢 | 高 | 最终结果 |
| `extreme` | 很慢 | 最高 | 专业用途 |

### 3. 图像建议
- **数量**: 3-20张图像
- **重叠度**: 相邻图像60-80%重叠
- **质量**: 高分辨率、清晰、良好光照
- **格式**: JPEG, PNG, TIFF

## 相关链接

- [COLMAP官网](https://colmap.github.io/)
- [COLMAP安装文档](https://colmap.github.io/install.html)
- [PyColmap文档](https://github.com/colmap/pycolmap)
- [uv包管理器](https://github.com/astral-sh/uv)

## 总结

**推荐安装命令**:
```bash
# 一键安装所有COLMAP相关依赖
curl -LsSf https://astral.sh/uv/install.sh | sh && \
sudo apt install colmap && \
uv pip install pycolmap && \
python -c "import pycolmap; print('✅ 安装完成!')"
```

🎯 **最佳实践**: 使用uv安装Python依赖，享受极速的包管理体验！ 