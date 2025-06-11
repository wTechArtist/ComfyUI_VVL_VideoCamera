# COLMAP安装指南

## 概述

VVL COLMAP-MVS原生深度估计节点需要安装COLMAP软件。这是一个专业的3D重建软件，提供了最先进的多视图立体（MVS）算法。

## 安装方法

### Windows

1. **下载预编译版本**（推荐）
   - 访问 [COLMAP Releases](https://github.com/colmap/colmap/releases)
   - 下载最新的 `COLMAP-3.x-windows.zip`
   - 解压到合适的位置，例如 `C:\Program Files\COLMAP`
   - 将 `C:\Program Files\COLMAP\bin` 添加到系统PATH环境变量

2. **验证安装**
   ```cmd
   colmap --help
   ```

### Linux (Ubuntu/Debian)

1. **使用APT安装**（推荐）
   ```bash
   sudo apt-get update
   sudo apt-get install colmap
   ```

2. **从源码编译**（获取最新功能）
   ```bash
   # 安装依赖
   sudo apt-get install \
       git \
       cmake \
       build-essential \
       libboost-program-options-dev \
       libboost-filesystem-dev \
       libboost-graph-dev \
       libboost-system-dev \
       libboost-test-dev \
       libeigen3-dev \
       libsuitesparse-dev \
       libfreeimage-dev \
       libmetis-dev \
       libgoogle-glog-dev \
       libgflags-dev \
       libglew-dev \
       qtbase5-dev \
       libqt5opengl5-dev \
       libcgal-dev \
       libceres-dev

   # 克隆并编译
   git clone https://github.com/colmap/colmap.git
   cd colmap
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   ```

### macOS

1. **使用Homebrew安装**
   ```bash
   brew install colmap
   ```

## GPU支持

### NVIDIA GPU (CUDA)

COLMAP可以使用CUDA加速某些操作：

1. **Windows**: 预编译版本通常包含CUDA支持
2. **Linux**: 
   - 安装CUDA Toolkit
   - 从源码编译时确保CMake检测到CUDA

### 验证GPU支持

```bash
colmap -h | grep gpu
```

## 常见问题

### 1. "colmap: command not found"

**解决方案**: 确保COLMAP的bin目录在系统PATH中

### 2. GPU未被使用

**解决方案**: 
- 检查CUDA是否正确安装
- 使用支持CUDA的COLMAP版本
- 在节点中启用GPU选项

### 3. 内存不足

**解决方案**:
- 降低质量设置（使用"low"或"medium"）
- 减少图像分辨率
- 增加系统内存

## 性能优化建议

1. **使用GPU加速**: 在节点中启用`use_gpu`选项
2. **调整质量设置**: 
   - `low`: 快速预览
   - `medium`: 平衡质量和速度
   - `high`: 高质量结果
   - `extreme`: 最高质量（慢）
3. **图像数量**: 3-20张图像通常效果最好
4. **图像质量**: 使用高质量、清晰的图像

## 验证安装

运行测试脚本：

```bash
python test_colmap_mvs.py
```

如果看到"COLMAP可执行文件找到"，说明安装成功。

## 相关链接

- [COLMAP官网](https://colmap.github.io/)
- [COLMAP文档](https://colmap.github.io/tutorial.html)
- [COLMAP论文](https://demuc.de/papers/schoenberger2016sfm.pdf) 