# ComfyUI_VVL_VideoCamera

VVL 视频相机参数估计插件，为ComfyUI提供从视频中估计相机参数、深度信息和3D点云重建的完整工作流程。

## 功能特性

### 1. 图片序列相机估计 (ImageSequenceCameraEstimator)
- 使用COLMAP算法从图片序列估计相机内参和外参
- 支持多种匹配策略（sequential、exhaustive、spatial）
- 自动估计焦距、主点和畸变参数
- 输出相机位姿轨迹和3D重建统计信息

### 2. COLMAP-MVS原生深度估计 (VVLColmapMVSDepthNode)
- **使用真正的COLMAP-MVS原生算法进行深度估计**
- 执行完整的COLMAP MVS流水线：
  - 稀疏重建创建
  - 图像去畸变
  - Patch Match Stereo立体匹配
  - 深度图融合
- 支持多种质量设置（low、medium、high、extreme）
- 需要COLMAP软件安装（不是PyColmap）
- GPU加速支持（CUDA）
- 原始深度数据输出接口

### 3. COLMAP点云重建 (VVLColmapPointCloudNode)
- 从相机参数和深度图生成高质量3D点云
- 支持多种融合算法：
  - **simple**: 快速逐图像投影
  - **weighted**: 加权融合平衡质量和速度
  - **tsdf**: Open3D的TSDF体素融合（需Open3D）
  - **colmap_native**: 原生COLMAP稠密重建（最高质量）
- **新增原生COLMAP支持**: 直接使用COLMAP binary进行完整的MVS流水线
- 可选mask支持，过滤不需要的区域
- 支持多种点云格式输出（PLY、PCD、XYZ、JSON）
- Open3D集成，提供高级后处理
- 智能回退机制：如果高级方法失败，自动切换到可用方法

## 安装要求

### 必需依赖
- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy

### COLMAP安装 (强烈推荐)

**VVLColmapMVSDepthNode和原生COLMAP点云重建需要安装完整的COLMAP软件**:

1. **Windows**:
   - 下载: https://github.com/colmap/colmap/releases
   - 解压并添加到PATH环境变量
   
2. **Linux**:
   ```bash
   sudo apt-get install colmap
   ```
   
3. **macOS**:
   ```bash
   brew install colmap
   ```

**注意**: 
- 如果没有安装原生COLMAP，点云节点将自动回退到PyColmap方法
- 原生COLMAP提供最高质量的重建结果
- 详细安装说明请参考 [COLMAP_INSTALLATION.md](COLMAP_INSTALLATION.md)

## 使用方法

### COLMAP-MVS深度估计

1. **在ComfyUI中添加节点**:
   - 节点路径: `💃VVL/VideoCamera` → `VVL COLMAP-MVS原生深度估计`

2. **输入参数**:
   - **images** (必需): 多张输入图像 (至少3张)
   - **intrinsics_json** (必需): 相机内参JSON数据
   - **poses_json** (必需): 相机位姿JSON数据
   - **use_gpu** (可选): 是否使用GPU加速 (默认: True)
   - **quality** (可选): 质量设置 (默认: medium)
     - `low`: 快速处理，较低质量
     - `medium`: 平衡质量和速度
     - `high`: 高质量，较慢处理
     - `extreme`: 最高质量，最慢处理
   - **output_dir** (可选): 输出目录路径
   - **save_to_disk** (可选): 是否保存文件到磁盘 (默认: False)

3. **输出**:
   - **depth_maps**: 深度图序列 (IMAGE格式，用于预览)
   - **file_paths**: 高精度文件路径 (STRING格式)
   - **raw_depth_data**: 原始深度数据 (DEPTH_MAPS格式，用于下游节点)

## 质量设置详细说明

| 设置 | 图像尺寸 | 特征数量 | 处理时间 | 内存使用 |
|------|----------|----------|----------|----------|
| low | 800px | 2000 | 快 | 低 |
| medium | 1200px | 5000 | 中等 | 中等 |
| high | 1600px | 8000 | 慢 | 高 |
| extreme | 2400px | 12000 | 很慢 | 很高 |

## 高精度深度数据

### 输出格式详细说明

| 格式 | 精度 | 用途 | 文件大小 | 读取工具 |
|------|------|------|----------|----------|
| image | 8位归一化 | ComfyUI预览 | 最小 | 标准图像查看器 |
| npy | 32位浮点 | 数值计算分析 | 中等 | NumPy, Python |
| bin | 32位浮点 | COLMAP兼容 | 中等 | COLMAP, 自定义 |
| both | 混合 | 预览+分析 | 最大 | 组合使用 |

### 使用高精度深度数据

1. **读取NPY格式**:
   ```python
   import numpy as np
   depth_map = np.load('depth_map_0001.npy')
   print(f"深度范围: {depth_map.min():.6f} - {depth_map.max():.6f}")
   ```

2. **读取BIN格式**:
   ```python
   from depth_utils import DepthFileHandler
   depth_map = DepthFileHandler.read_colmap_depth_bin('depth_map_0001.bin')
   ```

3. **验证和可视化**:
   ```bash
   python depth_utils.py depth_map_0001.npy --validate --visualize
   ```

4. **在ComfyUI工作流中使用原始深度数据**:
   - 连接深度估计器的 `raw_depth_data` 输出到下游节点
   - 使用 `VVL 深度数据访问器` 获取单个深度图的numpy数组
   - 使用 `VVL 深度数据处理器` 进行批量处理和分析
   - 使用 `VVL 深度数组转图像` 将numpy数组转换为可视化图像

## 注意事项

1. **图像要求**:
   - 至少需要3张图像以确保可靠的深度估计
   - 图像间应有足够的重叠区域
   - 图像质量要好，有丰富的纹理信息

2. **硬件要求**:
   - 推荐使用GPU以获得更快的处理速度
   - 高质量设置需要更多内存和计算时间

3. **输出格式选择**:
   - 仅预览使用: 选择 `image`
   - 高精度分析: 选择 `npy` 或 `bin`
   - 需要兼容COLMAP: 选择 `bin`
   - 同时需要预览和分析: 选择 `both`

4. **失败处理**:
   - 如果COLMAP-MVS失败，节点会抛出详细的错误信息
   - 不会提供任何替代或虚假的深度图结果

## 故障排除

### 常见问题

1. **"未找到可用的COLMAP-MVS实现"**
   - 安装PyColmap: `pip install pycolmap`
   - 或安装命令行COLMAP并确保在PATH中

2. **"COLMAP 3D重建失败"**
   - 检查图像质量和重叠度
   - 尝试降低质量设置
   - 确保至少有3张图像

3. **"所有COLMAP-MVS深度图都无法加载"**
   - 图像间重叠不足
   - 纹理信息不够丰富
   - 尝试调整质量设置或使用更多图像

### 测试安装

运行测试脚本检查安装是否正确:
```bash
cd custom_nodes/ComfyUI_VVL_VideoCamera
python test_import.py
```

## 技术支持

如果遇到问题，请检查:
1. 所有依赖项是否正确安装
2. COLMAP是否可用 (PyColmap或命令行版本)
3. 输入图像是否满足要求
4. 硬件资源是否充足

## 版本历史

- v1.0.0: 初始版本，支持严格的COLMAP-MVS深度估计

## 许可证

本项目遵循相应的开源许可证。 