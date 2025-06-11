# ComfyUI_VVL_VideoCamera 使用示例

## 原始深度数据接口使用指南

### 基础工作流

#### 1. 基本深度估计 + 原始数据访问
```
[输入图像] → [VVL COLMAP-MVS深度估计] → [VVL 深度数据访问器] → [VVL 深度数组转图像]
                      ↓
              [预览图像输出]
```

**节点配置**:
- **VVL COLMAP-MVS深度估计**:
  - `output_format`: "image" (用于预览)
  - `return_raw_data`: True (启用原始数据输出)
  - `quality`: "medium"

- **VVL 深度数据访问器**:
  - `index`: 0 (访问第一个深度图)
  - `output_format`: "numpy_array"

- **VVL 深度数组转图像**:
  - `colormap`: "jet" (彩色深度图)
  - `normalize`: True

#### 2. 批量深度数据处理
```
[输入图像] → [VVL COLMAP-MVS深度估计] → [VVL 深度数据处理器] → [输出结果]
```

**处理操作**:
- `operation`: "statistics" - 获取统计信息
- `operation`: "filter" - 过滤深度范围
- `operation`: "normalize" - 标准化深度值
- `operation`: "export" - 导出为文件

### 高级工作流示例

#### 3. 深度数据分析流水线
```
[输入图像] → [VVL COLMAP-MVS深度估计]
                      ↓ raw_depth_data
                [VVL 深度数据处理器]
                   ↙        ↘
         [统计分析]      [过滤处理]
            ↓              ↓
    [信息输出节点]   [VVL 深度数据访问器]
                            ↓
                  [VVL 深度数组转图像]
                            ↓
                      [可视化输出]
```

#### 4. 多深度图对比工作流
```
[输入图像] → [VVL COLMAP-MVS深度估计] → raw_depth_data
                                           ↓
                                    [分支到多个访问器]
                                    ↙     ↓     ↘
                            [访问器#0] [访问器#1] [访问器#2]
                                ↓       ↓       ↓
                           [转图像#0][转图像#1][转图像#2]
                                ↓       ↓       ↓
                            [对比显示输出]
```

## 具体使用步骤

### 步骤1: 设置深度估计节点
1. 添加 `VVL COLMAP-MVS深度估计` 节点
2. 连接输入图像 (至少3张)
3. 设置参数:
   ```
   output_format: "both"  # 同时输出预览和文件
   return_raw_data: True  # 启用原始数据输出
   quality: "medium"      # 根据需要调整
   save_to_disk: True     # 如需保存文件
   ```

### 步骤2: 连接原始数据处理节点
1. 将深度估计节点的 `raw_depth_data` 输出连接到下游节点
2. 可选的下游节点:
   - **数据访问器**: 获取单个深度图
   - **数据处理器**: 批量分析和处理
   - **自定义节点**: 处理原始numpy数据

### 步骤3: 访问单个深度图
1. 使用 `VVL 深度数据访问器`:
   ```
   raw_depth_data: 连接上游输出
   index: 0  # 第一个深度图
   ```
2. 输出:
   - `depth_image`: 预览图像
   - `depth_info`: 详细信息
   - `numpy_array`: 原始numpy数组

### 步骤4: 可视化处理
1. 使用 `VVL 深度数组转图像`:
   ```
   numpy_array: 来自访问器的输出
   colormap: "jet"      # 颜色映射
   normalize: True      # 自动标准化
   ```

## 数据格式说明

### 原始深度数据结构 (DEPTH_MAPS)
```python
{
    "depth_maps": [numpy_array1, numpy_array2, ...],  # 深度图列表
    "count": 3,                                        # 深度图数量
    "shapes": [(480, 640), (480, 640), ...],          # 每个图的尺寸
    "dtypes": ["float32", "float32", ...],             # 数据类型
    "depth_ranges": [(0.1, 50.0), (0.1, 45.0), ...],  # 深度范围
    "valid_pixel_ratios": [0.85, 0.82, ...],          # 有效像素比例
    "quality": "medium",                               # 质量设置
    "gpu_used": True                                   # 是否使用GPU
}
```

### numpy数组特征
- **数据类型**: `numpy.float32`
- **形状**: `(height, width)`
- **深度值**: 米为单位的真实距离
- **无效值**: 0.0 表示无深度信息
- **精度**: 32位浮点数，保持COLMAP原始精度

## 常见用例

### 1. 深度数据质量检查
```python
# 在自定义节点中
def check_depth_quality(raw_depth_data):
    depth_maps = raw_depth_data["depth_maps"]
    for i, depth_map in enumerate(depth_maps):
        valid_ratio = (depth_map > 0).sum() / depth_map.size
        print(f"深度图{i}: 有效像素 {valid_ratio:.2%}")
        print(f"深度范围: {depth_map.min():.3f} - {depth_map.max():.3f}米")
```

### 2. 深度数据后处理
```python
def process_depth_map(numpy_array):
    # 过滤异常值
    filtered = numpy_array.copy()
    filtered[filtered > 100.0] = 0  # 移除超远距离
    filtered[filtered < 0.1] = 0    # 移除超近距离
    
    # 平滑处理
    from scipy import ndimage
    smoothed = ndimage.gaussian_filter(filtered, sigma=1.0)
    
    return smoothed
```

### 3. 深度图对比分析
```python
def compare_depth_maps(depth_map1, depth_map2):
    # 计算差异
    valid_mask1 = depth_map1 > 0
    valid_mask2 = depth_map2 > 0
    common_mask = valid_mask1 & valid_mask2
    
    if common_mask.sum() > 0:
        diff = np.abs(depth_map1[common_mask] - depth_map2[common_mask])
        return {
            "mean_diff": diff.mean(),
            "std_diff": diff.std(),
            "max_diff": diff.max()
        }
    return None
```

## 故障排除

### 常见问题
1. **"错误：无效的深度数据"**
   - 检查上游深度估计是否成功
   - 确认 `return_raw_data` 设置为 True

2. **"索引超出范围"**
   - 检查深度图数量和访问索引
   - 使用数据处理器查看可用深度图数量

3. **numpy数组为空**
   - 确认深度估计成功完成
   - 检查COLMAP-MVS是否正常工作

### 性能优化
- 对于大量深度图，使用批量处理节点
- 合理设置质量参数平衡速度和精度
- 使用GPU加速提升处理速度

## 扩展开发

### 创建自定义深度处理节点
```python
class CustomDepthProcessorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_depth_data": ("DEPTH_MAPS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "process_custom"
    
    def process_custom(self, raw_depth_data):
        depth_maps = raw_depth_data["depth_maps"]
        # 自定义处理逻辑
        processed_result = your_custom_function(depth_maps)
        return processed_result
```

这个接口设计允许您在ComfyUI工作流中直接访问和处理高精度的深度数据，无需文件读写，实现更高效的深度数据处理流水线。 