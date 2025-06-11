#!/usr/bin/env python3
"""
深度图文件处理工具
支持读取和验证NPY、BIN格式的深度数据
"""

import numpy as np
import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import cv2

class DepthFileHandler:
    """深度图文件处理器"""
    
    @staticmethod
    def read_npy_depth(filepath: str) -> Optional[np.ndarray]:
        """
        读取NPY格式深度图
        
        Args:
            filepath: NPY文件路径
            
        Returns:
            深度图数组，失败返回None
        """
        try:
            depth_map = np.load(filepath)
            print(f"读取NPY深度图: {filepath}")
            print(f"  形状: {depth_map.shape}")
            print(f"  数据类型: {depth_map.dtype}")
            print(f"  深度范围: {depth_map.min():.6f} - {depth_map.max():.6f}")
            print(f"  有效像素比例: {(depth_map > 0).sum() / depth_map.size:.2%}")
            return depth_map
        except Exception as e:
            print(f"读取NPY深度图失败: {e}")
            return None
    
    @staticmethod
    def read_colmap_depth_bin(filepath: str) -> Optional[np.ndarray]:
        """
        读取COLMAP BIN格式深度图
        
        Args:
            filepath: BIN文件路径
            
        Returns:
            深度图数组，失败返回None
        """
        try:
            with open(filepath, 'rb') as f:
                # 读取宽度和高度
                width = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                height = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                
                # 读取深度数据
                depth_data = np.frombuffer(f.read(), dtype=np.float32)
                
                if len(depth_data) != width * height:
                    print(f"BIN文件数据大小不匹配: 期望{width*height}, 实际{len(depth_data)}")
                    return None
                
                depth_map = depth_data.reshape(height, width)
                
                print(f"读取BIN深度图: {filepath}")
                print(f"  形状: {depth_map.shape}")
                print(f"  数据类型: {depth_map.dtype}")
                print(f"  深度范围: {depth_map.min():.6f} - {depth_map.max():.6f}")
                print(f"  有效像素比例: {(depth_map > 0).sum() / depth_map.size:.2%}")
                
                return depth_map
                
        except Exception as e:
            print(f"读取BIN深度图失败: {e}")
            return None
    
    @staticmethod
    def validate_depth_map(depth_map: np.ndarray) -> Dict[str, Any]:
        """
        验证深度图数据质量
        
        Args:
            depth_map: 深度图数组
            
        Returns:
            验证结果字典
        """
        if depth_map is None or depth_map.size == 0:
            return {"valid": False, "reason": "空数据"}
        
        result = {
            "valid": True,
            "shape": depth_map.shape,
            "dtype": str(depth_map.dtype),
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
            "std_depth": float(depth_map.std()),
        }
        
        # 有效深度像素分析
        valid_pixels = depth_map > 0
        result["total_pixels"] = depth_map.size
        result["valid_pixels"] = int(valid_pixels.sum())
        result["valid_ratio"] = float(result["valid_pixels"] / result["total_pixels"])
        
        # 质量检查
        issues = []
        
        if result["valid_ratio"] < 0.01:
            issues.append("有效像素比例过低 (<1%)")
        
        if result["min_depth"] <= 0 and result["valid_pixels"] > 0:
            issues.append("包含无效深度值")
        
        if result["max_depth"] > 1000:
            issues.append("最大深度值异常大 (>1000)")
        
        if result["min_depth"] >= result["max_depth"]:
            issues.append("深度值范围无效")
        
        result["issues"] = issues
        result["quality"] = "良好" if not issues else "需要检查"
        
        return result
    
    @staticmethod
    def visualize_depth_map(depth_map: np.ndarray, 
                          title: str = "深度图", 
                          save_path: Optional[str] = None) -> None:
        """
        可视化深度图
        
        Args:
            depth_map: 深度图数组
            title: 图表标题
            save_path: 保存路径 (可选)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # 原始深度图
        im1 = axes[0, 0].imshow(depth_map, cmap='jet')
        axes[0, 0].set_title('原始深度图')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 有效深度区域
        valid_mask = depth_map > 0
        masked_depth = np.where(valid_mask, depth_map, np.nan)
        im2 = axes[0, 1].imshow(masked_depth, cmap='jet')
        axes[0, 1].set_title('有效深度区域')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 深度值分布直方图
        valid_depths = depth_map[valid_mask]
        if len(valid_depths) > 0:
            axes[1, 0].hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('深度值分布')
            axes[1, 0].set_xlabel('深度值')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 统计信息文本
        validation = DepthFileHandler.validate_depth_map(depth_map)
        stats_text = f"""
        形状: {validation['shape']}
        数据类型: {validation['dtype']}
        深度范围: {validation['min_depth']:.4f} - {validation['max_depth']:.4f}
        平均深度: {validation['mean_depth']:.4f}
        标准差: {validation['std_depth']:.4f}
        有效像素: {validation['valid_pixels']} / {validation['total_pixels']}
        有效比例: {validation['valid_ratio']:.2%}
        质量评估: {validation['quality']}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化图表已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def convert_depth_format(input_path: str, output_path: str, 
                           input_format: str = "auto", 
                           output_format: str = "npy") -> bool:
        """
        转换深度图格式
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            input_format: 输入格式 ("auto", "npy", "bin")
            output_format: 输出格式 ("npy", "bin", "png")
            
        Returns:
            转换是否成功
        """
        # 自动检测输入格式
        if input_format == "auto":
            ext = os.path.splitext(input_path)[1].lower()
            if ext == ".npy":
                input_format = "npy"
            elif ext == ".bin":
                input_format = "bin"
            else:
                print(f"无法识别的文件格式: {ext}")
                return False
        
        # 读取深度图
        if input_format == "npy":
            depth_map = DepthFileHandler.read_npy_depth(input_path)
        elif input_format == "bin":
            depth_map = DepthFileHandler.read_colmap_depth_bin(input_path)
        else:
            print(f"不支持的输入格式: {input_format}")
            return False
        
        if depth_map is None:
            return False
        
        # 保存为指定格式
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_format == "npy":
                np.save(output_path, depth_map.astype(np.float32))
            elif output_format == "bin":
                DepthFileHandler._save_colmap_depth_bin(depth_map, output_path)
            elif output_format == "png":
                # 归一化并保存为图像
                normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(output_path, normalized)
            else:
                print(f"不支持的输出格式: {output_format}")
                return False
            
            print(f"格式转换成功: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            print(f"格式转换失败: {e}")
            return False
    
    @staticmethod
    def _save_colmap_depth_bin(depth_map: np.ndarray, filepath: str):
        """保存为COLMAP二进制深度图格式"""
        height, width = depth_map.shape
        
        with open(filepath, 'wb') as f:
            # 写入宽度和高度 (4字节无符号整数)
            f.write(np.array([width], dtype=np.uint32).tobytes())
            f.write(np.array([height], dtype=np.uint32).tobytes())
            
            # 写入深度数据 (4字节浮点数)
            depth_data = depth_map.astype(np.float32).flatten()
            f.write(depth_data.tobytes())


def main():
    """命令行工具主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='深度图文件处理工具')
    parser.add_argument('input', help='输入文件路径')
    parser.add_argument('--validate', action='store_true', help='验证深度图')
    parser.add_argument('--visualize', action='store_true', help='可视化深度图')
    parser.add_argument('--convert', help='转换格式 (npy, bin, png)')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return
    
    # 读取深度图
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".npy":
        depth_map = DepthFileHandler.read_npy_depth(args.input)
    elif ext == ".bin":
        depth_map = DepthFileHandler.read_colmap_depth_bin(args.input)
    else:
        print(f"不支持的文件格式: {ext}")
        return
    
    if depth_map is None:
        print("读取深度图失败")
        return
    
    # 验证
    if args.validate:
        validation = DepthFileHandler.validate_depth_map(depth_map)
        print(f"\n深度图验证结果:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
    
    # 可视化
    if args.visualize:
        save_path = args.output if args.output else None
        DepthFileHandler.visualize_depth_map(depth_map, 
                                           title=f"深度图: {os.path.basename(args.input)}", 
                                           save_path=save_path)
    
    # 格式转换
    if args.convert:
        if not args.output:
            base_name = os.path.splitext(args.input)[0]
            args.output = f"{base_name}_converted.{args.convert}"
        
        success = DepthFileHandler.convert_depth_format(
            args.input, args.output, output_format=args.convert
        )
        
        if success:
            print(f"转换成功: {args.output}")
        else:
            print("转换失败")


if __name__ == "__main__":
    main() 