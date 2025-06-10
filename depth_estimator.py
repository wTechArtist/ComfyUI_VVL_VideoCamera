import torch
import numpy as np
import json
import os
import tempfile
import cv2
import shutil
from PIL import Image
from typing import List, Dict, Any, Tuple
import pathlib

# 添加ComfyUI类型导入
try:
    from comfy.comfy_types import IO
except ImportError:
    # 如果无法导入，创建一个兼容的类
    class IO:
        IMAGE = "IMAGE"

class ColmapMVSDepthEstimator:
    """使用COLMAP-MVS进行密集深度图估计的核心类"""

    def __init__(self):
        self.check_dependencies()

    def check_dependencies(self):
        """检查PyColmap依赖"""
        try:
            import pycolmap
            self.pycolmap = pycolmap
            print("PyColmap MVS 已成功导入")
        except ImportError as e:
            print(f"PyColmap 导入失败: {e}")
            print("请安装 PyColmap: pip install pycolmap")
            raise ImportError("PyColmap 是MVS深度估计的必需依赖")

    def estimate_depth_from_reconstruction(self, 
                                         images: List[torch.Tensor],
                                         reconstruction_result: Dict,
                                         mvs_quality: str = "medium",
                                         output_format: str = "npy",
                                         max_image_size: int = 1200,
                                         patch_match_params: Dict = None) -> Dict:
        """从重建结果生成密集深度图"""
        
        if not reconstruction_result.get("success", False):
            return {
                "success": False,
                "error": "输入的重建结果无效",
                "depth_maps": [],
                "depth_info": {}
            }

        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp(prefix="colmap_mvs_")
        images_dir = os.path.join(temp_dir, "images")
        database_path = os.path.join(temp_dir, "database.db")
        sparse_path = os.path.join(temp_dir, "sparse")
        dense_path = os.path.join(temp_dir, "dense")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(sparse_path, exist_ok=True)
            os.makedirs(dense_path, exist_ok=True)

            # 1. 重新保存图片和重建结果
            print(f"准备 {len(images)} 张图片用于MVS...")
            image_paths = self._save_images_to_temp(images, images_dir)
            
            # 2. 重新执行COLMAP重建（需要完整的重建流程）
            print("重新执行COLMAP重建...")
            reconstructions = self._run_colmap_reconstruction(
                database_path, images_dir, sparse_path, mvs_quality
            )
            
            if not reconstructions:
                return {
                    "success": False,
                    "error": "COLMAP重建失败，无法进行MVS",
                    "depth_maps": [],
                    "depth_info": {}
                }
            
            # 3. 执行MVS密集重建
            print("开始MVS密集重建...")
            self._run_dense_reconstruction(
                images_dir, sparse_path, dense_path, mvs_quality, 
                max_image_size, patch_match_params
            )
            
            # 4. 读取并处理深度图
            print("处理深度图...")
            depth_results = self._process_depth_maps(
                dense_path, images_dir, output_format
            )
            
            return depth_results
            
        except Exception as e:
            print(f"COLMAP MVS 深度估计出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"MVS深度估计失败: {str(e)}",
                "depth_maps": [],
                "depth_info": {}
            }
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def _save_images_to_temp(self, images: List[torch.Tensor], images_dir: str) -> List[str]:
        """将图片张量保存到临时目录"""
        image_paths = []
        
        for i, image_tensor in enumerate(images):
            # 转换张量格式
            if image_tensor.dim() == 4:  # Batch dimension
                image_tensor = image_tensor.squeeze(0)
            
            # 确保是 HWC 格式
            if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:  # CHW -> HWC
                image_tensor = image_tensor.permute(1, 2, 0)
            
            # 转换为numpy数组
            image_np = image_tensor.cpu().numpy()
            
            # 确保值在0-255范围内
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # 保存图片
            image_filename = f"image_{i:06d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # 转换为BGR格式保存（OpenCV格式）
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            image_paths.append(image_path)
        
        return image_paths

    def _run_colmap_reconstruction(self, database_path: str, images_dir: str, 
                                 sparse_path: str, quality: str):
        """运行完整的COLMAP重建流程"""
        try:
            # 设置质量参数
            quality_settings = self._get_quality_settings(quality)
            
            # 特征提取
            sift_options = self.pycolmap.SiftExtractionOptions()
            sift_options.max_image_size = quality_settings["max_image_size"]
            sift_options.max_num_features = quality_settings["max_num_features"]
            
            self.pycolmap.extract_features(
                database_path=database_path,
                image_path=images_dir,
                sift_options=sift_options
            )
            
            # 特征匹配
            matching_options = self.pycolmap.SequentialMatchingOptions()
            matching_options.overlap = 10
            self.pycolmap.match_sequential(
                database_path=database_path,
                matching_options=matching_options
            )
            
            # 增量重建
            try:
                pipeline_options = self.pycolmap.IncrementalPipelineOptions()
                if hasattr(pipeline_options, "mapper_options"):
                    mapper_opts = pipeline_options.mapper_options
                    mapper_opts.ba_refine_focal_length = True
                    mapper_opts.ba_refine_principal_point = True
            except AttributeError:
                pipeline_options = None

            if pipeline_options is not None:
                reconstructions = self.pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=sparse_path,
                    options=pipeline_options
                )
            else:
                reconstructions = self.pycolmap.incremental_mapping(
                    database_path=database_path,
                    image_path=images_dir,
                    output_path=sparse_path
                )
            
            return reconstructions
            
        except Exception as e:
            print(f"COLMAP重建失败: {e}")
            return None

    def _get_quality_settings(self, quality: str) -> Dict:
        """获取质量设置"""
        quality_settings = {
            "low": {"max_image_size": 800, "max_num_features": 4096},
            "medium": {"max_image_size": 1200, "max_num_features": 8192},
            "high": {"max_image_size": 1600, "max_num_features": 16384},
            "extreme": {"max_image_size": 2400, "max_num_features": 32768}
        }
        return quality_settings.get(quality, quality_settings["medium"])

    def _run_dense_reconstruction(self, images_dir: str, sparse_path: str, 
                                dense_path: str, quality: str, max_image_size: int,
                                patch_match_params: Dict = None):
        """执行MVS密集重建"""
        
        try:
            # 步骤1: Image undistortion
            print("1. 图像去畸变...")
            self.pycolmap.undistort_images(
                image_path=images_dir,
                input_path=sparse_path,
                output_path=dense_path,
                output_type="COLMAP"
            )
            
            # 步骤2: Patch match stereo
            print("2. 立体匹配...")
            self.pycolmap.patch_match_stereo(
                workspace_path=dense_path,
                workspace_format="COLMAP",
                pmvs_option_name="option-all",
                config_file_name=""
            )
            
            # 步骤3: Stereo fusion (可选)
            print("3. 立体融合...")
            try:
                self.pycolmap.stereo_fusion(
                    workspace_path=dense_path,
                    workspace_format="COLMAP",
                    input_type="geometric",
                    output_path=os.path.join(dense_path, "fused.ply")
                )
            except Exception as e:
                print(f"立体融合失败，跳过: {e}")
            
        except Exception as e:
            print(f"MVS密集重建过程出错: {e}")
            # 尝试使用备用方法
            try:
                print("尝试使用备用MVS方法...")
                # 只执行patch match
                self.pycolmap.patch_match_stereo(
                    workspace_path=dense_path,
                    workspace_format="COLMAP"
                )
                
            except Exception as e2:
                print(f"备用MVS方法也失败: {e2}")
                raise e

    def _process_depth_maps(self, dense_path: str, images_dir: str, 
                          output_format: str) -> Dict:
        """处理生成的深度图"""
        
        depth_maps = []
        depth_info = {
            "num_depth_maps": 0,
            "depth_range": {"min": float('inf'), "max": float('-inf')},
            "resolution": None,
            "format": output_format
        }
        
        try:
            # 查找深度图文件
            stereo_dir = os.path.join(dense_path, "stereo", "depth_maps")
            if not os.path.exists(stereo_dir):
                # 尝试其他可能的路径
                possible_paths = [
                    os.path.join(dense_path, "depth_maps"),
                    os.path.join(dense_path, "stereo"),
                    dense_path
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        depth_files = [f for f in os.listdir(path) if f.endswith('.geometric.bin')]
                        if depth_files:
                            stereo_dir = path
                            break
            
            if not os.path.exists(stereo_dir):
                return {
                    "success": False,
                    "error": "未找到深度图文件",
                    "depth_maps": [],
                    "depth_info": depth_info
                }
            
            # 读取深度图文件
            depth_files = [f for f in os.listdir(stereo_dir) if f.endswith('.geometric.bin')]
            depth_files.sort()
            
            print(f"找到 {len(depth_files)} 个深度图文件")
            
            for depth_file in depth_files:
                depth_path = os.path.join(stereo_dir, depth_file)
                
                try:
                    # 读取COLMAP深度图
                    depth_map = self._read_colmap_depth_map(depth_path)
                    
                    if depth_map is not None:
                        # 更新深度范围信息
                        valid_depths = depth_map[depth_map > 0]
                        if len(valid_depths) > 0:
                            depth_info["depth_range"]["min"] = min(
                                depth_info["depth_range"]["min"], 
                                float(np.min(valid_depths))
                            )
                            depth_info["depth_range"]["max"] = max(
                                depth_info["depth_range"]["max"], 
                                float(np.max(valid_depths))
                            )
                        
                        if depth_info["resolution"] is None:
                            depth_info["resolution"] = depth_map.shape
                        
                        # 保存深度图
                        if output_format == "npy":
                            depth_data = depth_map.astype(np.float32)
                        elif output_format == "bin":
                            depth_data = depth_map.tobytes()
                        else:
                            depth_data = depth_map.astype(np.float32)
                        
                        depth_maps.append({
                            "data": depth_data,
                            "filename": depth_file.replace('.geometric.bin', f'.{output_format}'),
                            "shape": depth_map.shape,
                            "dtype": str(depth_map.dtype)
                        })
                        
                except Exception as e:
                    print(f"处理深度图 {depth_file} 失败: {e}")
                    continue
            
            depth_info["num_depth_maps"] = len(depth_maps)
            
            return {
                "success": True,
                "depth_maps": depth_maps,
                "depth_info": depth_info
            }
            
        except Exception as e:
            print(f"处理深度图失败: {e}")
            return {
                "success": False,
                "error": f"处理深度图失败: {str(e)}",
                "depth_maps": [],
                "depth_info": depth_info
            }

    def _read_colmap_depth_map(self, depth_path: str) -> np.ndarray:
        """读取COLMAP格式的深度图"""
        try:
            # COLMAP深度图是二进制格式
            with open(depth_path, 'rb') as f:
                # 读取头部信息
                width = int.from_bytes(f.read(4), 'little')
                height = int.from_bytes(f.read(4), 'little')
                channels = int.from_bytes(f.read(4), 'little')
                
                # 读取深度数据
                depth_data = np.frombuffer(f.read(), dtype=np.float32)
                depth_map = depth_data.reshape((height, width, channels))
                
                # 如果是多通道，取第一个通道
                if channels > 1:
                    depth_map = depth_map[:, :, 0]
                
                return depth_map
                
        except Exception as e:
            print(f"读取深度图失败: {e}")
            # 尝试使用PyColmap读取
            try:
                depth_map = self.pycolmap.read_array(depth_path)
                return depth_map
            except:
                return None


class ImageSequenceDepthEstimator:
    """图片序列深度估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "输入图片序列"
                }),
            },
            "optional": {
                "mvs_quality": (["low", "medium", "high", "extreme"], {
                    "default": "medium",
                    "tooltip": "MVS重建质量，higher质量需要更多时间和内存"
                }),
                "output_format": (["npy", "bin"], {
                    "default": "npy",
                    "tooltip": "深度图输出格式"
                }),
                "max_image_size": ("INT", {
                    "default": 1200,
                    "min": 400,
                    "max": 4000,
                    "step": 100,
                    "tooltip": "MVS处理的最大图像尺寸"
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否将深度图保存到磁盘"
                }),
                "output_directory": ("STRING", {
                    "default": "./depth_output",
                    "tooltip": "深度图保存目录"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("depth_maps_info", "depth_statistics", "saved_files_info")
    FUNCTION = "estimate_depth_maps"
    CATEGORY = "💃VVL/VideoCamera"

    def __init__(self):
        try:
            self.estimator = ColmapMVSDepthEstimator()
        except ImportError as e:
            print(f"COLMAP MVS初始化失败: {e}")
            self.estimator = None

    def estimate_depth_maps(self, images,
                          mvs_quality: str = "medium",
                          output_format: str = "npy",
                          max_image_size: int = 1200,
                          save_to_disk: bool = True,
                          output_directory: str = "./depth_output") -> tuple:
        """从图片序列估计深度图的主函数"""
        
        try:
            if self.estimator is None:
                raise RuntimeError("COLMAP MVS初始化失败，请检查PyColmap安装")
            
            # 检查输入
            if images is None:
                raise ValueError("未提供图片输入")
            
            # 转换输入格式
            if isinstance(images, torch.Tensor):
                if images.dim() == 4:  # Batch of images
                    image_list = [images[i] for i in range(images.shape[0])]
                else:  # Single image
                    image_list = [images]
            elif isinstance(images, list):
                image_list = images
            else:
                raise ValueError(f"不支持的图片输入格式: {type(images)}")
            
            print(f"开始MVS深度估计，处理 {len(image_list)} 张图片")
            
            # 执行深度估计
            result = self.estimator.estimate_depth_from_reconstruction(
                images=image_list,
                reconstruction_result={"success": True},  # 简化的重建结果
                mvs_quality=mvs_quality,
                output_format=output_format,
                max_image_size=max_image_size
            )
            
            if not result["success"]:
                raise RuntimeError(f"深度估计失败: {result.get('error', '未知错误')}")
            
            # 准备输出信息
            depth_maps_info = self._format_depth_maps_info(result["depth_maps"], output_format)
            depth_statistics = self._format_depth_statistics(result["depth_info"])
            
            # 保存文件到磁盘
            saved_files_info = ""
            if save_to_disk and result["depth_maps"]:
                saved_files_info = self._save_depth_maps_to_disk(
                    result["depth_maps"], output_directory, output_format
                )
            else:
                saved_files_info = json.dumps({
                    "saved": False,
                    "reason": "save_to_disk为False或无深度图数据"
                }, ensure_ascii=False, indent=2)
            
            print(f"成功生成 {len(result['depth_maps'])} 个深度图")
            
            return (depth_maps_info, depth_statistics, saved_files_info)
            
        except Exception as e:
            error_msg = f"深度图估计出错: {str(e)}"
            print(error_msg)
            
            # 返回错误信息
            error_json = json.dumps({"error": error_msg, "success": False}, ensure_ascii=False, indent=2)
            
            return (error_json, error_json, error_json)

    def _format_depth_maps_info(self, depth_maps: List[Dict], output_format: str) -> str:
        """格式化深度图信息"""
        info = {
            "num_depth_maps": len(depth_maps),
            "output_format": output_format,
            "depth_maps": []
        }
        
        for i, depth_map in enumerate(depth_maps):
            map_info = {
                "index": i,
                "filename": depth_map["filename"],
                "shape": depth_map["shape"],
                "dtype": depth_map["dtype"],
                "data_size_bytes": len(depth_map["data"]) if isinstance(depth_map["data"], bytes) else depth_map["data"].nbytes
            }
            info["depth_maps"].append(map_info)
        
        return json.dumps(info, ensure_ascii=False, indent=2)

    def _format_depth_statistics(self, depth_info: Dict) -> str:
        """格式化深度统计信息"""
        return json.dumps(depth_info, ensure_ascii=False, indent=2)

    def _save_depth_maps_to_disk(self, depth_maps: List[Dict], 
                                output_dir: str, output_format: str) -> str:
        """保存深度图到磁盘"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            saved_files = []
            
            for depth_map in depth_maps:
                filename = depth_map["filename"]
                filepath = os.path.join(output_dir, filename)
                
                if output_format == "npy":
                    np.save(filepath, depth_map["data"])
                elif output_format == "bin":
                    with open(filepath, 'wb') as f:
                        if isinstance(depth_map["data"], bytes):
                            f.write(depth_map["data"])
                        else:
                            f.write(depth_map["data"].tobytes())
                
                saved_files.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size_bytes": os.path.getsize(filepath)
                })
            
            result = {
                "saved": True,
                "output_directory": output_dir,
                "num_files": len(saved_files),
                "files": saved_files,
                "total_size_mb": sum(f["size_bytes"] for f in saved_files) / (1024 * 1024)
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_result = {
                "saved": False,
                "error": str(e),
                "output_directory": output_dir
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageSequenceDepthEstimator": ImageSequenceDepthEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceDepthEstimator": "VVL Image Sequence Depth Estimator (COLMAP-MVS)"
} 