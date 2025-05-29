import torch
import numpy as np
import json
import os
import tempfile
import cv2
from PIL import Image
from typing import List, Dict, Any

# 添加ComfyUI类型导入
try:
    from comfy.comfy_types import IO
except ImportError:
    # 如果无法导入，创建一个兼容的类
    class IO:
        VIDEO = "VIDEO"

# 修复导入路径
try:
    from .utils.camera_utils import CameraParameterEstimator
except ImportError:
    try:
        # 尝试相对导入
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        utils_path = os.path.join(current_dir, 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        from camera_utils import CameraParameterEstimator
    except ImportError:
        # 最后的备用方案
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        try:
            from utils.camera_utils import CameraParameterEstimator
        except ImportError as e:
            print(f"无法导入 CameraParameterEstimator: {e}")
            # 创建一个虚拟类，避免启动失败
            class CameraParameterEstimator:
                def __init__(self):
                    self.error = "CameraParameterEstimator 导入失败"
                
                def estimate_from_video(self, *args, **kwargs):
                    return {
                        "success": False,
                        "error": self.error,
                        "intrinsics": None,
                        "poses": [],
                        "statistics": None,
                        "frame_count": 0,
                        "trajectory_visualization": None
                    }

# 全局估计器实例
_ESTIMATOR = None

class VideoCameraEstimator:
    """视频相机参数估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {
                    "tooltip": "从LoadVideo节点输入的视频，或者直接输入视频文件路径"
                }),
            },
            "optional": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "视频文件路径（可选，如果video输入为空则使用此路径）"
                }),
                "frame_interval": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "提取帧的间隔，数值越小提取的帧越多"
                }),
                "max_frames": ("INT", {
                    "default": 50,
                    "min": 5,
                    "max": 200,
                    "step": 5,
                    "tooltip": "最大提取帧数，用于控制计算量"
                }),
                "estimation_method": (["opencv_sfm", "feature_matching", "hybrid"], {
                    "default": "hybrid",
                    "tooltip": "相机参数估计方法"
                }),
                "enable_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否生成轨迹可视化图像"
                }),
                "output_format": (["json", "detailed_json"], {
                    "default": "detailed_json",
                    "tooltip": "输出格式，详细模式包含更多统计信息"
                })
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("intrinsics_json", "trajectory_visualization", "poses_json", "statistics_json")
    FUNCTION = "estimate_camera_parameters"
    CATEGORY = "💃VVL/VideoCamera"

    def __init__(self):
        global _ESTIMATOR
        if _ESTIMATOR is None:
            _ESTIMATOR = CameraParameterEstimator()
        self.estimator = _ESTIMATOR

    def estimate_camera_parameters(self, video, video_path: str = "", frame_interval: int = 10, max_frames: int = 50,
                                 estimation_method: str = "hybrid", enable_visualization: bool = True,
                                 output_format: str = "detailed_json") -> tuple:
        """
        从视频估计相机参数的主函数
        
        Args:
            video: 来自LoadVideo节点的视频对象或None
            video_path: 视频文件路径（备用）
            frame_interval: 帧提取间隔
            max_frames: 最大帧数
            estimation_method: 估计方法
            enable_visualization: 是否生成可视化
            output_format: 输出格式
            
        Returns:
            tuple: (内参JSON, 轨迹可视化图像, 位姿JSON, 统计信息JSON)
        """
        
        try:
            # 确定视频文件路径
            actual_video_path = None
            
            # 优先使用video输入（来自LoadVideo节点）
            if video is not None:
                # 如果video是VideoFromFile对象，获取其文件路径
                if hasattr(video, '_VideoFromFile__file'):
                    # 访问私有属性 __file
                    file_attr = video._VideoFromFile__file
                    if isinstance(file_attr, str):
                        actual_video_path = file_attr
                    else:
                        print(f"video.__file 不是字符串类型: {type(file_attr)}")
                elif hasattr(video, 'path'):
                    actual_video_path = video.path
                elif hasattr(video, 'video_path'):
                    actual_video_path = video.video_path
                elif hasattr(video, '_path'):
                    actual_video_path = video._path
                elif hasattr(video, 'file_path'):
                    actual_video_path = video.file_path
                else:
                    # 尝试直接使用video作为路径
                    if isinstance(video, str):
                        actual_video_path = video
                    else:
                        print(f"无法从video对象获取路径，video类型: {type(video)}")
                        print(f"video对象属性: {[attr for attr in dir(video) if not attr.startswith('_')]}")
                        if hasattr(video, '__dict__'):
                            print(f"video对象内容: {video.__dict__}")
                        
                        # 尝试调用get_components来看是否能获取信息
                        try:
                            if hasattr(video, 'get_components'):
                                components = video.get_components()
                                print(f"video components: {components}")
                        except Exception as e:
                            print(f"调用get_components失败: {e}")
            
            # 如果video输入无效，使用video_path
            if actual_video_path is None or not actual_video_path:
                if video_path and video_path.strip():
                    actual_video_path = video_path.strip()
                else:
                    raise ValueError("未提供有效的视频输入。请连接LoadVideo节点到video输入，或在video_path中输入文件路径。")
            
            # 验证视频文件
            if not os.path.exists(actual_video_path):
                raise FileNotFoundError(f"视频文件不存在: {actual_video_path}")
            
            # 验证视频格式
            if not self._is_valid_video_format(actual_video_path):
                raise ValueError("不支持的视频格式")
            
            print(f"开始处理视频: {actual_video_path}")
            print(f"参数 - 帧间隔: {frame_interval}, 最大帧数: {max_frames}, 方法: {estimation_method}")
            
            # 使用估计器处理视频
            result = self.estimator.estimate_from_video(
                video_path=actual_video_path,
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            
            if not result["success"]:
                raise RuntimeError(f"相机参数估计失败: {result.get('error', '未知错误')}")
            
            # 准备输出
            intrinsics_json = self._format_intrinsics_output(result["intrinsics"], output_format)
            poses_json = self._format_poses_output(result["poses"], output_format)
            statistics_json = self._format_statistics_output(result["statistics"], result, output_format)
            
            # 处理可视化图像
            if enable_visualization and result.get("trajectory_visualization") is not None:
                trajectory_img = self._prepare_visualization_image(result["trajectory_visualization"])
            else:
                trajectory_img = self._create_empty_visualization()
            
            print(f"成功处理 {result['frame_count']} 帧")
            print(f"估计的焦距: {result['intrinsics']['focal_length']:.2f}")
            
            return (intrinsics_json, trajectory_img, poses_json, statistics_json)
            
        except Exception as e:
            error_msg = f"视频相机参数估计出错: {str(e)}"
            print(error_msg)
            
            # 返回错误信息
            error_json = json.dumps({"error": error_msg, "success": False}, ensure_ascii=False, indent=2)
            empty_img = self._create_empty_visualization()
            
            return (error_json, empty_img, error_json, error_json)

    def _is_valid_video_format(self, video_path: str) -> bool:
        """验证视频格式"""
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        ext = os.path.splitext(video_path)[1].lower()
        return ext in valid_extensions

    def _format_intrinsics_output(self, intrinsics: Dict, output_format: str) -> str:
        """格式化内参输出"""
        if output_format == "json":
            # 简化版本，只包含核心参数
            simplified = {
                "focal_length": intrinsics["focal_length"],
                "principal_point": intrinsics["principal_point"],
                "image_size": intrinsics["image_size"]
            }
            return json.dumps(simplified, ensure_ascii=False, indent=2)
        else:
            # 详细版本
            return json.dumps(intrinsics, ensure_ascii=False, indent=2)

    def _format_poses_output(self, poses: List[Dict], output_format: str) -> str:
        """格式化位姿输出"""
        if output_format == "json":
            # 简化版本，只包含位置和欧拉角
            simplified_poses = []
            for i, pose in enumerate(poses):
                simplified_poses.append({
                    "frame": i,
                    "position": pose["position"],
                    "rotation_euler": pose["rotation_euler"]
                })
            return json.dumps(simplified_poses, ensure_ascii=False, indent=2)
        else:
            # 详细版本
            detailed_poses = []
            for i, pose in enumerate(poses):
                detailed_pose = pose.copy()
                detailed_pose["frame"] = i
                detailed_poses.append(detailed_pose)
            return json.dumps(detailed_poses, ensure_ascii=False, indent=2)

    def _format_statistics_output(self, statistics: Dict, result: Dict, output_format: str) -> str:
        """格式化统计信息输出"""
        if output_format == "json":
            # 简化版本
            simplified_stats = {
                "frame_count": result["frame_count"],
                "total_distance": statistics["total_distance"],
                "success": result["success"]
            }
            return json.dumps(simplified_stats, ensure_ascii=False, indent=2)
        else:
            # 详细版本
            detailed_stats = statistics.copy()
            detailed_stats.update({
                "frame_count": result["frame_count"],
                "success": result["success"],
                "processing_info": {
                    "total_poses": len(result["poses"]),
                    "has_visualization": result.get("trajectory_visualization") is not None
                }
            })
            return json.dumps(detailed_stats, ensure_ascii=False, indent=2)

    def _prepare_visualization_image(self, trajectory_data) -> torch.Tensor:
        """准备可视化图像"""
        try:
            if isinstance(trajectory_data, list):
                # 从列表转换为numpy数组
                img_array = np.array(trajectory_data, dtype=np.uint8)
            else:
                img_array = trajectory_data
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # BGR转RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                # 转换为torch tensor (H, W, C) -> (1, H, W, C)
                img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)
                return img_tensor.unsqueeze(0)
            else:
                return self._create_empty_visualization()
                
        except Exception as e:
            print(f"可视化图像处理错误: {e}")
            return self._create_empty_visualization()

    def _create_empty_visualization(self) -> torch.Tensor:
        """创建空的可视化图像"""
        # 创建一个简单的占位图像
        img = np.ones((400, 400, 3), dtype=np.float32) * 0.1  # 深灰色背景
        
        # 添加文字提示
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.putText(img_uint8, "No Trajectory", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(img_uint8, "Visualization", (110, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            img = img_uint8.astype(np.float32) / 255.0
        except:
            pass  # 如果添加文字失败，返回纯色图像
        
        return torch.from_numpy(img).unsqueeze(0)

# 第二个节点：视频帧提取器（辅助节点）
class VideoFrameExtractor:
    """视频帧提取节点（辅助功能）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "输入视频文件路径"
                }),
                "frame_indices": ("STRING", {
                    "default": "0,10,20,30",
                    "tooltip": "要提取的帧索引，用逗号分隔"
                })
            },
            "optional": {
                "resize_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "调整图像宽度，0表示保持原尺寸"
                }),
                "resize_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "tooltip": "调整图像高度，0表示保持原尺寸"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("extracted_frames", "frame_info")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "extract_frames"
    CATEGORY = "💃VVL/VideoCamera"

    def extract_frames(self, video_path: str, frame_indices: str, resize_width: int = 0, resize_height: int = 0) -> tuple:
        """提取指定的视频帧"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 解析帧索引
            indices = [int(x.strip()) for x in frame_indices.split(',') if x.strip()]
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            extracted_frames = []
            extracted_info = []
            
            for idx in indices:
                if 0 <= idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # BGR转RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 调整尺寸
                        if resize_width > 0 and resize_height > 0:
                            frame = cv2.resize(frame, (resize_width, resize_height))
                        
                        # 转换为torch tensor
                        frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
                        extracted_frames.append(frame_tensor)
                        
                        extracted_info.append({
                            "frame_index": idx,
                            "shape": list(frame.shape)
                        })
            
            cap.release()
            
            info_json = json.dumps({
                "total_extracted": len(extracted_frames),
                "video_total_frames": total_frames,
                "frame_details": extracted_info
            }, ensure_ascii=False, indent=2)
            
            return (extracted_frames, info_json)
            
        except Exception as e:
            error_msg = f"帧提取错误: {str(e)}"
            print(error_msg)
            
            # 返回空图像和错误信息
            empty_frame = torch.zeros((400, 400, 3), dtype=torch.float32)
            error_json = json.dumps({"error": error_msg}, ensure_ascii=False, indent=2)
            
            return ([empty_frame], error_json)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoCameraEstimator": VideoCameraEstimator,
    "VideoFrameExtractor": VideoFrameExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCameraEstimator": "VVL Video Camera Estimator",
    "VideoFrameExtractor": "VVL Video Frame Extractor"
} 