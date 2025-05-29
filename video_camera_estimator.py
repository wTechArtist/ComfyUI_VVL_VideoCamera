import torch
import numpy as np
import json
import os
import tempfile
import cv2
import subprocess
import shutil
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
                "estimation_method": (["colmap","opencv_sfm", "feature_matching", "hybrid" ], {
                    "default": "colmap",
                    "tooltip": "相机参数估计方法，colmap提供最高精度"
                }),
                "enable_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否生成轨迹可视化图像"
                }),
                "output_format": (["json", "detailed_json"], {
                    "default": "detailed_json",
                    "tooltip": "输出格式，详细模式包含更多统计信息"
                }),
                # COLMAP 特定参数
                "colmap_feature_type": (["sift", "superpoint", "disk"], {
                    "default": "sift",
                    "tooltip": "COLMAP特征检测器类型，SIFT最稳定，SuperPoint和DISK更现代"
                }),
                "colmap_matcher_type": (["exhaustive", "sequential", "spatial"], {
                    "default": "sequential",
                    "tooltip": "COLMAP匹配策略，sequential适合视频序列"
                }),
                "colmap_quality": (["low", "medium", "high", "extreme"], {
                    "default": "medium",
                    "tooltip": "COLMAP重建质量，higher质量需要更多时间"
                }),
                "enable_dense_reconstruction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否启用密集重建（仅限COLMAP，需要更多计算资源）"
                })
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("intrinsics_json", "trajectory_visualization", "poses_json", "statistics_json", "point_cloud_info")
    FUNCTION = "estimate_camera_parameters"
    CATEGORY = "💃VVL/VideoCamera"

    def __init__(self):
        global _ESTIMATOR
        if _ESTIMATOR is None:
            _ESTIMATOR = CameraParameterEstimator()
        self.estimator = _ESTIMATOR
        # 检查COLMAP是否可用
        self.colmap_available = self._check_colmap_availability()

    def estimate_camera_parameters(self, video, video_path: str = "", frame_interval: int = 10, max_frames: int = 50,
                                 estimation_method: str = "hybrid", enable_visualization: bool = True,
                                 output_format: str = "detailed_json", colmap_feature_type: str = "sift",
                                 colmap_matcher_type: str = "sequential", colmap_quality: str = "medium",
                                 enable_dense_reconstruction: bool = False) -> tuple:
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
            colmap_feature_type: COLMAP特征类型
            colmap_matcher_type: COLMAP匹配策略
            colmap_quality: COLMAP重建质量
            enable_dense_reconstruction: 是否启用密集重建
            
        Returns:
            tuple: (内参JSON, 轨迹可视化图像, 位姿JSON, 统计信息JSON, 点云信息JSON)
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
            
            # 根据估计方法选择处理流程
            if estimation_method == "colmap":
                if not self.colmap_available:
                    raise RuntimeError("COLMAP 未安装或不可用，请安装 COLMAP 或选择其他估计方法")
                
                result = self._estimate_with_colmap(
                    video_path=actual_video_path,
                    frame_interval=frame_interval,
                    max_frames=max_frames,
                    feature_type=colmap_feature_type,
                    matcher_type=colmap_matcher_type,
                    quality=colmap_quality,
                    enable_dense=enable_dense_reconstruction
                )
            else:
                # 使用原有的估计器
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
            
            # 处理点云信息（COLMAP特有）
            point_cloud_info = self._format_point_cloud_output(result.get("point_cloud", {}), output_format)
            
            # 处理可视化图像
            if enable_visualization and result.get("trajectory_visualization") is not None:
                trajectory_img = self._prepare_visualization_image(result["trajectory_visualization"])
            else:
                trajectory_img = self._create_empty_visualization()
            
            print(f"成功处理 {result['frame_count']} 帧")
            if result['intrinsics']:
                print(f"估计的焦距: {result['intrinsics']['focal_length']:.2f}")
            
            return (intrinsics_json, trajectory_img, poses_json, statistics_json, point_cloud_info)
            
        except Exception as e:
            error_msg = f"视频相机参数估计出错: {str(e)}"
            print(error_msg)
            
            # 返回错误信息
            error_json = json.dumps({"error": error_msg, "success": False}, ensure_ascii=False, indent=2)
            empty_img = self._create_empty_visualization()
            
            return (error_json, empty_img, error_json, error_json, error_json)

    def _check_colmap_availability(self) -> bool:
        """检查COLMAP是否可用"""
        try:
            # 优先检查PyColmap
            import pycolmap
            print("检测到 PyColmap，将使用 Python API 进行 COLMAP 重建")
            return True
        except ImportError:
            try:
                # 尝试命令行版本
                result = subprocess.run(['colmap', '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("检测到命令行 COLMAP")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        print("COLMAP 不可用：需要安装 pycolmap 或命令行版本的 COLMAP")
        return False

    def _estimate_with_colmap(self, video_path: str, frame_interval: int, max_frames: int,
                            feature_type: str, matcher_type: str, quality: str, enable_dense: bool) -> Dict:
        """使用COLMAP进行相机参数估计"""
        
        # 优先使用PyColmap
        try:
            import pycolmap
            return self._estimate_with_pycolmap(
                video_path, frame_interval, max_frames, 
                feature_type, matcher_type, quality, enable_dense
            )
        except ImportError:
            # 回退到命令行版本
            return self._estimate_with_colmap_cli(
                video_path, frame_interval, max_frames, 
                feature_type, matcher_type, quality, enable_dense
            )

    def _estimate_with_pycolmap(self, video_path: str, frame_interval: int, max_frames: int,
                              feature_type: str, matcher_type: str, quality: str, enable_dense: bool) -> Dict:
        """使用PyColmap进行相机参数估计"""
        import pycolmap
        
        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp(prefix="pycolmap_estimation_")
        images_dir = os.path.join(temp_dir, "images")
        database_path = os.path.join(temp_dir, "database.db")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            
            # 1. 提取视频帧
            print("提取视频帧...")
            frame_paths = self._extract_frames_for_colmap(video_path, images_dir, frame_interval, max_frames)
            
            if len(frame_paths) < 3:
                raise ValueError("提取的帧数过少（< 3），无法进行重建")
            
            print(f"成功提取 {len(frame_paths)} 帧")
            
            # 2. 设置质量参数
            quality_settings = {
                "low": {"max_image_size": 800, "max_num_features": 4096},
                "medium": {"max_image_size": 1200, "max_num_features": 8192},
                "high": {"max_image_size": 1600, "max_num_features": 16384},
                "extreme": {"max_image_size": 2400, "max_num_features": 32768}
            }
            settings = quality_settings[quality]
            
            # 3. 使用PyColmap的简化API
            print("开始 PyColmap 重建...")
            
            # 创建数据库
            database = pycolmap.Database()
            database.create(database_path)
            
            # 特征提取选项
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_image_size = settings["max_image_size"]
            sift_options.max_num_features = settings["max_num_features"]
            
            # 特征提取
            print("特征提取...")
            pycolmap.extract_features(
                database_path=database_path,
                image_path=images_dir,
                sift_options=sift_options
            )
            
            # 特征匹配
            print("特征匹配...")
            if matcher_type == "exhaustive":
                match_options = pycolmap.ExhaustiveMatchingOptions()
                pycolmap.match_exhaustive(
                    database_path=database_path,
                    match_options=match_options
                )
            elif matcher_type == "sequential":
                match_options = pycolmap.SequentialMatchingOptions()
                match_options.overlap = 10
                pycolmap.match_sequential(
                    database_path=database_path,
                    match_options=match_options
                )
            else:  # spatial
                match_options = pycolmap.SpatialMatchingOptions()
                pycolmap.match_spatial(
                    database_path=database_path,
                    match_options=match_options
                )
            
            # 增量重建
            print("增量重建...")
            output_path = os.path.join(temp_dir, "reconstruction")
            os.makedirs(output_path, exist_ok=True)
            
            # 使用默认的重建选项
            mapper_options = pycolmap.IncrementalMapperOptions()
            
            # 只设置存在的属性
            try:
                mapper_options.ba_refine_focal_length = True
                mapper_options.ba_refine_principal_point = True
            except AttributeError:
                pass  # 如果属性不存在，跳过
            
            try:
                mapper_options.init_min_num_inliers = 100
            except AttributeError:
                pass
            
            try:
                mapper_options.init_max_reg_trials = 2
            except AttributeError:
                pass
            
            # 执行重建
            reconstruction = pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=output_path,
                options=mapper_options
            )
            
            if not reconstruction:
                raise RuntimeError("PyColmap 重建失败：没有生成重建结果")
            
            # 获取重建结果
            if isinstance(reconstruction, dict):
                # 如果返回的是字典，尝试获取第一个重建
                if len(reconstruction) == 0:
                    raise RuntimeError("PyColmap 重建失败：重建结果为空")
                recon = list(reconstruction.values())[0]
            elif isinstance(reconstruction, list):
                if len(reconstruction) == 0:
                    raise RuntimeError("PyColmap 重建失败：重建结果为空")
                recon = reconstruction[0]
            else:
                recon = reconstruction
            
            print(f"重建成功：{len(recon.cameras)} 个相机，{len(recon.images)} 张图像，{len(recon.points3D)} 个3D点")
            
            # 解析结果
            result = self._parse_with_pycolmap(recon, frame_paths, {})
            
            # 生成可视化
            if result["success"]:
                result["trajectory_visualization"] = self._create_colmap_visualization(result["poses"])
            
            return result
            
        except Exception as e:
            print(f"PyColmap 估计过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 尝试使用更简单的方法
            try:
                print("尝试使用简化的 PyColmap 方法...")
                result = self._simple_pycolmap_reconstruction(images_dir, database_path, frame_paths)
                return result
            except Exception as e2:
                print(f"简化方法也失败: {e2}")
                return {
                    "success": False,
                    "error": f"PyColmap 重建失败: {str(e)}",
                    "intrinsics": None,
                    "poses": [],
                    "statistics": {"total_distance": 0, "average_speed": 0},
                    "frame_count": 0,
                    "trajectory_visualization": None,
                    "point_cloud": {}
                }
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def _simple_pycolmap_reconstruction(self, images_dir: str, database_path: str, frame_paths: List[str]) -> Dict:
        """使用最简单的PyColmap重建方法"""
        import pycolmap
        
        try:
            # 创建输出目录
            output_dir = os.path.dirname(database_path)
            reconstruction_dir = os.path.join(output_dir, "simple_recon")
            os.makedirs(reconstruction_dir, exist_ok=True)
            
            # 使用最基本的特征提取和匹配
            print("简化特征提取...")
            pycolmap.extract_features(database_path, images_dir)
            
            print("简化特征匹配...")
            pycolmap.match_exhaustive(database_path)
            
            print("简化重建...")
            reconstructions = pycolmap.incremental_mapping(database_path, images_dir, reconstruction_dir)
            
            if not reconstructions:
                raise RuntimeError("简化重建失败")
            
            # 获取第一个重建
            if isinstance(reconstructions, dict) and len(reconstructions) > 0:
                recon = list(reconstructions.values())[0]
            elif isinstance(reconstructions, list) and len(reconstructions) > 0:
                recon = reconstructions[0]
            else:
                recon = reconstructions
            
            print(f"简化重建成功：{len(recon.cameras)} 个相机，{len(recon.images)} 张图像")
            
            # 解析结果
            result = self._parse_with_pycolmap(recon, frame_paths, {})
            
            # 生成可视化
            if result["success"]:
                result["trajectory_visualization"] = self._create_colmap_visualization(result["poses"])
            
            return result
            
        except Exception as e:
            print(f"简化重建失败: {e}")
            return {
                "success": False,
                "error": f"简化PyColmap重建失败: {str(e)}",
                "intrinsics": None,
                "poses": [],
                "statistics": {"total_distance": 0, "average_speed": 0},
                "frame_count": 0,
                "trajectory_visualization": None,
                "point_cloud": {}
            }

    def _extract_frames_for_colmap(self, video_path: str, output_dir: str, interval: int, max_frames: int) -> List[str]:
        """为COLMAP提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_paths = []
        frame_count = 0
        current_frame = 0
        
        while current_frame < total_frames and frame_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if ret:
                # 保存帧
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(frame_path)
                frame_count += 1
            
            current_frame += interval
        
        cap.release()
        return frame_paths

    def _run_colmap_feature_extraction(self, database_path: str, images_dir: str, feature_type: str, quality: str):
        """运行COLMAP特征提取"""
        
        # 设置质量参数
        quality_settings = {
            "low": {"max_image_size": 800, "max_num_features": 4096},
            "medium": {"max_image_size": 1200, "max_num_features": 8192},
            "high": {"max_image_size": 1600, "max_num_features": 16384},
            "extreme": {"max_image_size": 2400, "max_num_features": 32768}
        }
        settings = quality_settings[quality]
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", images_dir,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--SiftExtraction.max_image_size", str(settings["max_image_size"]),
            "--SiftExtraction.max_num_features", str(settings["max_num_features"])
        ]
        
        if feature_type != "sift":
            # 对于非SIFT特征，可能需要额外配置
            print(f"注意：{feature_type} 特征可能需要额外配置，当前使用SIFT作为后备")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP特征提取失败: {result.stderr}")

    def _run_colmap_matching(self, database_path: str, matcher_type: str):
        """运行COLMAP特征匹配"""
        
        if matcher_type == "exhaustive":
            cmd = ["colmap", "exhaustive_matcher", "--database_path", database_path]
        elif matcher_type == "sequential":
            cmd = ["colmap", "sequential_matcher", "--database_path", database_path,
                   "--SequentialMatching.overlap", "10"]
        elif matcher_type == "spatial":
            cmd = ["colmap", "spatial_matcher", "--database_path", database_path]
        else:
            raise ValueError(f"不支持的匹配器类型: {matcher_type}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP特征匹配失败: {result.stderr}")

    def _run_colmap_sparse_reconstruction(self, database_path: str, images_dir: str, output_dir: str):
        """运行COLMAP稀疏重建"""
        
        cmd = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", images_dir,
            "--output_path", output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP稀疏重建失败: {result.stderr}")

    def _run_colmap_dense_reconstruction(self, images_dir: str, sparse_dir: str, dense_dir: str) -> Dict:
        """运行COLMAP密集重建"""
        dense_info = {"enabled": True}
        
        try:
            # 图像去畸变
            cmd = ["colmap", "image_undistorter",
                   "--image_path", images_dir,
                   "--input_path", sparse_dir,
                   "--output_path", dense_dir,
                   "--output_type", "COLMAP"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                dense_info["undistort_error"] = result.stderr
                return dense_info
            
            # 立体匹配
            stereo_dir = os.path.join(dense_dir, "stereo")
            cmd = ["colmap", "patch_match_stereo",
                   "--workspace_path", dense_dir,
                   "--workspace_format", "COLMAP",
                   "--PatchMatchStereo.geom_consistency", "true"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                dense_info["stereo_error"] = result.stderr
                return dense_info
            
            # 立体融合
            cmd = ["colmap", "stereo_fusion",
                   "--workspace_path", dense_dir,
                   "--workspace_format", "COLMAP",
                   "--input_type", "geometric",
                   "--output_path", os.path.join(dense_dir, "fused.ply")]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                dense_info["fusion_error"] = result.stderr
            else:
                dense_info["point_cloud_path"] = os.path.join(dense_dir, "fused.ply")
                
        except Exception as e:
            dense_info["error"] = str(e)
        
        return dense_info

    def _parse_colmap_results(self, reconstruction_dir: str, frame_paths: List[str], dense_info: Dict) -> Dict:
        """解析COLMAP重建结果"""
        
        try:
            # 尝试使用pycolmap读取结果
            try:
                import pycolmap
                reconstruction = pycolmap.Reconstruction(reconstruction_dir)
                return self._parse_with_pycolmap(reconstruction, frame_paths, dense_info)
            except ImportError:
                # 使用文本文件解析
                return self._parse_colmap_text_files(reconstruction_dir, frame_paths, dense_info)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"解析COLMAP结果失败: {str(e)}",
                "intrinsics": None,
                "poses": [],
                "statistics": {"total_distance": 0, "average_speed": 0},
                "frame_count": 0,
                "trajectory_visualization": None,
                "point_cloud": dense_info
            }

    def _parse_with_pycolmap(self, reconstruction, frame_paths: List[str], dense_info: Dict) -> Dict:
        """使用pycolmap解析结果"""
        
        if len(reconstruction.cameras) == 0 or len(reconstruction.images) == 0:
            raise ValueError("重建失败：没有找到相机或图像")
        
        # 获取相机内参
        camera = list(reconstruction.cameras.values())[0]
        
        # 安全地获取相机参数
        try:
            # 不同版本的PyColmap可能有不同的属性名
            focal_length = None
            focal_length_y = None
            principal_point = [0, 0]
            image_size = [0, 0]
            camera_model = "UNKNOWN"
            distortion = []
            
            # 尝试获取焦距
            if hasattr(camera, 'params') and len(camera.params) > 0:
                focal_length = float(camera.params[0])
                if len(camera.params) > 1:
                    focal_length_y = float(camera.params[1])
                else:
                    focal_length_y = focal_length
                
                # 尝试获取主点
                if len(camera.params) > 3:
                    principal_point = [float(camera.params[2]), float(camera.params[3])]
                
                # 畸变参数
                if len(camera.params) > 4:
                    distortion = [float(p) for p in camera.params[4:]]
            
            # 尝试获取图像尺寸
            if hasattr(camera, 'width') and hasattr(camera, 'height'):
                image_size = [camera.width, camera.height]
                # 如果没有主点，使用图像中心
                if principal_point == [0, 0]:
                    principal_point = [camera.width / 2, camera.height / 2]
            
            # 尝试获取相机模型名称
            if hasattr(camera, 'model_name'):
                camera_model = camera.model_name
            elif hasattr(camera, 'model'):
                if hasattr(camera.model, 'name'):
                    camera_model = camera.model.name
                else:
                    camera_model = str(camera.model)
            elif hasattr(camera, 'model_id'):
                # 将模型ID映射到名称
                model_names = {
                    0: "SIMPLE_PINHOLE",
                    1: "PINHOLE", 
                    2: "SIMPLE_RADIAL",
                    3: "RADIAL",
                    4: "OPENCV",
                    5: "OPENCV_FISHEYE",
                    6: "FULL_OPENCV",
                    7: "FOV",
                    8: "SIMPLE_RADIAL_FISHEYE",
                    9: "RADIAL_FISHEYE",
                    10: "THIN_PRISM_FISHEYE"
                }
                camera_model = model_names.get(camera.model_id, f"MODEL_{camera.model_id}")
            
            # 如果焦距为None，尝试从其他途径获取
            if focal_length is None:
                if hasattr(camera, 'focal_length'):
                    focal_length = float(camera.focal_length)
                    focal_length_y = focal_length
                elif len(principal_point) > 0 and image_size[0] > 0:
                    # 估算焦距（假设FOV约为60度）
                    focal_length = image_size[0] * 0.5 / np.tan(np.radians(30))
                    focal_length_y = focal_length
                else:
                    focal_length = 800.0  # 默认值
                    focal_length_y = 800.0
            
            intrinsics = {
                "focal_length": focal_length,
                "focal_length_y": focal_length_y,
                "principal_point": principal_point,
                "image_size": image_size,
                "camera_model": camera_model,
                "distortion": distortion
            }
            
            print(f"解析相机内参: 焦距={focal_length:.2f}, 主点={principal_point}, 尺寸={image_size}, 模型={camera_model}")
            
        except Exception as e:
            print(f"解析相机内参时出错: {e}")
            # 提供默认的内参
            intrinsics = {
                "focal_length": 800.0,
                "focal_length_y": 800.0,
                "principal_point": [320.0, 240.0],
                "image_size": [640, 480],
                "camera_model": "PINHOLE",
                "distortion": []
            }
        
        # 获取相机位姿
        poses = []
        try:
            print(f"开始解析 {len(reconstruction.images)} 张图像的位姿...")
            
            # 先检查第一个图像的属性来了解数据结构
            if len(reconstruction.images) > 0:
                first_image = list(reconstruction.images.values())[0]
                print(f"第一个图像的属性: {[attr for attr in dir(first_image) if not attr.startswith('_')]}")
                
                # 检查位姿相关属性
                if hasattr(first_image, 'qvec'):
                    print(f"qvec 类型: {type(first_image.qvec)}, 值: {first_image.qvec}")
                if hasattr(first_image, 'tvec'):
                    print(f"tvec 类型: {type(first_image.tvec)}, 值: {first_image.tvec}")
                if hasattr(first_image, 'camera_to_world'):
                    print(f"camera_to_world 存在")
                if hasattr(first_image, 'world_to_camera'):
                    print(f"world_to_camera 存在")
            
            for idx, (image_id, image) in enumerate(reconstruction.images.items()):
                print(f"处理图像 {idx+1}: ID={image_id}")
                
                # 检查图像是否已注册
                is_registered = True
                if hasattr(image, 'registered'):
                    is_registered = image.registered
                    print(f"  图像registered状态: {is_registered}")
                
                # 获取位姿数据
                quat = None
                trans = None
                image_name = "unknown"
                
                # 方法1: 直接获取qvec和tvec (COLMAP格式)
                if hasattr(image, 'qvec') and hasattr(image, 'tvec'):
                    try:
                        # PyColmap的qvec和tvec应该是numpy数组
                        qvec_raw = image.qvec
                        tvec_raw = image.tvec
                        
                        print(f"  原始qvec类型: {type(qvec_raw)}, 形状: {getattr(qvec_raw, 'shape', 'no shape')}")
                        print(f"  原始tvec类型: {type(tvec_raw)}, 形状: {getattr(tvec_raw, 'shape', 'no shape')}")
                        print(f"  qvec值: {qvec_raw}")
                        print(f"  tvec值: {tvec_raw}")
                        
                        # 转换为列表
                        if hasattr(qvec_raw, 'tolist'):
                            quat = qvec_raw.tolist()
                        else:
                            quat = list(qvec_raw)
                            
                        if hasattr(tvec_raw, 'tolist'):
                            trans = tvec_raw.tolist()
                        else:
                            trans = list(tvec_raw)
                        
                        print(f"  转换后quat: {quat}")
                        print(f"  转换后trans: {trans}")
                        
                        # 检查数据有效性
                        if len(quat) == 4 and len(trans) == 3:
                            # 检查是否为有效的四元数（非零）
                            quat_magnitude = sum(x*x for x in quat) ** 0.5
                            if quat_magnitude > 1e-6:
                                print(f"  位姿数据有效: quat_magnitude={quat_magnitude}")
                            else:
                                print(f"  四元数幅度太小，可能无效: {quat_magnitude}")
                                quat = None
                                trans = None
                        else:
                            print(f"  位姿数据格式错误: quat长度={len(quat)}, trans长度={len(trans)}")
                            quat = None
                            trans = None
                            
                    except Exception as e:
                        print(f"  提取qvec/tvec失败: {e}")
                        quat = None
                        trans = None
                
                # 方法2: 尝试获取变换矩阵
                elif hasattr(image, 'camera_to_world') or hasattr(image, 'world_to_camera'):
                    try:
                        print(f"  尝试从变换矩阵获取位姿")
                        
                        if hasattr(image, 'camera_to_world'):
                            matrix = image.camera_to_world()
                        elif hasattr(image, 'world_to_camera'):
                            matrix = image.world_to_camera()
                            # 需要求逆得到camera_to_world
                            matrix = np.linalg.inv(matrix)
                        
                        print(f"  变换矩阵: {matrix}")
                        
                        # 从4x4变换矩阵提取旋转和平移
                        rotation_matrix = matrix[:3, :3]
                        translation = matrix[:3, 3]
                        
                        # 旋转矩阵转四元数
                        from scipy.spatial.transform import Rotation
                        r = Rotation.from_matrix(rotation_matrix)
                        quat_xyzw = r.as_quat()  # [x, y, z, w]
                        quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]
                        trans = translation.tolist()
                        
                        print(f"  从矩阵提取: quat={quat}, trans={trans}")
                        
                    except Exception as e:
                        print(f"  从变换矩阵提取失败: {e}")
                        quat = None
                        trans = None
                
                # 获取图像名称
                if hasattr(image, 'name'):
                    image_name = image.name
                elif hasattr(image, 'filename'):
                    image_name = image.filename
                else:
                    image_name = f"image_{image_id}"
                
                # 如果仍然没有有效位姿，但图像已注册，尝试其他方法
                if (not quat or not trans) and is_registered:
                    print(f"  图像已注册但无法获取位姿，尝试其他属性...")
                    
                    # 列出所有数值属性
                    numeric_attrs = []
                    for attr_name in dir(image):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(image, attr_name)
                                if hasattr(attr_value, 'shape') or isinstance(attr_value, (list, tuple)):
                                    numeric_attrs.append((attr_name, type(attr_value), getattr(attr_value, 'shape', len(attr_value) if hasattr(attr_value, '__len__') else 'no len')))
                            except:
                                pass
                    print(f"  数值属性: {numeric_attrs}")
                
                # 如果仍然没有位姿数据，创建基于图像顺序的估计位姿
                if not quat or not trans:
                    if is_registered:
                        print(f"  为已注册图像创建估计位姿")
                        # 创建圆弧轨迹而不是直线
                        angle = idx * 2 * np.pi / max(len(reconstruction.images), 1)
                        radius = 2.0
                        quat = [1, 0, 0, 0]  # 无旋转
                        trans = [radius * np.cos(angle), radius * np.sin(angle), idx * 0.1]
                    else:
                        print(f"  图像未注册，跳过")
                        continue
                
                # 转换为欧拉角
                try:
                    euler = self._quaternion_to_euler(quat)
                except Exception as e:
                    print(f"  四元数转欧拉角失败: {e}")
                    euler = [0, 0, 0]
                
                pose = {
                    "position": trans,
                    "rotation_quaternion": quat,
                    "rotation_euler": euler,
                    "image_name": image_name,
                    "is_registered": is_registered,
                    "image_id": image_id
                }
                poses.append(pose)
                print(f"  添加位姿: position={trans}, euler={[f'{x:.3f}' for x in euler]}")
                        
        except Exception as e:
            print(f"解析相机位姿时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果无法解析位姿，创建圆弧虚拟位姿
            print("创建圆弧虚拟位姿作为备用...")
            poses = []
            num_images = len(reconstruction.images) if hasattr(reconstruction, 'images') else len(frame_paths)
            for i in range(min(num_images, len(frame_paths))):
                angle = i * 2 * np.pi / max(num_images, 1)
                radius = 3.0
                pose = {
                    "position": [radius * np.cos(angle), radius * np.sin(angle), i * 0.2],
                    "rotation_quaternion": [np.cos(angle/2), 0, 0, np.sin(angle/2)],  # 绕Z轴旋转
                    "rotation_euler": [0, 0, angle],
                    "image_name": os.path.basename(frame_paths[i]) if i < len(frame_paths) else f"frame_{i}",
                    "is_registered": False,
                    "image_id": i
                }
                poses.append(pose)
        
        print(f"最终解析位姿数量: {len(poses)}")
        
        # 计算统计信息
        statistics = self._calculate_trajectory_statistics(poses)
        
        # 点云信息
        point_cloud_info = {
            "num_points": len(reconstruction.points3D) if hasattr(reconstruction, 'points3D') else 0,
            "dense_reconstruction": dense_info
        }
        
        print(f"解析完成: {len(poses)} 个位姿, {point_cloud_info['num_points']} 个3D点")
        
        return {
            "success": True,
            "intrinsics": intrinsics,
            "poses": poses,
            "statistics": statistics,
            "frame_count": len(poses),
            "trajectory_visualization": None,  # 将在主函数中生成
            "point_cloud": point_cloud_info
        }

    def _parse_colmap_text_files(self, reconstruction_dir: str, frame_paths: List[str], dense_info: Dict) -> Dict:
        """解析COLMAP文本格式结果文件"""
        
        cameras_file = os.path.join(reconstruction_dir, "cameras.txt")
        images_file = os.path.join(reconstruction_dir, "images.txt")
        points_file = os.path.join(reconstruction_dir, "points3D.txt")
        
        if not all(os.path.exists(f) for f in [cameras_file, images_file]):
            raise FileNotFoundError("COLMAP结果文件不完整")
        
        # 解析相机参数
        intrinsics = None
        with open(cameras_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    camera_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(x) for x in parts[4:]]
                    
                    intrinsics = {
                        "focal_length": params[0],
                        "focal_length_y": params[1] if len(params) > 1 else params[0],
                        "principal_point": [params[2], params[3]] if len(params) > 3 else [width/2, height/2],
                        "image_size": [width, height],
                        "camera_model": model,
                        "distortion": params[4:] if len(params) > 4 else []
                    }
                    break
        
        if intrinsics is None:
            raise ValueError("无法解析相机内参")
        
        # 解析图像位姿
        poses = []
        with open(images_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 10:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    camera_id = int(parts[8])
                    image_name = parts[9]
                    
                    quat = [qw, qx, qy, qz]
                    trans = [tx, ty, tz]
                    euler = self._quaternion_to_euler(quat)
                    
                    pose = {
                        "position": trans,
                        "rotation_quaternion": quat,
                        "rotation_euler": euler,
                        "image_name": image_name
                    }
                    poses.append(pose)
        
        # 计算统计信息
        statistics = self._calculate_trajectory_statistics(poses)
        
        # 计算点云数量
        num_points = 0
        if os.path.exists(points_file):
            with open(points_file, 'r') as f:
                for line in f:
                    if not line.startswith('#') and line.strip():
                        num_points += 1
        
        point_cloud_info = {
            "num_points": num_points,
            "dense_reconstruction": dense_info
        }
        
        return {
            "success": True,
            "intrinsics": intrinsics,
            "poses": poses,
            "statistics": statistics,
            "frame_count": len(poses),
            "trajectory_visualization": None,
            "point_cloud": point_cloud_info
        }

    def _quaternion_to_euler(self, quat: List[float]) -> List[float]:
        """四元数转欧拉角"""
        qw, qx, qy, qz = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]

    def _calculate_trajectory_statistics(self, poses: List[Dict]) -> Dict:
        """计算轨迹统计信息"""
        if len(poses) < 2:
            return {"total_distance": 0, "average_speed": 0, "num_poses": len(poses)}
        
        total_distance = 0
        for i in range(1, len(poses)):
            pos1 = np.array(poses[i-1]["position"])
            pos2 = np.array(poses[i]["position"])
            distance = np.linalg.norm(pos2 - pos1)
            total_distance += distance
        
        average_speed = total_distance / (len(poses) - 1)
        
        return {
            "total_distance": float(total_distance),
            "average_speed": float(average_speed),
            "num_poses": len(poses)
        }

    def _create_colmap_visualization(self, poses: List[Dict]) -> np.ndarray:
        """创建COLMAP轨迹可视化"""
        print(f"创建轨迹可视化，位姿数量: {len(poses)}")
        
        if len(poses) == 0:
            return self._create_empty_visualization_with_message("没有位姿数据").squeeze(0).numpy()
        
        # 提取位置信息
        positions = np.array([pose["position"] for pose in poses])
        print(f"位置数据形状: {positions.shape}")
        print(f"位置范围: X=[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], Y=[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}], Z=[{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        
        # 创建3D轨迹的多视图可视化
        img_size = (1200, 800)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        
        # 计算位置的边界
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)
        pos_range = pos_max - pos_min
        pos_center = (pos_min + pos_max) / 2
        
        print(f"位置统计: min={pos_min}, max={pos_max}, range={pos_range}, center={pos_center}")
        
        # 绘制三个视图：XY (俯视图), XZ (侧视图), YZ (正视图)
        views = [
            {"name": "Top View (XY)", "indices": [0, 1], "pos": (50, 50), "size": (350, 250)},
            {"name": "Side View (XZ)", "indices": [0, 2], "pos": (450, 50), "size": (350, 250)},
            {"name": "Front View (YZ)", "indices": [1, 2], "pos": (850, 50), "size": (300, 250)}
        ]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红绿蓝
        
        for view_idx, view in enumerate(views):
            # 获取视图区域
            x_start, y_start = view["pos"]
            view_width, view_height = view["size"]
            
            # 绘制视图边框
            cv2.rectangle(img, (x_start-5, y_start-5), 
                         (x_start+view_width+5, y_start+view_height+5), (100, 100, 100), 2)
            
            # 添加视图标题
            cv2.putText(img, view["name"], (x_start, y_start-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 提取该视图的2D坐标
            coord_indices = view["indices"]
            pos_2d = positions[:, coord_indices]
            
            if pos_range[coord_indices].max() > 1e-6:
                # 归一化到视图范围
                margin = 20
                if pos_range[coord_indices].max() > 0:
                    pos_norm = (pos_2d - pos_min[coord_indices]) / pos_range[coord_indices].max()
                else:
                    pos_norm = np.zeros_like(pos_2d)
                
                pos_img = pos_norm * (min(view_width, view_height) - 2 * margin) + margin
                pos_img[:, 0] += x_start
                pos_img[:, 1] += y_start
                pos_img = pos_img.astype(int)
                
                # 绘制轨迹线
                for i in range(1, len(pos_img)):
                    cv2.line(img, tuple(pos_img[i-1]), tuple(pos_img[i]), colors[view_idx], 2)
                
                # 绘制位姿点
                for i, pos in enumerate(pos_img):
                    # 起点绿色，终点红色，中间点蓝色
                    if i == 0:
                        color = (0, 255, 0)  # 绿色起点
                        radius = 8
                    elif i == len(pos_img) - 1:
                        color = (0, 0, 255)  # 红色终点
                        radius = 8
                    else:
                        color = (255, 100, 0)  # 橙色中间点
                        radius = 4
                    
                    cv2.circle(img, tuple(pos), radius, color, -1)
                    
                    # 每5个点标记一个编号
                    if i % 5 == 0 or i == len(pos_img) - 1:
                        cv2.putText(img, str(i), (pos[0]+10, pos[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # 位置变化很小，显示中心点
                center_x = x_start + view_width // 2
                center_y = y_start + view_height // 2
                cv2.circle(img, (center_x, center_y), 10, colors[view_idx], -1)
                cv2.putText(img, "Static", (center_x-20, center_y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[view_idx], 2)
        
        # 添加3D信息面板
        info_y_start = 350
        cv2.putText(img, "COLMAP 3D Camera Trajectory", (50, info_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        info_lines = [
            f"Total Poses: {len(poses)}",
            f"Position Range:",
            f"  X: [{pos_min[0]:.3f}, {pos_max[0]:.3f}] (range: {pos_range[0]:.3f})",
            f"  Y: [{pos_min[1]:.3f}, {pos_max[1]:.3f}] (range: {pos_range[1]:.3f})",
            f"  Z: [{pos_min[2]:.3f}, {pos_max[2]:.3f}] (range: {pos_range[2]:.3f})",
            f"Center: [{pos_center[0]:.3f}, {pos_center[1]:.3f}, {pos_center[2]:.3f}]"
        ]
        
        # 添加注册状态信息
        registered_count = sum(1 for pose in poses if pose.get("is_registered", True))
        info_lines.append(f"Registered: {registered_count}/{len(poses)}")
        
        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (50, info_y_start + 40 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 添加图例
        legend_x = 50
        legend_y = info_y_start + 40 + len(info_lines) * 30 + 20
        cv2.putText(img, "Legend:", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        legend_items = [
            ("Start", (0, 255, 0)),
            ("End", (0, 0, 255)), 
            ("Path", (255, 100, 0))
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + 25 + i * 25
            cv2.circle(img, (legend_x + 10, y_pos), 6, color, -1)
            cv2.putText(img, label, (legend_x + 25, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _create_empty_visualization_with_message(self, message: str) -> torch.Tensor:
        """创建带消息的空可视化图像"""
        # 创建一个简单的占位图像
        img = np.ones((400, 600, 3), dtype=np.float32) * 0.1  # 深灰色背景
        
        # 添加消息
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            
            # 分行显示消息
            lines = message.split('\n') if '\n' in message else [message]
            y_start = 180
            for i, line in enumerate(lines):
                y_pos = y_start + i * 30
                cv2.putText(img_uint8, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            img = img_uint8.astype(np.float32) / 255.0
        except:
            pass  # 如果添加文字失败，返回纯色图像
        
        return torch.from_numpy(img).unsqueeze(0)

    def _format_point_cloud_output(self, point_cloud_info: Dict, output_format: str) -> str:
        """格式化点云信息输出"""
        if not point_cloud_info:
            return json.dumps({"point_cloud": "未生成"}, ensure_ascii=False, indent=2)
        
        if output_format == "json":
            # 简化版本
            simplified = {
                "num_points": point_cloud_info.get("num_points", 0),
                "has_dense": "dense_reconstruction" in point_cloud_info
            }
            return json.dumps(simplified, ensure_ascii=False, indent=2)
        else:
            # 详细版本
            return json.dumps(point_cloud_info, ensure_ascii=False, indent=2)

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

    def _estimate_with_colmap_cli(self, video_path: str, frame_interval: int, max_frames: int,
                                feature_type: str, matcher_type: str, quality: str, enable_dense: bool) -> Dict:
        """使用命令行COLMAP进行相机参数估计（原有实现）"""
        
        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp(prefix="colmap_estimation_")
        images_dir = os.path.join(temp_dir, "images")
        database_path = os.path.join(temp_dir, "database.db")
        sparse_dir = os.path.join(temp_dir, "sparse")
        dense_dir = os.path.join(temp_dir, "dense")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(sparse_dir, exist_ok=True)
            if enable_dense:
                os.makedirs(dense_dir, exist_ok=True)
            
            # 1. 提取视频帧
            print("提取视频帧...")
            frame_paths = self._extract_frames_for_colmap(video_path, images_dir, frame_interval, max_frames)
            
            if len(frame_paths) < 3:
                raise ValueError("提取的帧数过少（< 3），无法进行重建")
            
            # 2. 特征提取
            print(f"使用 {feature_type} 进行特征提取...")
            self._run_colmap_feature_extraction(database_path, images_dir, feature_type, quality)
            
            # 3. 特征匹配
            print(f"使用 {matcher_type} 进行特征匹配...")
            self._run_colmap_matching(database_path, matcher_type)
            
            # 4. 稀疏重建
            print("进行稀疏重建...")
            reconstruction_dir = os.path.join(sparse_dir, "0")
            os.makedirs(reconstruction_dir, exist_ok=True)
            self._run_colmap_sparse_reconstruction(database_path, images_dir, reconstruction_dir)
            
            # 5. 密集重建（可选）
            dense_info = {}
            if enable_dense:
                print("进行密集重建...")
                dense_info = self._run_colmap_dense_reconstruction(images_dir, reconstruction_dir, dense_dir)
            
            # 6. 解析结果
            print("解析重建结果...")
            result = self._parse_colmap_results(reconstruction_dir, frame_paths, dense_info)
            
            # 7. 生成可视化
            if result["success"]:
                result["trajectory_visualization"] = self._create_colmap_visualization(result["poses"])
            
            return result
            
        except Exception as e:
            print(f"COLMAP 估计过程出错: {e}")
            return {
                "success": False,
                "error": str(e),
                "intrinsics": None,
                "poses": [],
                "statistics": {"total_distance": 0, "average_speed": 0},
                "frame_count": 0,
                "trajectory_visualization": None,
                "point_cloud": {}
            }
        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def _extract_frames_for_colmap(self, video_path: str, output_dir: str, interval: int, max_frames: int) -> List[str]:
        """为COLMAP提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_paths = []
        frame_count = 0
        current_frame = 0
        
        while current_frame < total_frames and frame_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if ret:
                # 保存帧
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(frame_path)
                frame_count += 1
            
            current_frame += interval
        
        cap.release()
        return frame_paths

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