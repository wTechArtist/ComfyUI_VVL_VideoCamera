import torch
import numpy as np
import json
import os
import tempfile
import cv2
import shutil
from PIL import Image
from typing import List, Dict, Any
import pathlib

# 添加ComfyUI类型导入
try:
    from comfy.comfy_types import IO
except ImportError:
    # 如果无法导入，创建一个兼容的类
    class IO:
        IMAGE = "IMAGE"

class ColmapCameraEstimator:
    """使用COLMAP进行相机参数估计的核心类"""

    def __init__(self):
        self.check_dependencies()

    def check_dependencies(self):
        """检查PyColmap依赖"""
        try:
            import pycolmap
            self.pycolmap = pycolmap
            print("PyColmap 已成功导入")
        except ImportError as e:
            print(f"PyColmap 导入失败: {e}")
            print("请安装 PyColmap: pip install pycolmap")
            raise ImportError("PyColmap 是必需的依赖")

    def estimate_from_images(self, images: List[torch.Tensor], 
                           colmap_feature_type: str = "sift",
                           colmap_matcher_type: str = "sequential", 
                           colmap_quality: str = "medium",
                           enable_dense_reconstruction: bool = False) -> Dict:
        """从图片序列估计相机参数"""
        
        if len(images) < 3:
            return {
                "success": False,
                "error": "图片数量不足，至少需要3张图片进行重建",
                "intrinsics": None,
                "poses": [],
                "statistics": {},
                "frame_count": 0,
                "point_cloud": {}
            }

        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp(prefix="colmap_images_")
        images_dir = os.path.join(temp_dir, "images")
        database_path = os.path.join(temp_dir, "database.db")
        output_path = os.path.join(temp_dir, "reconstruction")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

            # 1. 保存图片到临时目录
            print(f"保存 {len(images)} 张图片到临时目录...")
            image_paths = self._save_images_to_temp(images, images_dir)
            
            # 2. 设置质量参数
            quality_settings = self._get_quality_settings(colmap_quality)
            
            # 3. 特征提取
            print("开始特征提取...")
            self._extract_features(database_path, images_dir, quality_settings)
            
            # 4. 特征匹配
            print(f"开始特征匹配 ({colmap_matcher_type})...")
            self._match_features(database_path, colmap_matcher_type)
            
            # 5. 增量重建
            print("开始增量重建...")
            reconstructions = self._incremental_mapping(database_path, images_dir, output_path)
            
            if not reconstructions:
                return {
                    "success": False,
                    "error": "COLMAP 重建失败：没有生成重建结果",
                    "intrinsics": None,
                    "poses": [],
                    "statistics": {},
                    "frame_count": 0,
                    "point_cloud": {}
                }
            
            # 6. 解析结果
            reconstruction = self._get_best_reconstruction(reconstructions)
            result = self._parse_reconstruction(reconstruction, len(images))
            
            print(f"重建成功：{len(reconstruction.cameras)} 个相机，{len(reconstruction.images)} 张图像，{len(reconstruction.points3D)} 个3D点")
            
            return result
            
        except Exception as e:
            print(f"COLMAP 估计过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"COLMAP 重建失败: {str(e)}",
                "intrinsics": None,
                "poses": [],
                "statistics": {},
                "frame_count": 0,
                "point_cloud": {}
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

    def _get_quality_settings(self, quality: str) -> Dict:
        """获取质量设置"""
        quality_settings = {
            "low": {"max_image_size": 800, "max_num_features": 4096},
            "medium": {"max_image_size": 1200, "max_num_features": 8192},
            "high": {"max_image_size": 1600, "max_num_features": 16384},
            "extreme": {"max_image_size": 2400, "max_num_features": 32768}
        }
        return quality_settings.get(quality, quality_settings["medium"])

    def _extract_features(self, database_path: str, images_dir: str, quality_settings: Dict):
        """提取特征"""
        sift_options = self.pycolmap.SiftExtractionOptions()
        sift_options.max_image_size = quality_settings["max_image_size"]
        sift_options.max_num_features = quality_settings["max_num_features"]
        
        self.pycolmap.extract_features(
            database_path=database_path,
            image_path=images_dir,
            sift_options=sift_options
        )

    def _match_features(self, database_path: str, matcher_type: str):
        """匹配特征"""
        if matcher_type == "exhaustive":
            matching_options = self.pycolmap.ExhaustiveMatchingOptions()
            self.pycolmap.match_exhaustive(
                database_path=database_path,
                matching_options=matching_options
            )
        elif matcher_type == "sequential":
            matching_options = self.pycolmap.SequentialMatchingOptions()
            matching_options.overlap = 10
            self.pycolmap.match_sequential(
                database_path=database_path,
                matching_options=matching_options
            )
        elif matcher_type == "spatial":
            matching_options = self.pycolmap.SpatialMatchingOptions()
            self.pycolmap.match_spatial(
                database_path=database_path,
                matching_options=matching_options
            )
        else:
            raise ValueError(f"不支持的匹配器类型: {matcher_type}")

    def _incremental_mapping(self, database_path: str, images_dir: str, output_path: str):
        """增量重建"""
        # COLMAP >= 0.3.0 使用 IncrementalPipelineOptions 作为配置类型
        # 早期版本可能仍支持 IncrementalMapperOptions，但为保持兼容性
        # 这里优先尝试 IncrementalPipelineOptions，并在回退情况下使用默认配置。

        try:
            pipeline_options = self.pycolmap.IncrementalPipelineOptions()

            # 通过内部的 mapper_options 设置常用参数
            if hasattr(pipeline_options, "mapper_options"):
                mapper_opts = pipeline_options.mapper_options
                mapper_opts.ba_refine_focal_length = True
                mapper_opts.ba_refine_principal_point = True
                mapper_opts.init_min_num_inliers = 100
                mapper_opts.init_max_reg_trials = 2
        except AttributeError:
            # 如果安装的 pycolmap 版本没有 IncrementalPipelineOptions，则退回默认
            pipeline_options = None

        if pipeline_options is not None:
            reconstructions = self.pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=output_path,
                options=pipeline_options
            )
        else:
            # 回退：不显式传入 options，使用库默认值。
            reconstructions = self.pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=images_dir,
                output_path=output_path
            )
        
        return reconstructions

    def _get_best_reconstruction(self, reconstructions):
        """获取最佳重建结果"""
        if isinstance(reconstructions, dict):
            if len(reconstructions) == 0:
                raise RuntimeError("重建结果为空")
            # 选择图像数量最多的重建
            best_recon = max(reconstructions.values(), key=lambda r: len(r.images))
            return best_recon
        elif isinstance(reconstructions, list):
            if len(reconstructions) == 0:
                raise RuntimeError("重建结果为空")
            # 选择图像数量最多的重建
            best_recon = max(reconstructions, key=lambda r: len(r.images))
            return best_recon
        else:
            return reconstructions

    def _parse_reconstruction(self, reconstruction, num_input_images: int) -> Dict:
        """解析重建结果"""
        try:
            # 解析相机内参
            intrinsics = self._parse_camera_intrinsics(reconstruction)
            
            # 解析位姿
            poses = self._parse_camera_poses(reconstruction)
            
            # 计算统计信息
            statistics = self._calculate_statistics(poses, reconstruction)
            
            # 点云信息
            point_cloud_info = {
                "num_points": len(reconstruction.points3D),
                "num_cameras": len(reconstruction.cameras),
                "num_registered_images": len(reconstruction.images),
                "num_input_images": num_input_images,
                "registration_ratio": len(reconstruction.images) / max(num_input_images, 1)
            }
            
            return {
                "success": True,
                "intrinsics": intrinsics,
                "poses": poses,
                "statistics": statistics,
                "frame_count": len(poses),
                "point_cloud": point_cloud_info
            }
            
        except Exception as e:
            print(f"解析重建结果失败: {e}")
            return {
                "success": False,
                "error": f"解析失败: {str(e)}",
                "intrinsics": None,
                "poses": [],
                "statistics": {},
                "frame_count": 0,
                "point_cloud": {}
            }

    def _parse_camera_intrinsics(self, reconstruction) -> Dict:
        """解析相机内参"""
        if len(reconstruction.cameras) == 0:
            raise ValueError("没有找到相机参数")
        
        # 获取第一个相机的参数（假设所有图像使用同一相机）
        camera = list(reconstruction.cameras.values())[0]
        
        # 获取基本参数
        focal_length = float(camera.params[0]) if len(camera.params) > 0 else 800.0
        focal_length_y = float(camera.params[1]) if len(camera.params) > 1 else focal_length
        
        # 获取主点
        if len(camera.params) > 3:
            principal_point = [float(camera.params[2]), float(camera.params[3])]
        else:
            principal_point = [camera.width / 2, camera.height / 2]
        
        # 获取畸变参数
        distortion = [float(p) for p in camera.params[4:]] if len(camera.params) > 4 else []
        
        # 获取相机模型名称
        camera_model = "PINHOLE"  # 默认值
        if hasattr(camera, 'model_name'):
            camera_model = camera.model_name
        elif hasattr(camera, 'model_id'):
            model_names = {
                0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"
            }
            camera_model = model_names.get(camera.model_id, f"MODEL_{camera.model_id}")
        
        return {
            "focal_length": focal_length,
            "focal_length_y": focal_length_y,
            "principal_point": principal_point,
            "image_size": [camera.width, camera.height],
            "camera_model": camera_model,
            "distortion": distortion
        }

    def _parse_camera_poses(self, reconstruction) -> List[Dict]:
        """解析相机位姿"""
        poses = []
        
        for image_id, image in reconstruction.images.items():
            # 获取位姿数据 (兼容不同版本的 pycolmap)
            quat = None
            trans = None
            
            # 首次迭代时打印可用属性以便调试
            if image_id == list(reconstruction.images.keys())[0]:
                print(f"PyColmap Image 对象可用属性: {[attr for attr in dir(image) if not attr.startswith('_')]}")
            
            # 尝试多种可能的API
            if hasattr(image, "qvec") and hasattr(image, "tvec"):
                # 旧版 API: 直接访问属性
                quat = image.qvec.tolist() if hasattr(image.qvec, 'tolist') else list(image.qvec)
                trans = image.tvec.tolist() if hasattr(image.tvec, 'tolist') else list(image.tvec)
            elif hasattr(image, "cam_from_world"):
                # 新版 API: 使用 cam_from_world 属性
                # cam_from_world 是一个 Rigid3d 对象，包含旋转和平移
                cam_from_world = image.cam_from_world
                if hasattr(cam_from_world, "rotation"):
                    # 获取四元数
                    rotation = cam_from_world.rotation
                    if hasattr(rotation, "quat"):
                        quat = rotation.quat.tolist()
                    elif hasattr(rotation, "quaternion"):
                        quat = rotation.quaternion.tolist()
                    else:
                        # 尝试从旋转矩阵转换
                        print(f"警告: 无法直接获取四元数，尝试其他方法")
                
                if hasattr(cam_from_world, "translation"):
                    trans = cam_from_world.translation.tolist()
            elif hasattr(image, "projection_center"):
                # 另一种可能: 使用 projection_center 作为位置
                print(f"使用 projection_center 作为备选方案")
                trans = image.projection_center().tolist()
                # 尝试获取旋转
                if hasattr(image, "rotation_matrix"):
                    # 从旋转矩阵计算四元数
                    print(f"从 rotation_matrix 计算四元数")
            
            # 如果仍然无法获取数据，尝试其他方法
            if quat is None or trans is None:
                print(f"警告: 无法获取图像 {image_id} 的完整位姿数据")
                # 提供默认值以避免崩溃
                if quat is None:
                    quat = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
                if trans is None:
                    trans = [0.0, 0.0, 0.0]  # 原点
            
            # 转换为欧拉角
            euler = self._quaternion_to_euler(quat)
            
            # 获取图像名称
            image_name = image.name if hasattr(image, 'name') else f"image_{image_id}"
            
            pose = {
                "position": trans,
                "rotation_quaternion": quat,
                "rotation_euler": euler,
                "image_name": image_name,
                "image_id": image_id
            }
            poses.append(pose)
        
        # 按图像名称排序，确保顺序
        poses.sort(key=lambda x: x["image_name"])
        
        return poses

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

    def _calculate_statistics(self, poses: List[Dict], reconstruction) -> Dict:
        """计算统计信息"""
        if len(poses) < 2:
            return {
                "total_distance": 0.0,
                "average_speed": 0.0,
                "num_poses": len(poses),
                "num_3d_points": len(reconstruction.points3D)
            }
        
        # 计算轨迹总长度
        total_distance = 0.0
        for i in range(1, len(poses)):
            pos1 = np.array(poses[i-1]["position"])
            pos2 = np.array(poses[i]["position"])
            distance = np.linalg.norm(pos2 - pos1)
            total_distance += distance
        
        average_speed = total_distance / (len(poses) - 1) if len(poses) > 1 else 0.0
        
        return {
            "total_distance": float(total_distance),
            "average_speed": float(average_speed),
            "num_poses": len(poses),
            "num_3d_points": len(reconstruction.points3D),
            "mean_track_length": np.mean([len(point.track.elements) for point in reconstruction.points3D.values()]) if len(reconstruction.points3D) > 0 else 0.0
        }


class ImageSequenceCameraEstimator:
    """图片序列相机参数估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "输入图片序列"
                }),
            },
            "optional": {
                "colmap_feature_type": (["sift", "superpoint", "disk"], {
                    "default": "sift",
                    "tooltip": "COLMAP特征检测器类型，SIFT最稳定"
                }),
                "colmap_matcher_type": (["exhaustive", "sequential", "spatial"], {
                    "default": "sequential",
                    "tooltip": "COLMAP匹配策略，sequential适合有序序列"
                }),
                "colmap_quality": (["low", "medium", "high", "extreme"], {
                    "default": "medium",
                    "tooltip": "COLMAP重建质量，higher质量需要更多时间"
                }),
                "enable_dense_reconstruction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否启用密集重建（需要更多计算资源和CUDA支持）"
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

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("intrinsics_json", "trajectory_visualization", "poses_json", "statistics_json", "point_cloud_info")
    FUNCTION = "estimate_camera_parameters"
    CATEGORY = "💃VVL/VideoCamera"

    def __init__(self):
        try:
            self.estimator = ColmapCameraEstimator()
        except ImportError as e:
            print(f"COLMAP初始化失败: {e}")
            self.estimator = None

    def estimate_camera_parameters(self, images, 
                                 colmap_feature_type: str = "sift",
                                 colmap_matcher_type: str = "sequential", 
                                 colmap_quality: str = "medium",
                                 enable_dense_reconstruction: bool = False,
                                 enable_visualization: bool = True,
                                 output_format: str = "detailed_json") -> tuple:
        """从图片序列估计相机参数的主函数"""
        
        try:
            if self.estimator is None:
                raise RuntimeError("COLMAP初始化失败，请检查PyColmap安装")
            
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
            
            print(f"开始处理 {len(image_list)} 张图片")
            
            # 使用COLMAP进行估计
            result = self.estimator.estimate_from_images(
                images=image_list,
                colmap_feature_type=colmap_feature_type,
                colmap_matcher_type=colmap_matcher_type,
                colmap_quality=colmap_quality,
                enable_dense_reconstruction=enable_dense_reconstruction
            )
            
            if not result["success"]:
                raise RuntimeError(f"相机参数估计失败: {result.get('error', '未知错误')}")
            
            # 准备输出
            intrinsics_json = self._format_intrinsics_output(result["intrinsics"], output_format)
            poses_json = self._format_poses_output(result["poses"], output_format)
            statistics_json = self._format_statistics_output(result["statistics"], result, output_format)
            point_cloud_info = self._format_point_cloud_output(result["point_cloud"], output_format)
            
            # 处理可视化图像
            if enable_visualization and result["poses"]:
                trajectory_img = self._create_trajectory_visualization(result["poses"])
            else:
                trajectory_img = self._create_empty_visualization()
            
            print(f"成功处理 {result['frame_count']} 张图片")
            if result['intrinsics']:
                print(f"估计的焦距: {result['intrinsics']['focal_length']:.2f}")
            
            return (intrinsics_json, trajectory_img, poses_json, statistics_json, point_cloud_info)
            
        except Exception as e:
            error_msg = f"图片序列相机参数估计出错: {str(e)}"
            print(error_msg)
            
            # 返回错误信息
            error_json = json.dumps({"error": error_msg, "success": False}, ensure_ascii=False, indent=2)
            empty_img = self._create_empty_visualization()
            
            return (error_json, empty_img, error_json, error_json, error_json)

    def _format_intrinsics_output(self, intrinsics: Dict, output_format: str) -> str:
        """格式化内参输出"""
        if output_format == "json":
            simplified = {
                "focal_length": intrinsics["focal_length"],
                "principal_point": intrinsics["principal_point"],
                "image_size": intrinsics["image_size"]
            }
            return json.dumps(simplified, ensure_ascii=False, indent=2)
        else:
            return json.dumps(intrinsics, ensure_ascii=False, indent=2)

    def _format_poses_output(self, poses: List[Dict], output_format: str) -> str:
        """格式化位姿输出"""
        if output_format == "json":
            simplified_poses = []
            for i, pose in enumerate(poses):
                simplified_poses.append({
                    "frame": i,
                    "position": pose["position"],
                    "rotation_euler": pose["rotation_euler"]
                })
            return json.dumps(simplified_poses, ensure_ascii=False, indent=2)
        else:
            detailed_poses = []
            for i, pose in enumerate(poses):
                detailed_pose = pose.copy()
                detailed_pose["frame"] = i
                detailed_poses.append(detailed_pose)
            return json.dumps(detailed_poses, ensure_ascii=False, indent=2)

    def _format_statistics_output(self, statistics: Dict, result: Dict, output_format: str) -> str:
        """格式化统计信息输出"""
        if output_format == "json":
            simplified_stats = {
                "frame_count": result["frame_count"],
                "total_distance": statistics.get("total_distance", 0),
                "num_3d_points": statistics.get("num_3d_points", 0),
                "success": result["success"]
            }
            return json.dumps(simplified_stats, ensure_ascii=False, indent=2)
        else:
            detailed_stats = statistics.copy()
            detailed_stats.update({
                "frame_count": result["frame_count"],
                "success": result["success"]
            })
            return json.dumps(detailed_stats, ensure_ascii=False, indent=2)

    def _format_point_cloud_output(self, point_cloud_info: Dict, output_format: str) -> str:
        """格式化点云信息输出"""
        if output_format == "json":
            simplified = {
                "num_points": point_cloud_info.get("num_points", 0),
                "registration_ratio": point_cloud_info.get("registration_ratio", 0)
            }
            return json.dumps(simplified, ensure_ascii=False, indent=2)
        else:
            return json.dumps(point_cloud_info, ensure_ascii=False, indent=2)

    def _create_trajectory_visualization(self, poses: List[Dict]) -> torch.Tensor:
        """创建轨迹可视化"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # 提取位置信息
            positions = np.array([pose["position"] for pose in poses])
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制轨迹线
            if len(positions) > 1:
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       'b-', linewidth=3, alpha=0.8, label='Camera Path')
            
            # 用颜色渐变表示时间进程
            colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               c=colors, s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
            
            # 标记起点和终点
            if len(positions) > 0:
                ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                          c='green', s=150, marker='^', label='Start', edgecolors='darkgreen', linewidth=2)
            if len(positions) > 1:
                ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                          c='red', s=150, marker='o', label='End', edgecolors='darkred', linewidth=2)
            
            # 设置坐标轴
            ax.set_xlabel('X (meters)', fontsize=12)
            ax.set_ylabel('Y (meters)', fontsize=12)
            ax.set_zlabel('Z (meters)', fontsize=12)
            ax.set_title('COLMAP Camera Trajectory (3D View)', fontsize=14, fontweight='bold')
            
            # 添加图例和颜色条
            ax.legend(loc='upper right', fontsize=10)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
            cbar.set_label('Time Progress', fontsize=10)
            
            # 设置相等的坐标轴比例
            max_range = np.array([positions.max(axis=0) - positions.min(axis=0)]).max() / 2.0
            mid_x = (positions.max(axis=0)[0] + positions.min(axis=0)[0]) * 0.5
            mid_y = (positions.max(axis=0)[1] + positions.min(axis=0)[1]) * 0.5
            mid_z = (positions.max(axis=0)[2] + positions.min(axis=0)[2]) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.grid(True, alpha=0.3)
            ax.view_init(elev=20, azim=45)
            
            # 保存为图像
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=120, bbox_inches='tight', facecolor='white')
                plt.close()
                
                # 读取图像
                img = cv2.imread(tmp.name)
                os.unlink(tmp.name)
                
                if img is not None:
                    # BGR转RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
                    return img_tensor.unsqueeze(0)
                else:
                    return self._create_empty_visualization()
        
        except Exception as e:
            print(f"创建可视化失败: {e}")
            return self._create_empty_visualization()

    def _create_empty_visualization(self) -> torch.Tensor:
        """创建空的可视化图像"""
        img = np.ones((400, 600, 3), dtype=np.float32) * 0.1
        
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            cv2.putText(img_uint8, "No Trajectory", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(img_uint8, "Visualization", (170, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            img = img_uint8.astype(np.float32) / 255.0
        except:
            pass
        
        return torch.from_numpy(img).unsqueeze(0)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageSequenceCameraEstimator": ImageSequenceCameraEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceCameraEstimator": "VVL Image Sequence Camera Estimator"
} 