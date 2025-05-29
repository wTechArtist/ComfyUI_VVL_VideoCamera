import cv2
import numpy as np
import torch
import json
import os
import tempfile
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from colormap import Colormap

class CameraParameterEstimator:
    """相机参数估计器"""
    
    def __init__(self):
        self.colormap = Colormap()
        
    def extract_frames_from_video(self, video_path: str, frame_interval: int = 10, max_frames: int = 50) -> List[np.ndarray]:
        """从视频中提取关键帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        saved_count = 0
        
        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
                saved_count += 1
                
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_and_match_features(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测和匹配特征点"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # 使用SIFT检测器（更稳定）
        try:
            detector = cv2.SIFT_create(nfeatures=2000)
        except:
            # 如果SIFT不可用，使用ORB
            detector = cv2.ORB_create(nfeatures=2000)
        
        # 检测关键点和描述符
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, None
        
        # 特征匹配
        if hasattr(detector, 'defaultNorm') and detector.defaultNorm() == cv2.NORM_HAMMING:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 提取匹配点
        if len(matches) > 50:  # 确保有足够的匹配点
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:200]]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:200]]).reshape(-1, 1, 2)
            return pts1, pts2
        
        return None, None
    
    def estimate_intrinsic_parameters(self, frames: List[np.ndarray]) -> Dict:
        """估计相机内参"""
        h, w = frames[0].shape[:2]
        
        # 使用多种方法估计焦距
        focal_estimates = []
        
        # 方法1: 基于图像尺寸的经验估计
        empirical_focal = max(w, h) * 1.2  # 经验值
        focal_estimates.append(empirical_focal)
        
        # 方法2: 基于特征匹配的估计
        if len(frames) >= 2:
            focal_from_motion = self._estimate_focal_from_motion(frames[:5])
            if focal_from_motion is not None:
                focal_estimates.append(focal_from_motion)
        
        # 取平均值作为最终估计
        focal_length = np.mean(focal_estimates) if focal_estimates else empirical_focal
        
        # 主点通常在图像中心
        cx = w / 2.0
        cy = h / 2.0
        
        # 构建相机内参矩阵
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        
        # 畸变参数（假设为径向畸变）
        distortion_coeffs = np.array([0.1, -0.05, 0, 0, 0])  # 初始估计
        
        intrinsics = {
            "camera_matrix": camera_matrix.tolist(),
            "focal_length": float(focal_length),
            "principal_point": [float(cx), float(cy)],
            "distortion_coefficients": distortion_coeffs.tolist(),
            "image_size": [int(w), int(h)]
        }
        
        return intrinsics
    
    def _estimate_focal_from_motion(self, frames: List[np.ndarray]) -> Optional[float]:
        """通过相机运动估计焦距"""
        focal_estimates = []
        
        for i in range(len(frames) - 1):
            pts1, pts2 = self.detect_and_match_features(frames[i], frames[i + 1])
            
            if pts1 is not None and pts2 is not None and len(pts1) > 8:
                # 使用五点法估计本质矩阵和焦距
                h, w = frames[i].shape[:2]
                
                # 假设主点在图像中心
                cx, cy = w / 2.0, h / 2.0
                
                # 尝试不同的焦距值，找到最佳的
                best_focal = None
                max_inliers = 0
                
                for f in np.linspace(w * 0.5, w * 2.0, 20):
                    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
                    
                    try:
                        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        inliers = np.sum(mask)
                        
                        if inliers > max_inliers:
                            max_inliers = inliers
                            best_focal = f
                    except:
                        continue
                
                if best_focal is not None:
                    focal_estimates.append(best_focal)
        
        return np.median(focal_estimates) if focal_estimates else None
    
    def estimate_camera_poses(self, frames: List[np.ndarray], intrinsics: Dict) -> List[Dict]:
        """估计相机外参（位姿）"""
        camera_matrix = np.array(intrinsics["camera_matrix"])
        poses = []
        
        # 第一帧作为世界坐标系原点
        poses.append({
            "rotation_matrix": np.eye(3).tolist(),
            "translation_vector": np.zeros(3).tolist(),
            "rotation_euler": [0.0, 0.0, 0.0],
            "position": [0.0, 0.0, 0.0]
        })
        
        cumulative_R = np.eye(3)
        cumulative_t = np.zeros(3)
        
        for i in range(len(frames) - 1):
            pts1, pts2 = self.detect_and_match_features(frames[i], frames[i + 1])
            
            if pts1 is not None and pts2 is not None:
                # 估计本质矩阵
                E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
                
                # 恢复位姿
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix)
                
                # 累积变换
                cumulative_R = R @ cumulative_R
                cumulative_t = R @ cumulative_t + t.flatten()
                
                # 计算欧拉角
                euler = self._rotation_matrix_to_euler(cumulative_R)
                
                pose = {
                    "rotation_matrix": cumulative_R.tolist(),
                    "translation_vector": cumulative_t.tolist(),
                    "rotation_euler": euler,
                    "position": (-cumulative_R.T @ cumulative_t).tolist()  # 相机在世界坐标系中的位置
                }
                poses.append(pose)
            else:
                # 如果匹配失败，复制上一个位姿
                poses.append(poses[-1])
        
        return poses
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> List[float]:
        """将旋转矩阵转换为欧拉角"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return [float(np.degrees(x)), float(np.degrees(y)), float(np.degrees(z))]
    
    def create_trajectory_visualization(self, poses: List[Dict]) -> np.ndarray:
        """创建相机轨迹可视化"""
        positions = np.array([pose["position"] for pose in poses])
        
        if len(positions) == 0:
            # 返回空图像
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        if len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, alpha=0.7)
        
        # 用颜色表示时间
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c=colors, s=50, alpha=0.8)
        
        # 标记起点和终点
        if len(positions) > 0:
            ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 
                      c='red', s=100, marker='o', label='Start')
        if len(positions) > 1:
            ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 
                      c='green', s=100, marker='s', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory')
        ax.legend()
        
        # 保存为图像
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, dpi=100, bbox_inches='tight')
            plt.close()
            
            # 读取图像
            img = cv2.imread(tmp.name)
            os.unlink(tmp.name)
            
            return img if img is not None else np.zeros((400, 400, 3), dtype=np.uint8)
    
    def estimate_from_video(self, video_path: str, frame_interval: int = 10, max_frames: int = 50) -> Dict:
        """从视频估计完整的相机参数"""
        try:
            # 1. 提取帧
            frames = self.extract_frames_from_video(video_path, frame_interval, max_frames)
            
            if len(frames) == 0:
                raise ValueError("无法从视频中提取帧")
            
            # 2. 估计内参
            intrinsics = self.estimate_intrinsic_parameters(frames)
            
            # 3. 估计外参
            poses = self.estimate_camera_poses(frames, intrinsics)
            
            # 4. 创建可视化
            trajectory_img = self.create_trajectory_visualization(poses)
            
            # 5. 计算统计信息
            stats = self._calculate_statistics(poses)
            
            result = {
                "success": True,
                "intrinsics": intrinsics,
                "poses": poses,
                "statistics": stats,
                "frame_count": len(frames),
                "trajectory_visualization": trajectory_img.tolist() if trajectory_img is not None else None
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "intrinsics": None,
                "poses": [],
                "statistics": None,
                "frame_count": 0,
                "trajectory_visualization": None
            }
    
    def _calculate_statistics(self, poses: List[Dict]) -> Dict:
        """计算相机运动统计信息"""
        if len(poses) < 2:
            return {
                "total_distance": 0.0,
                "max_distance": 0.0,
                "average_speed": 0.0,
                "rotation_range": [0.0, 0.0, 0.0]
            }
        
        positions = np.array([pose["position"] for pose in poses])
        rotations = np.array([pose["rotation_euler"] for pose in poses])
        
        # 计算总路程
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        max_distance = np.max(distances) if len(distances) > 0 else 0.0
        average_speed = total_distance / len(poses) if len(poses) > 1 else 0.0
        
        # 计算旋转范围
        rotation_range = [
            float(np.max(rotations[:, i]) - np.min(rotations[:, i])) 
            for i in range(3)
        ]
        
        return {
            "total_distance": float(total_distance),
            "max_distance": float(max_distance),
            "average_speed": float(average_speed),
            "rotation_range": rotation_range
        } 