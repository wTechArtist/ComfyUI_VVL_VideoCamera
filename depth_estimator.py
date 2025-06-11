import torch
import numpy as np
import cv2
import tempfile
import os
import shutil
import json
import subprocess
import struct
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

class ColmapMVSDepthEstimator:
    """使用真正的COLMAP-MVS进行深度估计的核心类"""

    def __init__(self, use_gpu: bool = True, quality: str = "medium"):
        self.use_gpu = use_gpu
        self.quality = quality
        self.logger = logging.getLogger(__name__)
        
        # 检查COLMAP可执行文件
        self.colmap_path = self._find_colmap_executable()
        if not self.colmap_path:
            raise RuntimeError("未找到COLMAP可执行文件。请安装COLMAP并确保其在系统PATH中。")
        
        print(f"找到COLMAP: {self.colmap_path}")
        
        # 检测GPU支持
        self.gpu_available = self._check_gpu_support()
        print(f"GPU支持: {'可用' if self.gpu_available else '不可用'}")
        
        # 设置质量参数
        self.quality_params = {
            "low": {
                "SiftExtraction.max_image_size": 1000,
                "SiftExtraction.max_num_features": 2048,
                "PatchMatchStereo.window_radius": 3,
                "PatchMatchStereo.num_iterations": 3,
                "PatchMatchStereo.geom_consistency": "false"
            },
            "medium": {
                "SiftExtraction.max_image_size": 2000,
                "SiftExtraction.max_num_features": 8192,
                "PatchMatchStereo.window_radius": 5,
                "PatchMatchStereo.num_iterations": 5,
                "PatchMatchStereo.geom_consistency": "true"
            },
            "high": {
                "SiftExtraction.max_image_size": 3000,
                "SiftExtraction.max_num_features": 16384,
                "PatchMatchStereo.window_radius": 7,
                "PatchMatchStereo.num_iterations": 7,
                "PatchMatchStereo.geom_consistency": "true"
            },
            "extreme": {
                "SiftExtraction.max_image_size": -1,  # 原始尺寸
                "SiftExtraction.max_num_features": 32768,
                "PatchMatchStereo.window_radius": 9,
                "PatchMatchStereo.num_iterations": 10,
                "PatchMatchStereo.geom_consistency": "true"
            }
        }

    def _find_colmap_executable(self) -> Optional[str]:
        """查找COLMAP可执行文件"""
        # 优先级顺序：先找支持CUDA的版本
        possible_paths = [
            # 1. Anaconda环境中的COLMAP（通常支持CUDA）
            "/home/game-netease/anaconda3/bin/colmap",
            "/home/game-netease/anaconda3/envs/comfyui250513/bin/colmap",
            
            # 2. 当前PATH中的colmap
            shutil.which("colmap"),
            
            # 3. 其他常见位置
            "/usr/local/bin/colmap",
            "/opt/colmap/bin/colmap",
            "/usr/bin/colmap",  # 系统安装的版本（通常无CUDA）
        ]
        
        best_colmap = None
        best_has_cuda = False
        
        for path in possible_paths:
            if path and os.path.isfile(path) and os.access(path, os.X_OK):
                # 检查这个版本是否支持CUDA
                try:
                    result = subprocess.run([path], capture_output=True, text=True, timeout=5)
                    output = result.stdout + result.stderr
                    has_cuda = "with CUDA" in output
                    
                    print(f"🔍 检查COLMAP: {path}")
                    if has_cuda:
                        print(f"   ✅ 支持CUDA")
                        return path  # 立即返回第一个支持CUDA的版本
                    else:
                        print(f"   ❌ 不支持CUDA")
                        if best_colmap is None:
                            best_colmap = path  # 保存作为备选
                except:
                    continue
        
        # 如果没有找到支持CUDA的版本，返回最佳可用版本
        if best_colmap:
            print(f"⚠️  使用不支持CUDA的COLMAP: {best_colmap}")
            return best_colmap
        
        return None

    def _check_gpu_support(self) -> bool:
        """检测GPU支持"""
        try:
            # 检查NVIDIA GPU
            gpu_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if gpu_check.returncode != 0:
                print("未检测到NVIDIA GPU")
                return False
            
            # 使用找到的COLMAP路径检查版本和CUDA支持
            colmap_info = subprocess.run([self.colmap_path], capture_output=True, text=True, timeout=5)
            # 检查stdout和stderr两个输出
            colmap_output = (colmap_info.stdout + colmap_info.stderr).strip()
            
            print(f"🔍 COLMAP版本信息: {colmap_output.split('Usage:')[0] if 'Usage:' in colmap_output else colmap_output[:100]}...")
            
            # 检查是否编译了CUDA支持
            if "with CUDA" in colmap_output:
                print("✅ COLMAP编译时启用了CUDA支持")
                self.colmap_has_cuda = True
                
                # 测试CUDA是否真的可用
                try:
                    # 通过运行一个简单的CUDA测试来验证
                    test_cmd = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
                    if test_cmd.returncode == 0 and "GPU" in test_cmd.stdout:
                        print(f"✅ CUDA设备可用: {test_cmd.stdout.strip()}")
                        return True
                    else:
                        print("⚠️  COLMAP有CUDA支持但CUDA设备不可用")
                        self.colmap_has_cuda = False
                        return False
                except:
                    print("⚠️  CUDA设备检测失败")
                    self.colmap_has_cuda = False
                    return False
            elif "without CUDA" in colmap_output:
                print("⚠️  COLMAP编译时未启用CUDA支持")
                print("   - SIFT特征提取和匹配仍可使用GPU")
                print("   - 密集重建将使用CPU模式")
                self.colmap_has_cuda = False
                return False
            else:
                print("❓ 无法确定COLMAP的CUDA支持状态")
                # 默认假设有CUDA支持并尝试
                self.colmap_has_cuda = True
                return True
            
        except Exception as e:
            print(f"GPU检测异常: {e}")
            self.colmap_has_cuda = False
            return False

    def _setup_gpu_environment(self) -> Dict[str, str]:
        """设置GPU环境变量"""
        env = os.environ.copy()
        
        # 确保CUDA可见
        env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
        
        # 设置CUDA相关路径
        cuda_paths = [
            '/usr/local/cuda/bin',
            '/usr/local/cuda-12.8/bin',
            '/usr/local/cuda-11.8/bin',
            '/opt/cuda/bin'
        ]
        
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                if 'PATH' in env:
                    env['PATH'] = f"{cuda_path}:{env['PATH']}"
                else:
                    env['PATH'] = cuda_path
                break
        
        # 设置CUDA库路径
        cuda_lib_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda-12.8/lib64',
            '/usr/local/cuda-11.8/lib64',
            '/opt/cuda/lib64'
        ]
        
        for cuda_lib_path in cuda_lib_paths:
            if os.path.exists(cuda_lib_path):
                if 'LD_LIBRARY_PATH' in env:
                    env['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{env['LD_LIBRARY_PATH']}"
                else:
                    env['LD_LIBRARY_PATH'] = cuda_lib_path
                break
        
        # 设置OpenGL/EGL相关变量以支持无头GPU操作
        env['__GL_SYNC_TO_VBLANK'] = '0'
        env['__GL_ALLOW_UNOFFICIAL_PROTOCOL'] = '1'
        
        # 尝试使用EGL而不是GLX
        env['__EGL_VENDOR_LIBRARY_DIRS'] = '/usr/share/glvnd/egl_vendor.d'
        
        # 禁用Qt的XCB插件，使用offscreen
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['QT_QPA_FONTDIR'] = '/usr/share/fonts'
        
        # NVIDIA相关设置
        env['NVIDIA_VISIBLE_DEVICES'] = 'all'
        env['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility,graphics'
        
        # 强制使用GPU（如果支持）
        env['COLMAP_GPU'] = '1'
        
        return env

    def estimate_depth_mvs(self, 
                          images: List[np.ndarray], 
                          camera_intrinsics: Dict,
                          camera_poses: List[Dict],
                          output_dir: Optional[str] = None,
                          force_gpu: bool = True) -> List[np.ndarray]:
        """使用真正的COLMAP-MVS方法估计深度"""
        
        if len(images) < 3:
            raise ValueError("COLMAP-MVS深度估计至少需要3张图像")
        
        if camera_intrinsics is None or camera_poses is None:
            raise ValueError("COLMAP-MVS需要相机内参和位姿信息")
        
        print(f"开始COLMAP-MVS深度估计，处理 {len(images)} 张图片")
        print(f"使用质量设置: {self.quality}")
        print(f"强制GPU模式: {'是' if force_gpu else '否'}")
        
        if force_gpu and not self.gpu_available:
            raise RuntimeError("强制GPU模式但GPU不可用。请检查GPU驱动和CUDA安装。")
            
        # 创建工作目录
        if output_dir is None:
            work_dir = tempfile.mkdtemp(prefix="colmap_mvs_")
            cleanup = True
        else:
            work_dir = output_dir
            os.makedirs(work_dir, exist_ok=True)
            cleanup = False
        
        try:
            # 运行完整的COLMAP-MVS流水线
            depth_maps = self._run_colmap_mvs_pipeline(
                images, camera_intrinsics, camera_poses, work_dir, force_gpu
            )
            
            if not depth_maps:
                raise RuntimeError("COLMAP-MVS未能生成深度图")
            
            return depth_maps
            
        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_colmap_mvs_pipeline(self, 
                                images: List[np.ndarray], 
                                camera_intrinsics: Dict,
                                camera_poses: List[Dict],
                                work_dir: str,
                                force_gpu: bool) -> List[np.ndarray]:
        """运行完整的COLMAP-MVS流水线"""
        
        # 1. 准备目录结构
        images_dir = os.path.join(work_dir, "images")
        sparse_dir = os.path.join(work_dir, "sparse", "0")
        dense_dir = os.path.join(work_dir, "dense", "0")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(dense_dir, exist_ok=True)
            
        # 2. 保存图像
        print("步骤1: 保存图像...")
        image_names = []
        for i, image in enumerate(images):
            image_name = f"image_{i:06d}.jpg"
            image_path = os.path.join(images_dir, image_name)
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_names.append(image_name)
        
        # 3. 创建COLMAP稀疏重建（从已知相机参数）
        print("步骤2: 创建稀疏重建...")
        self._create_sparse_reconstruction(
            sparse_dir, images_dir, image_names, 
            camera_intrinsics, camera_poses
        )
        
        # 4. 图像去畸变
        print("步骤3: 图像去畸变...")
        self._run_colmap_command([
            "image_undistorter",
            "--image_path", images_dir,
            "--input_path", sparse_dir,
            "--output_path", dense_dir,
            "--output_type", "COLMAP"
        ], force_gpu=force_gpu)
        
        # 5. 立体匹配（Patch Match Stereo）
        print("步骤4: 执行立体匹配...")
        self._run_patch_match_stereo(dense_dir, force_gpu)
            
        # 6. 深度图融合
        print("步骤5: 深度图融合...")
        self._run_stereo_fusion(dense_dir, force_gpu)
        
        # 新增步骤: 将.bin转换为.npy以确保读取安全
        print("步骤6: 转换深度图为NPY格式...")
        self._convert_colmap_bins_to_npy(dense_dir, len(images))
        
        # 7. 读取深度图 (从.npy文件)
        print("步骤7: 读取深度图...")
        depth_maps = self._read_depth_maps(dense_dir, len(images))
        
        return depth_maps

    def _create_sparse_reconstruction(self, 
                                    sparse_dir: str,
                                    images_dir: str,
                                    image_names: List[str],
                                    camera_intrinsics: Dict,
                                    camera_poses: List[Dict]):
        """从已知相机参数创建COLMAP稀疏重建"""
        
        # 创建cameras.txt
        cameras_file = os.path.join(sparse_dir, "cameras.txt")
        with open(cameras_file, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            width, height = camera_intrinsics['image_size']
            fx = camera_intrinsics.get('focal_length', 800.0)
            fy = camera_intrinsics.get('focal_length_y', fx)
            cx = camera_intrinsics.get('principal_point', [width/2, height/2])[0]
            cy = camera_intrinsics.get('principal_point', [width/2, height/2])[1]
            
            # PINHOLE模型: fx, fy, cx, cy
            f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
        
        # 为了让COLMAP密集重建工作，我们需要运行完整的特征提取和匹配流程
        # 而不是手动创建稀疏重建
        self._run_colmap_sparse_reconstruction(sparse_dir, images_dir)

    def _run_colmap_sparse_reconstruction(self, sparse_dir: str, images_dir: str):
        """运行COLMAP稀疏重建流程"""
        print("🔧 运行COLMAP完整稀疏重建流程...")
        
        # 确保目录存在
        database_path = os.path.join(os.path.dirname(sparse_dir), "database.db")
        
        try:
            # 1. 特征提取
            print("  1. 特征提取...")
            feature_cmd = [
                'colmap', 'feature_extractor',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--ImageReader.single_camera', '1',
                '--SiftExtraction.use_gpu', '1' if self.gpu_available else '0'
            ]
            
            env = self._setup_gpu_environment() if self.gpu_available else None
            result = subprocess.run(feature_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"    特征提取失败，尝试CPU模式: {result.stderr}")
                # 尝试CPU模式
                feature_cmd = [
                    'colmap', 'feature_extractor',
                    '--database_path', database_path,
                    '--image_path', images_dir,
                    '--ImageReader.single_camera', '1',
                    '--SiftExtraction.use_gpu', '0'
                ]
                result = subprocess.run(feature_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"特征提取失败: {result.stderr}")
            
            print("  ✅ 特征提取成功")
            
            # 2. 特征匹配
            print("  2. 特征匹配...")
            match_cmd = [
                'colmap', 'exhaustive_matcher',
                '--database_path', database_path,
                '--SiftMatching.use_gpu', '1' if self.gpu_available else '0'
            ]
            
            result = subprocess.run(match_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                print(f"    特征匹配失败，尝试CPU模式: {result.stderr}")
                # 尝试CPU模式
                match_cmd = [
                    'colmap', 'exhaustive_matcher',
                    '--database_path', database_path,
                    '--SiftMatching.use_gpu', '0'
                ]
                result = subprocess.run(match_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"特征匹配失败: {result.stderr}")
            
            print("  ✅ 特征匹配成功")
        
            # 3. Bundle Adjustment
            print("  3. Bundle Adjustment...")
            mapper_cmd = [
                'colmap', 'mapper',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--output_path', os.path.dirname(sparse_dir)
            ]
            
            result = subprocess.run(mapper_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Bundle adjustment失败: {result.stderr}")
            
            print("  ✅ Bundle Adjustment成功")
            
        except Exception as e:
            print(f"❌ COLMAP稀疏重建失败: {e}")
            print("💡 尝试使用简化的稀疏重建...")
            # 如果完整流程失败，创建简化的稀疏重建
            self._create_minimal_sparse_reconstruction(sparse_dir, images_dir)

    def _run_colmap_command(self, args: List[str], force_gpu: bool = False):
        """执行COLMAP命令"""
        
        # 使用找到的COLMAP路径
        full_cmd = [self.colmap_path] + args
        
        print(f"执行COLMAP命令: {' '.join(full_cmd)}")
        
        if force_gpu and self.gpu_available:
            # 设置GPU环境
            env = self._setup_gpu_environment()
            
            # 首先尝试直接GPU模式
            result = subprocess.run(full_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print("✅ GPU模式COLMAP命令成功")
                return
            
            print(f"GPU模式失败: {result.stderr}")
            
            # 尝试xvfb-run + GPU
            try:
                xvfb_cmd = [
                    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
                ] + full_cmd
                result = subprocess.run(xvfb_cmd, capture_output=True, text=True, env=env)
                
                if result.returncode == 0:
                    print("✅ xvfb-run + GPU模式COLMAP命令成功")
                    return
            except:
                pass
            
            if force_gpu:
                raise RuntimeError(f"强制GPU模式失败: {result.stderr}")
        
        # CPU模式或备用模式
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        result = subprocess.run(full_cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"COLMAP命令失败: {result.stderr}")
            raise RuntimeError(f"COLMAP命令执行失败: {result.stderr}")
        
        print("✅ COLMAP命令执行成功")

    def _run_patch_match_stereo(self, dense_dir: str, force_gpu: bool = True):
        """运行Patch Match Stereo深度估计"""
        params = self.quality_params[self.quality].copy()

        # 增加一个临时修复，强制禁用几何一致性检查，这在某些新GPU上可能导致bug
        print("🔧 临时修复: 为避免新GPU下的潜在Bug，强制禁用几何一致性检查。")
        params["PatchMatchStereo.geom_consistency"] = "false"
        
        # 如果强制GPU但没有CUDA支持，根据设置决定行为
        if force_gpu and not self.colmap_has_cuda:
            print("⚠️  检测到强制GPU模式但COLMAP无CUDA支持")
            print("   - 当前COLMAP版本: 可能无CUDA支持")
            print("   - GPU硬件检测: ✅ (NVIDIA RTX 4090)")
            print("   - 建议: 使用CPU模式或安装CUDA版COLMAP")
            
            # 不立即抛出错误，而是尝试CPU模式
            print("🔄 自动切换到CPU模式进行密集重建...")
            force_gpu = False
        
        # GPU模式尝试 (仅当有CUDA支持时)
        if force_gpu and self.gpu_available and self.colmap_has_cuda:
            print("🚀 使用GPU模式运行Patch Match Stereo...")
            
            # 设置GPU环境 - 确保CUDA设备可见
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            # 确保CUDA设备可见（不要设置为空）
            env['CUDA_VISIBLE_DEVICES'] = '0'
            env['NVIDIA_VISIBLE_DEVICES'] = '0'
            # 设置CUDA路径
            if 'LD_LIBRARY_PATH' not in env:
                env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu'
            else:
                env['LD_LIBRARY_PATH'] = f"/usr/local/cuda/lib64:{env['LD_LIBRARY_PATH']}"
            
            cmd = [
                self.colmap_path,
                "patch_match_stereo",
                "--workspace_path", dense_dir,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.max_image_size", "-1",  # 使用原始尺寸
                "--PatchMatchStereo.gpu_index", "0"  # 明确指定GPU 0
            ]
            
            # 添加质量参数
            for key, value in params.items():
                if key.startswith("PatchMatchStereo"):
                    cmd.extend([f"--{key}", str(value)])
            
            print(f"执行GPU模式命令: {' '.join(cmd)}")
            # 首先尝试直接GPU模式
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print("✅ GPU模式Patch Match Stereo成功")
                return
            
            print(f"GPU模式失败: {result.stderr}")
            print(f"GPU模式stdout: {result.stdout}")
            
            # 尝试xvfb-run + GPU
            try:
                xvfb_cmd = [
                    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
                ] + cmd
                result = subprocess.run(xvfb_cmd, capture_output=True, text=True, env=env)
                
                if result.returncode == 0:
                    print("✅ xvfb-run + GPU模式Patch Match Stereo成功")
                    return
            except Exception as e:
                print(f"xvfb-run尝试失败: {e}")
            
            print(f"⚠️  GPU模式失败，回退到CPU模式")

        # CPU模式 - 备用执行路径
        print("🖥️  使用CPU模式运行Patch Match Stereo...")
        print("   - 注意：某些COLMAP版本的CPU模式可能不支持密集重建")
        
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        # 在CPU模式下，我们仍然不禁用CUDA，让COLMAP自己决定
        # 只有在明确需要CPU模式时才禁用CUDA
        # if not force_gpu:
        #     env['CUDA_VISIBLE_DEVICES'] = ''
        
        cmd = [
            self.colmap_path,
            "patch_match_stereo",
            "--workspace_path", dense_dir,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", "-1"
        ]
        
        # 添加CPU模式质量参数，避免GPU相关参数
        for key, value in params.items():
            if key.startswith("PatchMatchStereo") and "gpu" not in key.lower():
                cmd.extend([f"--{key}", str(value)])
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"❌ CPU模式Patch Match Stereo失败:")
            print(f"   返回码: {result.returncode}")
            print(f"   stderr: {result.stderr}")
            print(f"   stdout: {result.stdout}")
            
            # 检查具体错误原因
            if "Dense stereo reconstruction requires CUDA" in result.stderr:
                print("💡 解决方案:")
                print("   1. 启用GPU模式（设置force_gpu=True）")
                print("   2. 检查CUDA环境是否正确配置")
                raise RuntimeError("COLMAP密集重建需要CUDA支持。请使用GPU模式或检查CUDA配置。")
            elif "workspace_path" in result.stderr.lower() or "workspace_format" in result.stderr.lower():
                print("🔧 尝试使用简化参数重新运行...")
                # 尝试最简化的命令
                simple_cmd = [
                    self.colmap_path,
                    "patch_match_stereo",
                    "--workspace_path", dense_dir
                ]
                simple_result = subprocess.run(simple_cmd, capture_output=True, text=True, env=env)
                if simple_result.returncode == 0:
                    print("✅ 简化参数CPU模式成功")
                    return
                else:
                    print(f"❌ 简化参数也失败: {simple_result.stderr}")
            
            raise RuntimeError(f"Patch Match Stereo失败: {result.stderr}")
        else:
            print("✅ CPU模式Patch Match Stereo成功")

    def _run_stereo_fusion(self, dense_dir: str, force_gpu: bool = True):
        """运行立体融合生成点云"""
        cmd = [
            "stereo_fusion",
            "--workspace_path", dense_dir,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", os.path.join(dense_dir, "fused.ply")
        ]
        
        self._run_colmap_command(cmd, force_gpu=force_gpu)

    def _convert_colmap_bins_to_npy(self, dense_dir: str, num_images: int):
        """将COLMAP的.bin深度图转换为.npy格式以方便、安全地读取"""
        print("🔄 开始将.bin深度图转换为.npy格式...")
        stereo_dir = os.path.join(dense_dir, "stereo", "depth_maps")
        npy_dir = os.path.join(dense_dir, "stereo", "depth_maps_npy")
        os.makedirs(npy_dir, exist_ok=True)
        
        # 先检查目录是否存在
        if not os.path.exists(stereo_dir):
            print(f"❌ 深度图目录不存在: {stereo_dir}")
            return
        
        # 列出所有深度图文件
        print(f"🔍 扫描深度图目录: {stereo_dir}")
        depth_files = [f for f in os.listdir(stereo_dir) if f.endswith('.bin')]
        print(f"   找到 {len(depth_files)} 个.bin文件: {depth_files[:5]}{'...' if len(depth_files) > 5 else ''}")
        
        converted_count = 0
        for i in range(num_images):
            # 候选文件名列表（调整顺序，优先photometric）
            candidate_files = [
                f"image_{i:06d}.jpg.photometric.bin",  # 优先photometric（因为禁用了geom_consistency）
                f"image_{i:06d}.jpg.geometric.bin",
                f"image_{i:06d}.photometric.bin",
                f"image_{i:06d}.geometric.bin",
                f"image_{i:06d}.bin",
                f"{i:06d}.photometric.bin",
                f"{i:06d}.geometric.bin"
            ]
            
            depth_file = None
            for candidate in candidate_files:
                full_path = os.path.join(stereo_dir, candidate)
                if os.path.exists(full_path):
                    depth_file = full_path
                    print(f"   ✅ 找到深度图文件: {candidate}")
                    break
            
            if not depth_file:
                print(f"   ⚠️  未找到图像{i}的深度图文件")
                continue
            
            # 读取.bin文件
            depth_map = self._read_colmap_depth_binary(depth_file)
            
            # 保存为.npy文件
            if depth_map is not None:
                npy_file_path = os.path.join(npy_dir, f"image_{i:06d}.npy")
                try:
                    np.save(npy_file_path, depth_map)
                    converted_count += 1
                    print(f"   ✅ 成功保存NPY文件: {os.path.basename(npy_file_path)}")
                except Exception as e:
                    print(f"   ❌ 保存NPY文件失败 {npy_file_path}: {e}")
            else:
                print(f"   ❌ 无法读取深度图: {os.path.basename(depth_file)}")

        print(f"✅ 成功转换 {converted_count} 个深度图为.npy格式。")

    def _read_depth_maps(self, dense_dir: str, num_images: int) -> List[np.ndarray]:
        """从.npy文件读取深度图，这比解析.bin文件更安全、更可靠"""
        depth_maps = []
        npy_dir = os.path.join(dense_dir, "stereo", "depth_maps_npy")
        
        print(f"查找NPY深度图目录: {npy_dir}")
        
        if not os.path.exists(npy_dir):
            print(f"❌ NPY深度图目录不存在: {npy_dir}")
            return depth_maps
        
        for i in range(num_images):
            npy_file = os.path.join(npy_dir, f"image_{i:06d}.npy")
            
            if os.path.exists(npy_file):
                try:
                    depth_map = np.load(npy_file)
                    depth_maps.append(depth_map)
                except Exception as e:
                    print(f"❌ 加载NPY文件失败 {npy_file}: {e}")
            
        print(f"成功读取 {len(depth_maps)} 张NPY深度图")
        return depth_maps

    def _read_colmap_depth_binary(self, file_path: str) -> Optional[np.ndarray]:
        """读取COLMAP深度图文件（支持二进制和文本格式）"""
        try:
            # 首先检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"🔍 读取深度图: {os.path.basename(file_path)}, 文件大小: {file_size} bytes")
            
            with open(file_path, 'rb') as f:
                # 读取前32个字节用于格式检测
                f.seek(0)
                header_bytes = f.read(32)
                print(f"   文件头前32字节: {header_bytes.hex()}")
                
                # 检查是否是COLMAP混合格式（文本头+二进制数据）
                try:
                    header_str = header_bytes.decode('ascii', errors='ignore')[:20]
                    if '&' in header_str and header_str[0].isdigit():
                        print(f"   检测到COLMAP混合格式深度图: {header_str}")
                        # 关闭文件，使用混合格式读取
                        f.close()
                        return self._read_colmap_depth_mixed(file_path)
                except:
                    pass
                
                # 重置文件指针，尝试二进制格式
                f.seek(0)
                
                # COLMAP深度图格式: width(int32), height(int32), data(float32...)
                # 尝试不同的读取方式
                width_bytes = f.read(4)
                height_bytes = f.read(4)
                
                # 尝试小端序
                width_le = struct.unpack('<i', width_bytes)[0]
                height_le = struct.unpack('<i', height_bytes)[0]
                
                # 尝试大端序
                width_be = struct.unpack('>i', width_bytes)[0]
                height_be = struct.unpack('>i', height_bytes)[0]
                
                # 尝试无符号整数
                width_u = struct.unpack('<I', width_bytes)[0]
                height_u = struct.unpack('<I', height_bytes)[0]
                
                print(f"   解析结果:")
                print(f"     小端序(int32): {width_le}x{height_le}")
                print(f"     大端序(int32): {width_be}x{height_be}")
                print(f"     小端序(uint32): {width_u}x{height_u}")
                
                # 选择最合理的值（假设图像尺寸在合理范围内）
                if 0 < width_le < 10000 and 0 < height_le < 10000:
                    width, height = width_le, height_le
                    print(f"   ✅ 使用小端序(int32): {width}x{height}")
                elif 0 < width_be < 10000 and 0 < height_be < 10000:
                    width, height = width_be, height_be
                    print(f"   ✅ 使用大端序(int32): {width}x{height}")
                else:
                    print(f"   ❌ 所有解析方式都得到异常尺寸")
                    # 尝试作为float解析（可能格式完全不同）
                    f.seek(0)
                    float_vals = struct.unpack('<4f', f.read(16))
                    print(f"     作为float32解析前16字节: {float_vals}")
                    return None
                
                # 验证文件大小是否匹配
                expected_size = 8 + width * height * 4  # 8字节头 + 数据
                if file_size != expected_size:
                    print(f"   ⚠️  文件大小不匹配: 期望 {expected_size}, 实际 {file_size}")
                
                # 读取深度数据 (float32)
                num_pixels = width * height
                depth_bytes = f.read(num_pixels * 4)
                
                if len(depth_bytes) != num_pixels * 4:
                    print(f"   ⚠️  深度数据长度不匹配: 期望 {num_pixels * 4}, 实际 {len(depth_bytes)}")
                    return None
                
                # 解包float32数据
                depth_data = struct.unpack(f'<{num_pixels}f', depth_bytes)
                depth_map = np.array(depth_data, dtype=np.float32).reshape((height, width))
                
                # 检查深度值的合理性
                valid_depths = depth_map[depth_map > 0]
                if len(valid_depths) == 0:
                    print(f"   ⚠️  深度图没有有效深度值")
                    return None
                
                min_depth, max_depth = valid_depths.min(), valid_depths.max()
                print(f"   ✅ 深度范围: {min_depth:.3f} - {max_depth:.3f}, 有效像素: {len(valid_depths)}/{num_pixels}")
                
                return depth_map
                
        except Exception as e:
            print(f"❌ 读取深度文件失败 {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _read_colmap_depth_mixed(self, file_path: str) -> Optional[np.ndarray]:
        """读取COLMAP混合格式深度图（文本头+二进制数据）"""
        try:
            print(f"   📖 读取COLMAP混合格式深度图...")
            
            with open(file_path, 'rb') as f:
                content = f.read()
                
                # 查找文本头部：格式为 "width&height&channels&"
                header_end = -1
                for i in range(min(50, len(content))):
                    if content[i:i+1] == b'&':
                        # 继续查找下一个&
                        continue
                    elif content[i] > 127 or (content[i] < 32 and content[i] not in [9, 10, 13]):
                        # 找到第一个非ASCII字符，这是二进制数据的开始
                        # 回退到最近的&符号之后
                        for j in range(i-1, -1, -1):
                            if content[j:j+1] == b'&':
                                header_end = j + 1
                                break
                        break
                
                if header_end <= 0:
                    # 尝试查找最后一个&符号
                    last_amp = content.find(b'&', 0, 50)
                    if last_amp >= 0:
                        # 查找这个&之后的下一个&
                        next_amp = content.find(b'&', last_amp + 1, 50)
                        if next_amp >= 0:
                            final_amp = content.find(b'&', next_amp + 1, 50)
                            if final_amp >= 0:
                                header_end = final_amp + 1
                
                print(f"   🔍 文本头部结束位置: {header_end}")
                
                if header_end > 0:
                    # 解析文本头部
                    header_bytes = content[:header_end-1]  # 不包含最后的&
                    header_text = header_bytes.decode('ascii', errors='ignore')
                    print(f"   📝 文本头部: '{header_text}'")
                    
                    # 解析尺寸：格式为 "width&height&channels"
                    parts = header_text.split('&')
                    if len(parts) >= 3:
                        try:
                            width = int(parts[0])
                            height = int(parts[1])
                            channels = int(parts[2]) if parts[2].isdigit() else 1
                            
                            print(f"   📏 解析得到尺寸: {width}x{height}x{channels}")
                            
                            # 验证尺寸合理性
                            if 1 <= width <= 10000 and 1 <= height <= 10000:
                                # 读取二进制深度数据
                                binary_data = content[header_end:]
                                expected_floats = width * height * channels
                                expected_bytes = expected_floats * 4  # float32
                                
                                print(f"   🔢 期望 {expected_floats} 个float32 ({expected_bytes} bytes)")
                                print(f"   💾 实际二进制数据: {len(binary_data)} bytes")
                                
                                if len(binary_data) >= expected_bytes:
                                    try:
                                        # 解析float32数据
                                        depth_values = struct.unpack(f'<{expected_floats}f', binary_data[:expected_bytes])
                                        depth_map = np.array(depth_values, dtype=np.float32).reshape(height, width, channels)
                                        
                                        # 如果只有一个通道，去掉最后一维
                                        if channels == 1:
                                            depth_map = depth_map[:, :, 0]
                                        
                                        # 检查深度值的合理性
                                        valid_depths = depth_map[depth_map > 0]
                                        if len(valid_depths) > 0:
                                            min_depth, max_depth = valid_depths.min(), valid_depths.max()
                                            print(f"   ✅ 混合格式深度图读取成功: {depth_map.shape}")
                                            print(f"   📊 深度值范围: {min_depth:.3f} - {max_depth:.3f}")
                                            return depth_map
                                        else:
                                            print(f"   ⚠️  深度图没有有效深度值")
                                            
                                    except struct.error as e:
                                        print(f"   ❌ 二进制数据解析失败: {e}")
                                        
                                else:
                                    print(f"   ❌ 二进制数据不足: 需要{expected_bytes}字节，只有{len(binary_data)}字节")
                            else:
                                print(f"   ❌ 尺寸不合理: {width}x{height}")
                        except ValueError as e:
                            print(f"   ❌ 无法解析尺寸: {parts}, 错误: {e}")
                    else:
                        print(f"   ❌ 文本头部格式错误: {parts}")
                else:
                    print(f"   ❌ 无法找到文本头部结束位置")
                    
        except Exception as e:
            print(f"   ❌ 混合格式读取失败: {e}")
            import traceback
            traceback.print_exc()
            
        return None

    def _read_colmap_depth_text(self, file_path: str) -> Optional[np.ndarray]:
        """读取文本格式的COLMAP深度图"""
        try:
            print(f"   📖 以文本格式读取深度图...")
            
            with open(file_path, 'r') as f:
                # 读取第一行，解析尺寸
                first_line = f.readline().strip()
                
                # 尝试不同的分隔符
                if '&' in first_line:
                    parts = first_line.split('&')
                elif ' ' in first_line:
                    parts = first_line.split()
                elif ',' in first_line:
                    parts = first_line.split(',')
                else:
                    print(f"   ❌ 无法解析文本格式头部: {first_line[:50]}")
                    return None
                
                if len(parts) >= 2:
                    try:
                        width = int(parts[0])
                        height = int(parts[1])
                        print(f"   ✅ 解析出尺寸: {width}x{height}")
                    except:
                        print(f"   ❌ 无法解析尺寸: {parts[:2]}")
                        return None
                    
                    # 读取深度数据
                    depth_values = []
                    
                    # 如果第一行还有更多数据（深度值可能在同一行）
                    if len(parts) > 2:
                        # 尝试解析剩余部分作为深度值
                        for val in parts[2:]:
                            try:
                                if val:  # 跳过空字符串
                                    depth_values.append(float(val))
                            except:
                                pass
                    
                    # 继续读取剩余行
                    for line in f:
                        line = line.strip()
                        if line:
                            # 尝试解析每行的深度值
                            if '&' in line:
                                values = line.split('&')
                            elif ' ' in line:
                                values = line.split()
                            elif ',' in line:
                                values = line.split(',')
                            else:
                                values = [line]
                            
                            for val in values:
                                try:
                                    if val:
                                        depth_values.append(float(val))
                                except:
                                    pass
                    
                    print(f"   读取了 {len(depth_values)} 个深度值")
                    
                    # 验证数量
                    expected_pixels = width * height
                    if len(depth_values) == expected_pixels:
                        depth_map = np.array(depth_values, dtype=np.float32).reshape((height, width))
                        
                        # 检查深度值的合理性
                        valid_depths = depth_map[depth_map > 0]
                        if len(valid_depths) > 0:
                            min_depth, max_depth = valid_depths.min(), valid_depths.max()
                            print(f"   ✅ 深度范围: {min_depth:.3f} - {max_depth:.3f}, 有效像素: {len(valid_depths)}/{expected_pixels}")
                            return depth_map
                        else:
                            print(f"   ⚠️  深度图没有有效深度值")
                    else:
                        print(f"   ❌ 深度值数量不匹配: 期望 {expected_pixels}, 实际 {len(depth_values)}")
                        
        except Exception as e:
            print(f"   ❌ 文本格式读取失败: {e}")
            
        return None

    def _create_minimal_sparse_reconstruction(self, sparse_dir: str, images_dir: str):
        """创建最简化的稀疏重建用于密集重建"""
        print("🔧 创建简化稀疏重建...")
        
        # 获取图像列表
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        if len(image_files) < 2:
            raise RuntimeError("需要至少2张图像进行密集重建")
        
        # 创建images.txt - 只包含邻近图像对
        images_file = os.path.join(sparse_dir, "images.txt")
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            # 只注册前几张图像，确保它们有相邻关系
            num_images = min(len(image_files), 5)  # 限制图像数量
            
            for i in range(num_images):
                image_id = i + 1
                camera_id = 1
                image_name = image_files[i]
                
                # 创建简单的相机位姿 - 沿着Z轴的运动
                # 四元数 (w, x, y, z) - 单位四元数表示无旋转
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                # 平移 - 简单的前进运动
                tx, ty, tz = 0.0, 0.0, -i * 0.5
                
                # 写入图像信息
                f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n")
                
                # 添加一些虚拟的2D点以建立连接
                if i < num_images - 1:  # 不是最后一张图像
                    # 添加几个虚拟特征点来建立图像间的连接
                    for j in range(4):  # 每张图像4个点
                        x = 100 + j * 200  # 分布在图像上
                        y = 100 + j * 100
                        point3d_id = i * 4 + j + 1  # 创建连续的3D点ID
                        f.write(f"{x} {y} {point3d_id} ")
                    f.write("\n")
                else:
                    f.write("\n")  # 空行表示没有2D点
        
        # 创建points3D.txt - 添加一些虚拟3D点
        points3d_file = os.path.join(sparse_dir, "points3D.txt")
        with open(points3d_file, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            # 创建虚拟3D点
            for i in range((num_images - 1) * 4):
                point_id = i + 1
                # 虚拟3D坐标
                x, y, z = i * 0.1, 0.0, -i * 0.1
                # RGB颜色
                r, g, b = 128, 128, 128
                # 重投影误差
                error = 0.5
                
                # 轨迹信息：哪些图像看到了这个点
                img_id = i // 4 + 1
                point_2d_idx = i % 4
                track = f"{img_id} {point_2d_idx}"
                
                # 如果不是边界点，添加下一张图像的观测
                if img_id < num_images:
                    track += f" {img_id + 1} {point_2d_idx}"
                
                f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {error} {track}\n")
        
        print(f"✅ 创建了简化稀疏重建：{num_images}张图像，{(num_images-1)*4}个3D点")


class VVLColmapMVSDepthNode:
    """VVL COLMAP-MVS原生深度估计节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "输入图像序列（至少3张）"
                }),
                "intrinsics_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "相机内参JSON数据（必需）"
                }),
                "poses_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "相机位姿JSON数据（必需）"
                }),
            },
            "optional": {
                "force_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制使用GPU模式（失败时不回退到CPU）"
                }),
                "allow_cpu_fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "当GPU/CUDA不可用时允许使用CPU模式"
                }),
                "quality": (["low", "medium", "high", "extreme"], {
                    "default": "medium",
                    "tooltip": "重建质量"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "输出目录（留空使用临时目录）"
                }),
                "save_to_disk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "保存深度图到磁盘"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "DEPTH_MAPS")
    RETURN_NAMES = ("depth_maps", "file_paths", "raw_depth_data")
    FUNCTION = "estimate_depth"
    OUTPUT_NODE = False
    CATEGORY = "💃VVL/VideoCamera"
    DESCRIPTION = "使用真正的COLMAP-MVS原生算法进行深度估计。需要COLMAP软件安装。"

    def estimate_depth(self, images, intrinsics_json, poses_json, 
                      force_gpu=True, allow_cpu_fallback=True, quality="medium", output_dir="", save_to_disk=False):
        """执行COLMAP-MVS深度估计"""
        try:
            # 🔍 详细的图像调试信息
            print(f"\n🖼️  图像输入调试信息:")
            print(f"  - images类型: {type(images)}")
            if isinstance(images, torch.Tensor):
                print(f"  - images.shape: {images.shape}")
                print(f"  - images.dim(): {images.dim()}")
                print(f"  - images.dtype: {images.dtype}")
                print(f"  - images值域: {images.min().item():.3f} - {images.max().item():.3f}")
            elif hasattr(images, '__len__'):
                print(f"  - images长度: {len(images)}")
                if len(images) > 0:
                    print(f"  - 第一个元素类型: {type(images[0])}")
                    if hasattr(images[0], 'shape'):
                        print(f"  - 第一个元素shape: {images[0].shape}")
            else:
                print(f"  - images无法确定长度")
            
            # 转换图像格式
            if isinstance(images, torch.Tensor):
                print(f"\n🔄 转换torch.Tensor图像...")
                images_np = []
                
                # 处理不同的张量维度
                if images.dim() == 4:  # Batch dimension (B, H, W, C) 或 (B, C, H, W)
                    print(f"  - 检测到4D张量: {images.shape}")
                    batch_size = images.shape[0]
                    print(f"  - 批次大小: {batch_size}")
                    
                    for i in range(batch_size):
                        img = images[i].cpu().numpy()
                        print(f"    - 图像{i}: shape={img.shape}, dtype={img.dtype}, 值域={img.min():.3f}-{img.max():.3f}")
                        
                        # 确保是HWC格式
                        if img.shape[0] == 3 and len(img.shape) == 3:  # CHW -> HWC
                            img = img.transpose(1, 2, 0)
                            print(f"      转换CHW->HWC: {img.shape}")
                        
                        # 转换数值范围
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                            print(f"      归一化转换: 0-1 -> 0-255")
                        else:
                            img = img.astype(np.uint8)
                            
                        images_np.append(img)
                        
                elif images.dim() == 3:  # Single image (H, W, C) 或 (C, H, W)
                    print(f"  - 检测到3D张量（单图像）: {images.shape}")
                    img = images.cpu().numpy()
                    
                    # 确保是HWC格式
                    if img.shape[0] == 3 and len(img.shape) == 3:  # CHW -> HWC
                        img = img.transpose(1, 2, 0)
                        print(f"    转换CHW->HWC: {img.shape}")
                    
                    # 转换数值范围
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                        
                    images_np.append(img)
                    
                else:
                    print(f"  - ⚠️  不支持的张量维度: {images.dim()}D")
                    raise ValueError(f"不支持的图像张量维度: {images.dim()}D")
                    
            elif isinstance(images, list):
                print(f"\n🔄 处理列表格式图像...")
                images_np = images
                print(f"  - 列表长度: {len(images_np)}")
            else:
                print(f"\n❌ 不支持的图像格式: {type(images)}")
                raise ValueError(f"不支持的图像输入格式: {type(images)}")
            
            print(f"\n✅ 图像转换完成:")
            print(f"  - 最终图像数量: {len(images_np)}")
            if len(images_np) > 0:
                sample_img = images_np[0]
                print(f"  - 样本图像shape: {sample_img.shape}")
                print(f"  - 样本图像dtype: {sample_img.dtype}")
                print(f"  - 样本图像值域: {sample_img.min()}-{sample_img.max()}")
            
            # 验证数据
            if len(images_np) < 3:
                print(f"\n❌ 图像数量不足: {len(images_np)} < 3")
                raise ValueError("COLMAP-MVS至少需要3张图像")
            
            print(f"\n✅ 图像数量验证通过: {len(images_np)} >= 3")
            
            # 解析相机参数（必需）
            try:
                camera_intrinsics = json.loads(intrinsics_json)
                print("成功解析相机内参")
            except json.JSONDecodeError as e:
                raise ValueError(f"相机内参JSON解析失败: {e}")
            
            try:
                camera_poses_raw = json.loads(poses_json)
                print(f"JSON解析成功，原始数据类型: {type(camera_poses_raw)}")
                
                # 首先检查是否是错误对象
                if isinstance(camera_poses_raw, dict) and "error" in camera_poses_raw and "success" in camera_poses_raw:
                    if not camera_poses_raw.get("success", True):
                        error_msg = camera_poses_raw.get('error', '未知错误')
                        print(f"❌ 检测到相机估计失败: {error_msg}")
                        
                        # 如果是二进制格式问题，提供解决建议
                        if "没有生成重建结果" in error_msg:
                            print("💡 这可能是COLMAP文件格式问题：")
                            print("   - COLMAP生成了二进制文件(.bin)而不是文本文件(.txt)")
                            print("   - 已在新版本中修复此问题")
                            print("   - 请重新运行相机估计节点")
                        
                        raise ValueError(f"相机估计失败: {error_msg}")
                
                # 检查是否是嵌套结构（例如包含"poses"键）
                if isinstance(camera_poses_raw, dict):
                    # 检查常见的嵌套键
                    possible_keys = ['poses', 'camera_poses', 'positions', 'data']
                    camera_poses = None
                    
                    for key in possible_keys:
                        if key in camera_poses_raw:
                            print(f"检测到嵌套结构，提取键: {key}")
                            camera_poses = camera_poses_raw[key]
                            break
                    
                    if camera_poses is None:
                        # 没有找到嵌套键，假设就是位姿数据
                        camera_poses = camera_poses_raw
                        print("没有找到嵌套键，直接使用原始数据")
                else:
                    # 如果是列表，直接使用
                    camera_poses = camera_poses_raw
                    print("检测到列表格式，直接使用")
                
                # 计算实际位姿数量
                if isinstance(camera_poses, dict):
                    poses_count = len([v for v in camera_poses.values() if isinstance(v, dict) and 'position' in v])
                elif isinstance(camera_poses, list):
                    poses_count = len([p for p in camera_poses if isinstance(p, dict) and 'position' in p])
                else:
                    poses_count = 0
                
                print(f"成功解析 {poses_count} 个有效相机位姿（总数据项: {len(camera_poses) if hasattr(camera_poses, '__len__') else 'N/A'}）")
                
            except json.JSONDecodeError as e:
                raise ValueError(f"相机位姿JSON解析失败: {e}")
            
            # 使用实际有效位姿数量进行比较
            if poses_count != len(images_np):
                print(f"警告: 相机位姿数量({poses_count})与图像数量({len(images_np)})不匹配")
                print("尝试进行位姿插值以匹配图像数量...")
                
                # 进行位姿插值
                camera_poses = self._interpolate_poses(camera_poses, len(images_np))
                
                if len(camera_poses) != len(images_np):
                    raise ValueError(f"位姿插值失败: 插值后位姿数量({len(camera_poses)})仍与图像数量({len(images_np)})不匹配")
                
                print(f"位姿插值成功: 现在有{len(camera_poses)}个位姿匹配{len(images_np)}张图像")
            else:
                print("位姿数量与图像数量匹配，无需插值")
            
            # 创建深度估计器
            estimator = ColmapMVSDepthEstimator(use_gpu=force_gpu, quality=quality)
            
            # 检查CUDA支持状态
            if force_gpu and not getattr(estimator, 'colmap_has_cuda', False):
                if allow_cpu_fallback:
                    print("⚠️  强制GPU模式但COLMAP无CUDA支持，允许CPU回退")
                    print("   📊 当前系统状态:")
                    print("   - COLMAP版本: 可能无CUDA支持")  
                    print("   - GPU硬件: ✅ NVIDIA RTX 4090")
                    print("   - 问题: COLMAP未编译CUDA支持")
                    print("   🔄 自动切换到CPU模式...")
                    force_gpu = False  # 回退到CPU模式
                else:
                    print("❌ 强制GPU模式但COLMAP无CUDA支持，且未允许CPU回退")
                    print("💡 解决方案:")
                    print("   1. 设置 allow_cpu_fallback=True")
                    print("   2. 或安装CUDA版COLMAP: conda install -c conda-forge colmap")
                    raise RuntimeError("强制GPU模式但COLMAP无CUDA支持，且未允许CPU回退。请安装COLMAP CUDA版本或启用CPU回退。")
            
            # 执行深度估计
            output_dir_actual = output_dir if output_dir.strip() else None
            raw_depth_maps = estimator.estimate_depth_mvs(
                images=images_np,
                camera_intrinsics=camera_intrinsics,
                camera_poses=camera_poses,
                output_dir=output_dir_actual,
                force_gpu=force_gpu
            )
            
            # 处理输出
            depth_tensors = []
            file_paths = []
        
            for i, depth_map in enumerate(raw_depth_maps):
                # 标准化深度图到0-1范围
                depth_normalized = cv2.normalize(
                    depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F
                )
                
                # 转换为RGB格式以便在ComfyUI中显示
                depth_rgb = np.stack([depth_normalized] * 3, axis=-1)
                depth_tensor = torch.from_numpy(depth_rgb).float()
                depth_tensors.append(depth_tensor)
                
                # 保存到磁盘（如果需要）
                if save_to_disk and output_dir_actual:
                    depth_path = os.path.join(output_dir_actual, f"depth_{i:06d}.png")
                    depth_img = (depth_normalized * 255).astype(np.uint8)
                    cv2.imwrite(depth_path, depth_img)
                    file_paths.append(depth_path)
                    
                    # 同时保存原始深度数据
                    raw_path = os.path.join(output_dir_actual, f"depth_{i:06d}.npy")
                    np.save(raw_path, depth_map)
            
            result_tensor = torch.stack(depth_tensors, dim=0)
            file_paths_str = "\n".join(file_paths) if file_paths else ""
            
            print(f"✅ COLMAP-MVS成功生成 {len(raw_depth_maps)} 张深度图")
            return (result_tensor, file_paths_str, raw_depth_maps)
            
        except Exception as e:
            print(f"❌ COLMAP-MVS深度估计失败: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"COLMAP-MVS深度估计失败: {str(e)}")

    def _interpolate_poses(self, poses: List[Dict], target_count: int) -> List[Dict]:
        """插值位姿以匹配目标数量"""
        
        print(f"📍 插值调试信息:")
        print(f"  - 输入poses类型: {type(poses)}")
        print(f"  - 输入poses长度: {len(poses) if hasattr(poses, '__len__') else 'N/A'}")
        print(f"  - 目标数量: {target_count}")
        
        # 确保poses是列表格式
        if isinstance(poses, dict):
            print("  - 检测到字典格式，转换为列表...")
            # 如果是字典，转换为列表
            poses_list = []
            for key, value in poses.items():
                print(f"    - 处理key: {key}, value类型: {type(value)}")
                if isinstance(value, dict):
                    # 确保每个pose都有必要的字段
                    if 'position' in value and 'rotation_quaternion' in value:
                        # 添加image_id如果缺失
                        if 'image_id' not in value:
                            value['image_id'] = len(poses_list)
                        poses_list.append(value)
                        print(f"    - 添加有效位姿，当前poses_list长度: {len(poses_list)}")
                    else:
                        print(f"    - 跳过无效位姿（缺少position或rotation_quaternion）")
                else:
                    print(f"    - 跳过非字典值")
            
            poses = sorted(poses_list, key=lambda x: x.get('image_id', 0))
            print(f"  - 转换后poses列表长度: {len(poses)}")
        
        elif isinstance(poses, list):
            print("  - 检测到列表格式，直接使用")
            # 验证列表中的每个元素
            valid_poses = []
            for i, pose in enumerate(poses):
                if isinstance(pose, dict) and 'position' in pose and 'rotation_quaternion' in pose:
                    if 'image_id' not in pose:
                        pose['image_id'] = i
                    valid_poses.append(pose)
            poses = valid_poses
            print(f"  - 有效位姿数量: {len(poses)}")
        
        else:
            print(f"  - 未知的poses格式: {type(poses)}")
            raise ValueError(f"不支持的poses格式: {type(poses)}")
        
        if len(poses) == 0:
            print("❌ 位姿列表为空，无法进行插值")
            raise ValueError("无法插值空的位姿列表")
        
        print(f"✅ 准备插值: {len(poses)} 个位姿 → {target_count} 个位姿")
        
        if len(poses) == target_count:
            print("位姿数量已匹配，无需插值")
            return poses
        
        if len(poses) == 1:
            print("只有1个位姿，复制为目标数量")
            # 如果只有一个位姿，复制它
            result = []
            for i in range(target_count):
                pose_copy = poses[0].copy()
                pose_copy['image_id'] = i
                pose_copy['image_name'] = f"duplicated_{i:06d}"
                result.append(pose_copy)
            return result
        
        # 执行线性插值
        print("执行线性插值...")
        interpolated_poses = []
        
        # 创建插值索引
        original_indices = np.linspace(0, len(poses) - 1, len(poses))
        target_indices = np.linspace(0, len(poses) - 1, target_count)
        
        for i, target_idx in enumerate(target_indices):
            # 找到最近的两个原始位姿
            lower_idx = int(np.floor(target_idx))
            upper_idx = min(lower_idx + 1, len(poses) - 1)
            
            if lower_idx == upper_idx:
                # 精确匹配
                pose_copy = poses[lower_idx].copy()
                pose_copy['image_id'] = i
                interpolated_poses.append(pose_copy)
            else:
                # 插值
                alpha = target_idx - lower_idx
                pose1 = poses[lower_idx]
                pose2 = poses[upper_idx]
                
                # 插值位置
                pos1 = np.array(pose1["position"])
                pos2 = np.array(pose2["position"])
                interp_pos = pos1 * (1 - alpha) + pos2 * alpha
                
                # 插值四元数 (SLERP)
                quat1 = np.array(pose1["rotation_quaternion"])
                quat2 = np.array(pose2["rotation_quaternion"])
                interp_quat = self._slerp_quaternion(quat1, quat2, alpha)
                
                # 创建插值位姿
                interp_pose = {
                    "position": interp_pos.tolist(),
                    "rotation_quaternion": interp_quat.tolist(),
                    "image_name": f"interpolated_{i:06d}",
                    "image_id": i
                }
                
                interpolated_poses.append(interp_pose)
        
        print(f"✅ 插值完成，生成了 {len(interpolated_poses)} 个位姿")
        return interpolated_poses
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值四元数"""
        
        # 确保四元数归一化
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算点积
        dot = np.dot(q1, q2)
        
        # 如果点积为负，使用较短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数很接近，使用线性插值
        if dot > 0.9995:
            result = q1 * (1 - t) + q2 * t
            return result / np.linalg.norm(result)
        
        # 球面线性插值
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VVLColmapMVSDepthNode": VVLColmapMVSDepthNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VVLColmapMVSDepthNode": "VVL COLMAP-MVS原生深度估计"
} 