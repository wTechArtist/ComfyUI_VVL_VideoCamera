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
    """使用原生COLMAP进行相机参数估计的核心类"""

    def __init__(self):
        self.check_dependencies()

    def check_dependencies(self):
        """检查原生COLMAP依赖（必需）"""
        # 检查原生COLMAP是否可用（必需）
        try:
            import subprocess
            result = subprocess.run(['colmap', 'help'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.colmap_available = True
                print("✓ 原生COLMAP 已成功检测到")
                
                # 检测COLMAP版本和GPU支持
                self._detect_colmap_capabilities()
            else:
                raise RuntimeError("原生COLMAP命令执行失败")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            raise RuntimeError(f"未找到原生COLMAP。请安装COLMAP并确保其在系统PATH中。错误: {e}")
        
        # 检查PyColmap（仅用于读取结果文件）
        try:
            import pycolmap
            self.pycolmap = pycolmap
            print("✓ PyColmap 可用于读取COLMAP结果文件")
        except ImportError as e:
            raise ImportError(f"PyColmap 是必需的依赖（用于读取COLMAP结果）: {e}")

    def _detect_colmap_capabilities(self):
        """检测COLMAP的版本和功能"""
        import subprocess
        
        # 检测版本
        try:
            result = subprocess.run(['colmap'], capture_output=True, text=True, timeout=5)
            self.colmap_version = "unknown"
            if "COLMAP" in result.stderr:
                # 尝试提取版本信息
                for line in result.stderr.split('\n'):
                    if 'COLMAP' in line and ('3.' in line or '4.' in line):
                        self.colmap_version = line.strip()
                        break
            print(f"COLMAP版本: {self.colmap_version}")
        except:
            self.colmap_version = "unknown"
        
        # 检测GPU支持
        self.gpu_available = self._check_gpu_support()
        print(f"GPU支持: {'可用' if self.gpu_available else '不可用'}")

    def _check_gpu_support(self) -> bool:
        """检测GPU支持"""
        try:
            import subprocess
            # 尝试运行简单的GPU测试
            result = subprocess.run([
                'colmap', 'feature_extractor', '--help'
            ], capture_output=True, text=True, timeout=5)
            
            # 检查帮助信息中是否包含GPU相关选项
            return 'gpu' in result.stdout.lower() or 'cuda' in result.stdout.lower()
        except:
            return False

    def estimate_from_images(self, images: List[torch.Tensor], 
                           colmap_feature_type: str = "sift",
                           colmap_matcher_type: str = "sequential", 
                           colmap_quality: str = "medium",
                           enable_dense_reconstruction: bool = False,
                           force_gpu: bool = True) -> Dict:
        """使用原生COLMAP从图片序列估计相机参数"""
        
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

        print("使用原生COLMAP进行相机参数估计...")
        print(f"强制GPU模式: {'是' if force_gpu else '否'}")
        
        return self._estimate_with_native_colmap(
            images, colmap_feature_type, colmap_matcher_type, 
            colmap_quality, enable_dense_reconstruction, force_gpu
        )

    def _estimate_with_native_colmap(self, images: List[torch.Tensor], 
                                   colmap_feature_type: str,
                                   colmap_matcher_type: str, 
                                   colmap_quality: str,
                                   enable_dense_reconstruction: bool,
                                   force_gpu: bool) -> Dict:
        """使用原生COLMAP进行估计"""
        
        import subprocess
        
        # 创建临时工作目录
        temp_dir = tempfile.mkdtemp(prefix="colmap_native_")
        images_dir = os.path.join(temp_dir, "images")
        database_path = os.path.join(temp_dir, "database.db")
        sparse_dir = os.path.join(temp_dir, "sparse")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(sparse_dir, exist_ok=True)

            # 1. 保存图片到临时目录
            print(f"保存 {len(images)} 张图片到临时目录...")
            image_paths = self._save_images_to_temp(images, images_dir)
            
            # 2. 设置质量参数
            quality_settings = self._get_quality_settings(colmap_quality)
            
            # 3. 特征提取 (原生COLMAP命令)
            print("开始原生COLMAP特征提取...")
            self._run_native_feature_extraction(database_path, images_dir, quality_settings, force_gpu)
            
            # 4. 特征匹配 (原生COLMAP命令)
            print(f"开始原生COLMAP特征匹配 ({colmap_matcher_type})...")
            self._run_native_feature_matching(database_path, colmap_matcher_type, force_gpu)
            
            # 5. 增量重建 (原生COLMAP命令)
            print("开始原生COLMAP增量重建...")
            self._run_native_mapping(database_path, images_dir, sparse_dir, force_gpu)
            
            # 6. 读取重建结果
            reconstruction = self._read_native_reconstruction(sparse_dir)
            
            if reconstruction is None:
                return {
                    "success": False,
                    "error": "原生COLMAP 重建失败：没有生成重建结果",
                    "intrinsics": None,
                    "poses": [],
                    "statistics": {},
                    "frame_count": 0,
                    "point_cloud": {}
                }
            
            # 7. 解析结果
            result = self._parse_native_reconstruction(reconstruction, len(images))
            
            print(f"原生COLMAP重建成功：注册了 {len(result['poses'])} 张图像")
            
            return result
            
        except Exception as e:
            print(f"原生COLMAP 估计过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": f"原生COLMAP 重建失败: {str(e)}",
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

    def _setup_gpu_environment(self) -> Dict[str, str]:
        """设置GPU环境变量"""
        import os
        
        env = os.environ.copy()
        
        # 强制使用GPU
        env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
        
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
        
        return env

    def _run_native_feature_extraction(self, database_path: str, images_dir: str, quality_settings: Dict, force_gpu: bool = True):
        """运行原生COLMAP特征提取"""
        import subprocess
        import os
        
        if force_gpu and self.gpu_available:
            print("🚀 使用强制GPU模式运行COLMAP特征提取...")
            
            # 设置GPU环境
            env = self._setup_gpu_environment()
            
            # GPU模式命令 - 使用更简洁的参数
            gpu_cmd = [
                'colmap', 'feature_extractor',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--ImageReader.single_camera', '1',
                '--SiftExtraction.use_gpu', '1',
                '--SiftExtraction.gpu_index', '0',  # 明确指定GPU 0
                '--SiftExtraction.max_image_size', str(quality_settings["max_image_size"]),
                '--SiftExtraction.max_num_features', str(quality_settings["max_num_features"])
            ]
            
            # 首先尝试直接GPU模式
            print(f"执行命令: {' '.join(gpu_cmd)}")
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print("✅ GPU模式特征提取成功")
                return
            
            print(f"直接GPU模式失败: {result.stderr}")
            
            # 尝试使用nvidia-smi检查GPU状态
            try:
                gpu_check = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if gpu_check.returncode == 0:
                    print("GPU检测正常，尝试使用xvfb-run包装...")
                    
                    # 使用xvfb-run + GPU
                    xvfb_gpu_cmd = [
                        'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
                    ] + gpu_cmd
                    
                    result = subprocess.run(xvfb_gpu_cmd, capture_output=True, text=True, env=env)
                    
                    if result.returncode == 0:
                        print("✅ xvfb-run + GPU模式特征提取成功")
                        return
                    
                    print(f"xvfb-run + GPU模式也失败: {result.stderr}")
            except:
                print("无法检查GPU状态")
        
        # 如果GPU模式失败，抛出异常（因为用户要求必须使用GPU）
        if force_gpu:
            raise RuntimeError("强制GPU模式失败，无法继续。请检查GPU驱动和CUDA安装。")
        
        # 备用CPU模式（仅在不强制GPU时使用）
        print("回退到CPU模式...")
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        cpu_cmd = [
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
            '--SiftExtraction.max_image_size', str(quality_settings["max_image_size"]),
            '--SiftExtraction.max_num_features', str(quality_settings["max_num_features"])
        ]
        
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP特征提取失败: {result.stderr}")

    def _run_native_feature_matching(self, database_path: str, matcher_type: str, force_gpu: bool = True):
        """运行原生COLMAP特征匹配"""
        import subprocess
        import os
        
        # 构建基础命令
        if matcher_type == "exhaustive":
            base_cmd = ['colmap', 'exhaustive_matcher', '--database_path', database_path]
        elif matcher_type == "sequential":
            base_cmd = ['colmap', 'sequential_matcher', '--database_path', database_path, '--SequentialMatching.overlap', '10']
        elif matcher_type == "spatial":
            base_cmd = ['colmap', 'spatial_matcher', '--database_path', database_path]
        else:
            raise ValueError(f"不支持的匹配器类型: {matcher_type}")
        
        if force_gpu and self.gpu_available:
            print("🚀 使用强制GPU模式运行COLMAP特征匹配...")
            
            # 设置GPU环境
            env = self._setup_gpu_environment()
            
            # GPU模式
            gpu_cmd = base_cmd + [
                '--SiftMatching.use_gpu', '1',
                '--SiftMatching.gpu_index', '0'
            ]
            
            # 直接GPU模式
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print("✅ GPU模式特征匹配成功")
                return
            
            print(f"直接GPU模式失败: {result.stderr}")
            
            # 尝试xvfb-run + GPU
            try:
                xvfb_gpu_cmd = [
                    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
                ] + gpu_cmd
                
                result = subprocess.run(xvfb_gpu_cmd, capture_output=True, text=True, env=env)
                
                if result.returncode == 0:
                    print("✅ xvfb-run + GPU模式特征匹配成功")
                    return
                
                print(f"xvfb-run + GPU模式也失败: {result.stderr}")
            except:
                pass
        
        # 如果GPU模式失败，抛出异常（因为用户要求必须使用GPU）
        if force_gpu:
            raise RuntimeError("强制GPU模式失败，无法继续。请检查GPU驱动和CUDA安装。")
        
        # 备用CPU模式
        print("回退到CPU模式...")
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        cpu_cmd = base_cmd + ['--SiftMatching.use_gpu', '0']
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP特征匹配失败: {result.stderr}")

    def _run_native_mapping(self, database_path: str, images_dir: str, sparse_dir: str, force_gpu: bool = True):
        """运行原生COLMAP增量重建"""
        import subprocess
        import os
        
        if force_gpu and self.gpu_available:
            print("🚀 使用强制GPU模式运行COLMAP重建...")
            
            # 设置GPU环境
            env = self._setup_gpu_environment()
            
            # 使用更宽松的重建参数
            gpu_cmd = [
                'colmap', 'mapper',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--output_path', sparse_dir,
                '--Mapper.init_min_num_inliers', '5',  # 非常低的阈值
                '--Mapper.min_num_matches', '5',       # 非常低的匹配要求
                '--Mapper.max_num_models', '100',      # 允许更多模型
                '--Mapper.init_min_tri_angle', '1.0',  # 降低三角化角度要求
                '--Mapper.multiple_models', '1',       # 允许多模型
                '--Mapper.extract_colors', '0'         # 关闭颜色提取加速
            ]
            
            print(f"执行GPU重建命令: {' '.join(gpu_cmd)}")
            
            # 直接GPU模式
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, env=env)
            
            print(f"COLMAP mapper 返回码: {result.returncode}")
            if result.stdout:
                print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            
            if result.returncode == 0:
                print("✅ GPU模式重建成功")
                self._check_reconstruction_output(sparse_dir)
                return
            
            print(f"GPU模式失败: {result.stderr}")
            
            # 如果是参数不识别的错误，尝试更基础的参数
            if "unrecognised option" in result.stderr:
                print("检测到参数不兼容，尝试基础GPU参数...")
                basic_gpu_cmd = [
                    'colmap', 'mapper',
                    '--database_path', database_path,
                    '--image_path', images_dir,
                    '--output_path', sparse_dir
                ]
                
                print(f"执行基础GPU重建命令: {' '.join(basic_gpu_cmd)}")
                result = subprocess.run(basic_gpu_cmd, capture_output=True, text=True, env=env)
                
                print(f"基础GPU mapper 返回码: {result.returncode}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                
                if result.returncode == 0:
                    print("✅ GPU模式基础参数重建成功")
                    self._check_reconstruction_output(sparse_dir)
                    return
            
            # 尝试xvfb-run + GPU
            try:
                print("尝试xvfb-run + GPU模式...")
                xvfb_gpu_cmd = [
                    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24 -ac +extension GLX +render -noreset'
                ] + gpu_cmd
                
                result = subprocess.run(xvfb_gpu_cmd, capture_output=True, text=True, env=env)
                
                print(f"xvfb GPU mapper 返回码: {result.returncode}")
                if result.stdout:
                    print(f"STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                
                if result.returncode == 0:
                    print("✅ xvfb-run + GPU模式重建成功")
                    self._check_reconstruction_output(sparse_dir)
                    return
            except Exception as e:
                print(f"xvfb-run执行失败: {e}")
        
        # 如果GPU模式失败，抛出异常（因为用户要求必须使用GPU）
        if force_gpu:
            print("❌ 所有GPU模式尝试都失败了")
            self._check_reconstruction_output(sparse_dir)
            raise RuntimeError("强制GPU模式失败，无法继续。请检查图像质量、特征匹配结果或考虑使用CPU模式。")
        
        # 备用CPU模式
        print("回退到CPU模式...")
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['CUDA_VISIBLE_DEVICES'] = ''
        
        # CPU模式使用更宽松的参数
        cpu_cmd = [
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--output_path', sparse_dir,
            '--Mapper.init_min_num_inliers', '3',  # 极低阈值
            '--Mapper.min_num_matches', '3',       # 极低匹配要求
            '--Mapper.max_num_models', '100',
            '--Mapper.init_min_tri_angle', '1.0',
            '--Mapper.multiple_models', '1',
            '--Mapper.extract_colors', '0'
        ]
        
        print(f"执行CPU重建命令: {' '.join(cpu_cmd)}")
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, env=env)
        
        print(f"CPU mapper 返回码: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            print("❌ CPU模式重建也失败了")
            self._check_reconstruction_output(sparse_dir)
            raise RuntimeError(f"COLMAP增量重建失败: {result.stderr}")
        
        print("✅ CPU模式重建成功")
        self._check_reconstruction_output(sparse_dir)

    def _check_reconstruction_output(self, sparse_dir: str):
        """检查重建输出结果"""
        print(f"\n🔍 检查重建输出目录: {sparse_dir}")
        
        if not os.path.exists(sparse_dir):
            print("❌ 重建输出目录不存在")
            return
        
        try:
            items = os.listdir(sparse_dir)
            print(f"输出目录内容: {items}")
            
            if not items:
                print("⚠️  重建输出目录为空")
                return
            
            for item in items:
                item_path = os.path.join(sparse_dir, item)
                if os.path.isdir(item_path):
                    print(f"\n📁 检查子目录: {item}")
                    sub_items = os.listdir(item_path)
                    print(f"  内容: {sub_items}")
                    
                    # 检查COLMAP文件
                    colmap_files = ['cameras.txt', 'images.txt', 'points3D.txt']
                    for colmap_file in colmap_files:
                        file_path = os.path.join(item_path, colmap_file)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path)
                            print(f"    {colmap_file}: {size} bytes")
                        else:
                            print(f"    {colmap_file}: 不存在")
                            
        except Exception as e:
            print(f"❌ 检查输出目录失败: {e}")

    def _read_native_reconstruction(self, sparse_dir: str):
        """读取原生COLMAP重建结果"""
        # 查找重建目录
        recon_dirs = []
        
        print(f"检查稀疏重建目录: {sparse_dir}")
        
        if not os.path.exists(sparse_dir):
            print(f"错误: 稀疏目录不存在: {sparse_dir}")
            return None
        
        # 列出所有内容进行诊断
        try:
            all_items = os.listdir(sparse_dir)
            print(f"稀疏目录内容: {all_items}")
        except Exception as e:
            print(f"无法列出稀疏目录内容: {e}")
            return None
        
        for item in all_items:
            item_path = os.path.join(sparse_dir, item)
            print(f"检查项目: {item} -> {item_path}")
            
            if os.path.isdir(item_path):
                print(f"  发现子目录: {item}")
                
                # 检查COLMAP重建文件（支持二进制和文本格式）
                txt_files = ['cameras.txt', 'images.txt', 'points3D.txt']
                bin_files = ['cameras.bin', 'images.bin', 'points3D.bin']
                
                # 检查文本格式文件
                txt_exist = all(os.path.exists(os.path.join(item_path, f)) for f in txt_files)
                # 检查二进制格式文件
                bin_exist = all(os.path.exists(os.path.join(item_path, f)) for f in bin_files)
                
                print(f"    文本格式文件: {'存在' if txt_exist else '不存在'}")
                print(f"    二进制格式文件: {'存在' if bin_exist else '不存在'}")
                
                if txt_exist:
                    # 如果文本文件存在，检查其有效性
                    if self._validate_reconstruction_dir(item_path, 'txt'):
                        recon_dirs.append(item_path)
                        print(f"    ✅ 有效的文本格式重建目录: {item}")
                    else:
                        print(f"    ⚠️  文本文件存在但内容无效: {item}")
                        
                elif bin_exist:
                    # 如果只有二进制文件，尝试转换为文本格式
                    print(f"    🔄 发现二进制格式，尝试转换为文本格式...")
                    if self._convert_bin_to_txt(item_path):
                        if self._validate_reconstruction_dir(item_path, 'txt'):
                            recon_dirs.append(item_path)
                            print(f"    ✅ 成功转换并验证重建目录: {item}")
                        else:
                            print(f"    ⚠️  转换成功但内容无效: {item}")
                    else:
                        # 转换失败，尝试直接使用PyColmap读取二进制格式
                        print(f"    🔄 转换失败，尝试直接读取二进制格式...")
                        if self._validate_reconstruction_dir(item_path, 'bin'):
                            recon_dirs.append(item_path)
                            print(f"    ✅ 可以直接读取二进制格式: {item}")
                        else:
                            print(f"    ❌ 二进制格式也无法读取: {item}")
                else:
                    print(f"    ❌ 既没有文本文件也没有二进制文件: {item}")
            else:
                print(f"  跳过文件: {item}")
        
        if not recon_dirs:
            print("❌ 未找到任何有效的重建目录")
            print("可能的原因:")
            print("  1. COLMAP重建失败但返回了成功状态")
            print("  2. 重建文件生成不完整")
            print("  3. 图像特征匹配不足")
            print("  4. 相机参数初始化失败")
            return None
        
        # 选择第一个重建目录
        recon_dir = recon_dirs[0]
        print(f"✅ 选择重建目录: {recon_dir}")
        
        # 使用PyColmap读取COLMAP格式文件
        try:
            print("正在使用PyColmap读取重建数据...")
            reconstruction = self.pycolmap.Reconstruction(recon_dir)
            
            # 验证读取的数据
            num_cameras = len(reconstruction.cameras)
            num_images = len(reconstruction.images)
            num_points = len(reconstruction.points3D)
            
            print(f"✅ 成功读取重建：{num_cameras}个相机，{num_images}张图像，{num_points}个3D点")
            
            if num_cameras == 0:
                print("⚠️  警告: 没有相机数据")
                return None
            
            if num_images == 0:
                print("⚠️  警告: 没有注册的图像")
                return None
            
            return reconstruction
            
        except Exception as e:
            print(f"❌ PyColmap读取失败: {e}")
            print("尝试手动检查文件格式...")
            
            # 尝试手动读取和诊断文件
            self._diagnose_colmap_files(recon_dir)
            return None

    def _convert_bin_to_txt(self, recon_dir: str) -> bool:
        """将COLMAP二进制格式转换为文本格式"""
        import subprocess
        
        try:
            print("    执行COLMAP模型转换...")
            
            # 设置环境变量
            env = os.environ.copy()
            env['QT_QPA_PLATFORM'] = 'offscreen'
            
            # 使用COLMAP的model_converter命令
            cmd = [
                'colmap', 'model_converter',
                '--input_path', recon_dir,
                '--output_path', recon_dir,
                '--output_type', 'TXT'
            ]
            
            print(f"    转换命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            print(f"    转换返回码: {result.returncode}")
            if result.stdout:
                print(f"    STDOUT: {result.stdout}")
            if result.stderr:
                print(f"    STDERR: {result.stderr}")
            
            if result.returncode == 0:
                print("    ✅ 二进制到文本转换成功")
                return True
            else:
                print(f"    ❌ 转换失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"    ❌ 转换过程出错: {e}")
            return False

    def _validate_reconstruction_dir(self, recon_dir: str, format_type: str) -> bool:
        """验证重建目录是否有效"""
        try:
            if format_type == 'txt':
                # 验证文本格式
                required_files = ['cameras.txt', 'images.txt', 'points3D.txt']
                for req_file in required_files:
                    file_path = os.path.join(recon_dir, req_file)
                    if not os.path.exists(file_path):
                        return False
                    if not self._validate_colmap_file(file_path, req_file):
                        return False
                return True
                
            elif format_type == 'bin':
                # 验证二进制格式 - 尝试用PyColmap读取
                try:
                    test_reconstruction = self.pycolmap.Reconstruction(recon_dir)
                    return len(test_reconstruction.cameras) > 0 and len(test_reconstruction.images) > 0
                except:
                    return False
                    
        except Exception as e:
            print(f"      验证目录失败: {e}")
            return False

    def _validate_colmap_file(self, file_path: str, file_type: str) -> bool:
        """验证COLMAP文件是否有效"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 过滤掉注释和空行
            data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            
            if file_type == 'cameras.txt':
                # 相机文件应该至少有一行数据
                if len(data_lines) == 0:
                    print(f"      {file_type}: 没有相机数据")
                    return False
                print(f"      {file_type}: {len(data_lines)} 个相机")
                
            elif file_type == 'images.txt':
                # 图像文件应该有成对的行（图像行 + 特征点行）
                if len(data_lines) == 0:
                    print(f"      {file_type}: 没有图像数据")
                    return False
                print(f"      {file_type}: {len(data_lines)} 行数据")
                
            elif file_type == 'points3D.txt':
                # 3D点文件可以为空（没有3D点也能进行相机估计）
                print(f"      {file_type}: {len(data_lines)} 个3D点")
            
            return True
            
        except Exception as e:
            print(f"      {file_type}: 读取失败 - {e}")
            return False

    def _diagnose_colmap_files(self, recon_dir: str):
        """诊断COLMAP文件内容"""
        print("\n🔍 详细文件诊断:")
        
        files_to_check = ['cameras.txt', 'images.txt', 'points3D.txt']
        
        for filename in files_to_check:
            file_path = os.path.join(recon_dir, filename)
            print(f"\n📄 检查 {filename}:")
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                print(f"  总行数: {len(lines)}")
                
                # 显示前几行内容
                print("  前5行内容:")
                for i, line in enumerate(lines[:5]):
                    print(f"    {i+1}: {repr(line.strip())}")
                
                # 统计数据行
                data_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                print(f"  数据行数: {len(data_lines)}")
                
                if len(data_lines) > 0:
                    print("  首个数据行:")
                    print(f"    {repr(data_lines[0].strip())}")
                
            except Exception as e:
                print(f"  ❌ 无法读取文件: {e}")

    def _parse_native_reconstruction(self, reconstruction, num_input_images: int) -> Dict:
        """解析原生COLMAP重建结果"""
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
            print(f"解析原生重建结果失败: {e}")
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
                "force_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制使用GPU模式（失败时不回退到CPU）"
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
                                 force_gpu: bool = True,
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
            print(f"GPU模式: {'强制启用' if force_gpu else '自适应'}")
            
            # 使用COLMAP进行估计
            result = self.estimator.estimate_from_images(
                images=image_list,
                colmap_feature_type=colmap_feature_type,
                colmap_matcher_type=colmap_matcher_type,
                colmap_quality=colmap_quality,
                enable_dense_reconstruction=enable_dense_reconstruction,
                force_gpu=force_gpu
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