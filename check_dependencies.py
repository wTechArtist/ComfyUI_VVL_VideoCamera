#!/usr/bin/env python3
"""
VVL VideoCamera 依赖检查脚本
检查所有必需和可选依赖的安装状态
"""

import sys
import importlib
import subprocess

def check_dependency(name, package_name=None, optional=False):
    """检查单个依赖"""
    if package_name is None:
        package_name = name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        status = "✅" if not optional else "✅ (可选)"
        print(f"{status} {name}: {version}")
        return True
    except ImportError:
        status = "❌" if not optional else "⚠️ (可选)"
        print(f"{status} {name}: 未安装")
        return not optional  # 必需依赖返回False，可选依赖返回True

def check_system_command(command, name, optional=False):
    """检查系统命令"""
    try:
        result = subprocess.run([command, '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status = "✅" if not optional else "✅ (可选)"
            print(f"{status} {name}: 可用")
            return True
        else:
            raise subprocess.CalledProcessError(result.returncode, command)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        status = "❌" if not optional else "⚠️ (可选)"
        print(f"{status} {name}: 不可用")
        return not optional

def main():
    """主检查函数"""
    print("=" * 50)
    print("VVL VideoCamera 依赖检查")
    print("=" * 50)
    
    success = True
    
    print("\n📦 核心Python依赖:")
    success &= check_dependency("PyTorch", "torch")
    success &= check_dependency("NumPy", "numpy")
    success &= check_dependency("OpenCV", "cv2")
    success &= check_dependency("Pillow", "PIL")
    success &= check_dependency("Matplotlib", "matplotlib")
    
    print("\n🔧 COLMAP集成:")
    success &= check_dependency("PyColmap", "pycolmap")
    success &= check_system_command("colmap", "COLMAP (系统版本)")
    
    print("\n🎯 可选依赖 (高级功能):")
    check_dependency("Open3D", "open3d", optional=True)
    check_system_command("nvidia-smi", "NVIDIA GPU", optional=True)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 所有必需依赖都已正确安装！")
        print("🚀 VVL VideoCamera 节点可以正常使用")
    else:
        print("❌ 部分必需依赖缺失")
        print("📖 请参考 REQUIREMENTS.md 进行安装")
        sys.exit(1)
    
    print("\n💡 提示:")
    print("- 运行 'pip install -r requirements.txt' 安装Python依赖")
    print("- 参考 REQUIREMENTS.md 了解完整安装指南")
    print("- 可选依赖缺失不影响基本功能")

if __name__ == "__main__":
    main() 