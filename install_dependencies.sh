#!/bin/bash
# VVL VideoCamera 依赖安装脚本

echo "🚀 VVL VideoCamera 依赖安装脚本"
echo "================================"

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ 检测到Linux系统"
else
    echo "❌ 此脚本仅支持Linux系统"
    exit 1
fi

# 更新包列表
echo "📦 更新包列表..."
sudo apt-get update

# 1. 安装COLMAP
echo ""
echo "1️⃣ 安装COLMAP..."
if command -v colmap &> /dev/null; then
    echo "✅ COLMAP已安装"
    colmap -h | head -n 1
else
    echo "安装COLMAP..."
    sudo apt-get install -y colmap
fi

# 2. 安装Xvfb（虚拟显示）
echo ""
echo "2️⃣ 安装Xvfb（解决无头服务器问题）..."
if command -v xvfb-run &> /dev/null; then
    echo "✅ Xvfb已安装"
else
    echo "安装Xvfb..."
    sudo apt-get install -y xvfb
fi

# 3. 安装Python依赖
echo ""
echo "3️⃣ 安装Python依赖..."
pip install pycolmap opencv-python pillow

# 4. 创建启动脚本
echo ""
echo "4️⃣ 创建启动脚本..."
cat > run_comfyui_with_colmap.sh << 'EOF'
#!/bin/bash
# ComfyUI启动脚本（支持COLMAP）

echo "🚀 启动ComfyUI（带COLMAP支持）"

# 检查是否需要Xvfb
if [ -z "$DISPLAY" ]; then
    echo "检测到无显示环境，使用xvfb-run"
    xvfb-run -a --server-args="-screen 0 1024x768x24" python main.py "$@"
else
    echo "使用现有显示环境"
    python main.py "$@"
fi
EOF

chmod +x run_comfyui_with_colmap.sh

# 5. 测试安装
echo ""
echo "5️⃣ 测试安装..."
echo "测试COLMAP..."
if xvfb-run -a colmap help &> /dev/null; then
    echo "✅ COLMAP可以正常运行"
else
    echo "⚠️  COLMAP测试失败，请检查安装"
fi

echo "测试Python导入..."
python -c "import pycolmap; print('✅ PyColmap导入成功')" 2>/dev/null || echo "❌ PyColmap导入失败"
python -c "import cv2; print('✅ OpenCV导入成功')" 2>/dev/null || echo "❌ OpenCV导入失败"

# 完成
echo ""
echo "✨ 安装完成！"
echo ""
echo "使用方法："
echo "1. 直接运行ComfyUI: ./run_comfyui_with_colmap.sh"
echo "2. 或手动设置: xvfb-run -a python main.py"
echo ""
echo "注意：VVL VideoCamera插件会自动处理显示环境问题" 