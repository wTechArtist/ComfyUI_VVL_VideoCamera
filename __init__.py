"""
ComfyUI_VVL_VideoCamera - VVL 视频相机参数估计插件

提供从视频中估计相机内参和外参的功能，支持多种估计方法
"""

try:
    from .video_camera_estimator import NODE_CLASS_MAPPINGS as VCE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VCE_DISPLAY_MAPPINGS
except ImportError as e:
    print(f"Warning: Could not import VideoCameraEstimator node: {e}")
    VCE_NODE_MAPPINGS = {}
    VCE_DISPLAY_MAPPINGS = {}

try:
    from .depth_estimator import NODE_CLASS_MAPPINGS as DE_NODE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DE_DISPLAY_MAPPINGS
except ImportError as e:
    print(f"Warning: Could not import DepthEstimator node: {e}")
    DE_NODE_MAPPINGS = {}
    DE_DISPLAY_MAPPINGS = {}

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(VCE_NODE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(DE_NODE_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(VCE_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DE_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 