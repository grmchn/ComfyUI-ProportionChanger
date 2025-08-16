"""
ProportionChanger utilities package
Provides organized utilities for DWPose processing, filtering, and rendering
"""

# Import log from the main utils module for backward compatibility
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Filtering utilities
from .filters import (
    OneEuroFilter,
    smoothing_factor,
    exponential_smoothing
)

# DWPose core functionality
from .dwpose_core import (
    DWposeDetector,
    update_transformer,
    draw_pose,
    pose_extract
)

# Pose data processing and conversion
from .pose_processing import (
    pose_keypoint_to_dwpose_format,
    dwpose_format_to_pose_keypoint
)

# Rendering functionality
from .rendering import (
    draw_dwpose_render,
    draw_dwpose_body_and_foot,
    draw_dwpose_handpose,
    draw_dwpose_facepose
)

__all__ = [
    # Logging
    "log",
    
    # Filtering
    "OneEuroFilter",
    "smoothing_factor", 
    "exponential_smoothing",
    
    # DWPose core
    "DWposeDetector",
    "update_transformer",
    "draw_pose",
    "pose_extract",
    
    # Data processing
    "pose_keypoint_to_dwpose_format",
    "dwpose_format_to_pose_keypoint",
    
    # Rendering
    "draw_dwpose_render",
    "draw_dwpose_body_and_foot",
    "draw_dwpose_handpose",
    "draw_dwpose_facepose"
]