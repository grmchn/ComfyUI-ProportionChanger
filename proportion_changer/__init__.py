"""
ProportionChanger nodes package
Provides organized node classes for DWPose detection and rendering
"""

# Import detector node classes
from .detector_nodes import (
    ProportionChangerDWPoseDetector
)

# Import ultimate detector node class
from .ultimate_detector_node import (
    ProportionChangerReference
)

# Import render node classes
from .render_nodes import (
    ProportionChangerDWPoseRender
)

# Import params node classes
from .params_node import (
    ProportionChangerParams
)

# Import interpolation node classes
from .interpolation_node import (
    ProportionChangerInterpolator
)

# Import keypoint denoiser node classes
from .keypoint_denoiser_node import (
    ProportionChangerKeypointDenoiser,
    ProportionChangerKeypointDenoiserAdvanced
)

__all__ = [
    # Detector nodes
    "ProportionChangerDWPoseDetector",
    
    # Ultimate detector node
    "ProportionChangerReference",
    
    # Render nodes
    "ProportionChangerDWPoseRender",
    
    # Params nodes
    "ProportionChangerParams",
    
    # Interpolation nodes
    "ProportionChangerInterpolator",
    
    # KeyPoint Denoiser nodes
    "ProportionChangerKeypointDenoiser",
    "ProportionChangerKeypointDenoiserAdvanced"
]