"""
ProportionChanger nodes package
Provides organized node classes for DWPose detection and rendering
"""

# Import detector node classes
from .detector_nodes import (
    ProportionChangerUniAnimateDWPoseDetector,
    ProportionChangerDWPoseDetectorForPoseKeypoint
)

# Import ultimate detector node class
from .ultimate_detector_node import (
    ProportionChangerUltimateUniAnimateDWPoseDetector
)

# Import render node classes
from .render_nodes import (
    ProportionChangerDWPoseRender
)

__all__ = [
    # Detector nodes
    "ProportionChangerUniAnimateDWPoseDetector",
    "ProportionChangerDWPoseDetectorForPoseKeypoint",
    
    # Ultimate detector node
    "ProportionChangerUltimateUniAnimateDWPoseDetector",
    
    # Render nodes
    "ProportionChangerDWPoseRender"
]