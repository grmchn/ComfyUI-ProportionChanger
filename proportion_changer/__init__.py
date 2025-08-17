"""
ProportionChanger nodes package
Provides organized node classes for DWPose detection and rendering
"""

# Import detector node classes
from .detector_nodes import (
    ProportionChangerUniAnimateDWPoseDetector,
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

__all__ = [
    # Detector nodes
    "ProportionChangerUniAnimateDWPoseDetector",
    "ProportionChangerDWPoseDetector",
    
    # Ultimate detector node
    "ProportionChangerReference",
    
    # Render nodes
    "ProportionChangerDWPoseRender",
    
    # Params nodes
    "ProportionChangerParams"
]