"""
ProportionChanger render node classes
Contains nodes for rendering POSE_KEYPOINT data using DWPose styling
"""

import numpy as np
import torch

# Import rendering utilities from our utils package
from ..utils import draw_dwpose_render
import os
import logging

# Debug mode control via environment variable (local implementation to avoid imports)
DEBUG_MODE = os.getenv('PROPORTION_CHANGER_DEBUG', 'false').lower() in ('true', '1', 'yes')
log = logging.getLogger(__name__)

def debug_log(message):
    """Conditional debug logging - only outputs if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        log.debug(f"🔍 {message}")


class ProportionChangerDWPoseRender:
    """
    DWPose Render Node with 25-point keypoint support including toe keypoints.
    Compatible with ultimate-openpose-render parameters but using DWPose rendering algorithms.
    Resolves coordinate misalignment issues when displaying ProportionChanger outputs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "show_face": ("BOOLEAN", {"default": True, "tooltip": "Draw face keypoints"}),
                "show_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "show_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw toe keypoints (DWPose 25-point feature)"}),
                "resolution_x": ("INT", {"default": -1, "min": -1, "max": 12800, "tooltip": "Output width (-1 for original)"}),
                "pose_marker_size": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Body keypoint marker size"}),
                "face_marker_size": ("INT", {"default": 3, "min": 0, "max": 100, "tooltip": "Face keypoint marker size"}),
                "hand_marker_size": ("INT", {"default": 2, "min": 0, "max": 100, "tooltip": "Hand keypoint marker size"}),
                "pose_keypoint": ("POSE_KEYPOINT", {"default": None, "tooltip": "POSE_KEYPOINT data to render"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_img"
    CATEGORY = "ProportionChanger"

    def render_img(self, show_body, show_face, show_hands, show_feet, resolution_x, 
                   pose_marker_size, face_marker_size, hand_marker_size, pose_keypoint=None):
        
        if pose_keypoint is None:
            raise ValueError("pose_keypoint input is required")
        
        # Debug: Print pose_keypoint structure
        debug_log(f"Debug pose_keypoint type: {type(pose_keypoint)}")
        if isinstance(pose_keypoint, list) and len(pose_keypoint) > 0:
            debug_log(f"Debug pose_keypoint length: {len(pose_keypoint)}")
            first_frame = pose_keypoint[0]
            debug_log(f"Debug first frame keys: {first_frame.keys() if isinstance(first_frame, dict) else 'Not a dict'}")
            if isinstance(first_frame, dict) and 'people' in first_frame:
                debug_log(f"Debug people count: {len(first_frame['people'])}")
                if len(first_frame['people']) > 0:
                    person = first_frame['people'][0]
                    if 'pose_keypoints_2d' in person:
                        keypoints = person['pose_keypoints_2d']
                        debug_log(f"Debug pose_keypoints_2d length: {len(keypoints)}")
                        debug_log(f"Debug first 9 keypoints: {keypoints[:9]}")
                        # Check for actual coordinate values
                        non_zero_coords = [(i//3, keypoints[i], keypoints[i+1], keypoints[i+2]) 
                                         for i in range(0, min(54, len(keypoints)), 3) 
                                         if keypoints[i] != 0 or keypoints[i+1] != 0]
                        debug_log(f"Debug non-zero coordinates (first 5): {non_zero_coords[:5]}")
        elif isinstance(pose_keypoint, dict):
            debug_log(f"Debug single frame keys: {pose_keypoint.keys()}")
        
        # Render using DWPose algorithms
        pose_imgs = draw_dwpose_render(
            pose_keypoint, resolution_x, show_body, show_face, show_hands, show_feet,
            pose_marker_size, face_marker_size, hand_marker_size
        )
        
        if pose_imgs:
            # Convert to ComfyUI tensor format
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return (torch.from_numpy(pose_imgs_np),)
        else:
            raise ValueError("Invalid input type. Expected an input to give an output.")