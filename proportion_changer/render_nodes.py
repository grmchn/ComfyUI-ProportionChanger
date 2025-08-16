"""
ProportionChanger render node classes
Contains nodes for rendering POSE_KEYPOINT data using DWPose styling
"""

import numpy as np
import torch

# Import rendering utilities from our utils package
from ..utils import draw_dwpose_render


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
                "POSE_KEYPOINT": ("POSE_KEYPOINT", {"default": None, "tooltip": "POSE_KEYPOINT data to render"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_img"
    CATEGORY = "ProportionChanger"

    def render_img(self, show_body, show_face, show_hands, show_feet, resolution_x, 
                   pose_marker_size, face_marker_size, hand_marker_size, POSE_KEYPOINT=None):
        
        if POSE_KEYPOINT is None:
            raise ValueError("POSE_KEYPOINT input is required")
        
        # Debug: Print POSE_KEYPOINT structure
        print(f"🔍 Debug POSE_KEYPOINT type: {type(POSE_KEYPOINT)}")
        if isinstance(POSE_KEYPOINT, list) and len(POSE_KEYPOINT) > 0:
            print(f"🔍 Debug POSE_KEYPOINT length: {len(POSE_KEYPOINT)}")
            first_frame = POSE_KEYPOINT[0]
            print(f"🔍 Debug first frame keys: {first_frame.keys() if isinstance(first_frame, dict) else 'Not a dict'}")
            if isinstance(first_frame, dict) and 'people' in first_frame:
                print(f"🔍 Debug people count: {len(first_frame['people'])}")
                if len(first_frame['people']) > 0:
                    person = first_frame['people'][0]
                    if 'pose_keypoints_2d' in person:
                        keypoints = person['pose_keypoints_2d']
                        print(f"🔍 Debug pose_keypoints_2d length: {len(keypoints)}")
                        print(f"🔍 Debug first 9 keypoints: {keypoints[:9]}")
                        # Check for actual coordinate values
                        non_zero_coords = [(i//3, keypoints[i], keypoints[i+1], keypoints[i+2]) 
                                         for i in range(0, min(54, len(keypoints)), 3) 
                                         if keypoints[i] != 0 or keypoints[i+1] != 0]
                        print(f"🔍 Debug non-zero coordinates (first 5): {non_zero_coords[:5]}")
        elif isinstance(POSE_KEYPOINT, dict):
            print(f"🔍 Debug single frame keys: {POSE_KEYPOINT.keys()}")
        
        # Render using DWPose algorithms
        pose_imgs = draw_dwpose_render(
            POSE_KEYPOINT, resolution_x, show_body, show_face, show_hands, show_feet,
            pose_marker_size, face_marker_size, hand_marker_size
        )
        
        if pose_imgs:
            # Convert to ComfyUI tensor format
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return (torch.from_numpy(pose_imgs_np),)
        else:
            raise ValueError("Invalid input type. Expected an input to give an output.")