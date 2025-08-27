"""
ProportionChanger detector node classes
Contains the main DWPose detector nodes for pose estimation
"""

import os
import torch
import comfy.model_management as mm
from comfy.utils import ProgressBar

# Import utilities from our utils package
from ..utils import (
    DWposeDetector,
    pose_extract,
    dwpose_format_to_pose_keypoint
)
from ..utils import log


class ProportionChangerDWPoseDetector:
    """
    DWPose detector node that extracts pose keypoints from image and outputs POSE_KEYPOINT format.
    This node is designed to work with ProportionChangerReference.
    Includes toe keypoints (19-24) which are essential for full pose estimation.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE", {"tooltip": "Input image for pose detection"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "detect_pose"
    CATEGORY = "ProportionChanger"

    def detect_pose(self, image, score_threshold):
        device = mm.get_torch_device()
        
        # Model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "unianimate", "models", "DWPose")

        model_det = os.path.join(model_base_path, yolo_model)
        model_pose = os.path.join(model_base_path, dw_pose_model)

        # Download models if not exists
        if not os.path.exists(model_det):
            log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        # Initialize JIT models
        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = DWposeDetector(self.det, self.pose) 

        # Process image using the same approach as working UniAnimate detector
        height, width = image.shape[1:3]
        image_np = image.cpu().numpy() * 255
        
        pose_keypoints = []
        comfy_pbar = ProgressBar(len(image_np))
        
        for i, img in enumerate(image_np):
            try:
                # Use the high-level DWPose detector call (same as working UniAnimate version)
                pose = self.dwpose_detector(img, score_threshold=score_threshold)
                
                # Convert to POSE_KEYPOINT format using actual canvas dimensions
                pose_keypoint_frame = dwpose_format_to_pose_keypoint(
                    pose['bodies']['candidate'], 
                    pose['faces'], 
                    pose['hands'], 
                    width,  # Use actual canvas width for pixel coordinates
                    height  # Use actual canvas height for pixel coordinates
                )
                
                # Add canvas size info for compatibility
                pose_keypoint_frame["canvas_width"] = width
                pose_keypoint_frame["canvas_height"] = height
                pose_keypoints.append(pose_keypoint_frame)
                
            except (RuntimeError, ValueError, OSError) as e:
                print(f"Detection error: {e}")
            except Exception as e:
                print(f"Unexpected detection error: {e}")
                # Create empty pose data for failed detection
                empty_pose = {
                    "version": "1.0",
                    "people": [],
                    "canvas_width": width,
                    "canvas_height": height
                }
                pose_keypoints.append(empty_pose)
            
            comfy_pbar.update(1)

        return (pose_keypoints,)
    
    def _normalize_pose_coordinates(self, pose_keypoint, canvas_width, canvas_height):
        """
        Normalize pose coordinates from pixel values to 0-1 range
        """
        if not pose_keypoint or "people" not in pose_keypoint:
            return pose_keypoint
        
        for person in pose_keypoint["people"]:
            # Normalize body keypoints
            if "pose_keypoints_2d" in person:
                body_kpts = person["pose_keypoints_2d"]
                body_len = len(body_kpts)
                for i in range(0, body_len, 3):
                    if i+1 < body_len:
                        body_kpts[i] = body_kpts[i] / canvas_width      # x coordinate
                        body_kpts[i+1] = body_kpts[i+1] / canvas_height  # y coordinate
            
            # Normalize face keypoints
            if "face_keypoints_2d" in person:
                face_kpts = person["face_keypoints_2d"]
                face_len = len(face_kpts)
                for i in range(0, face_len, 3):
                    if i+1 < face_len:
                        face_kpts[i] = face_kpts[i] / canvas_width
                        face_kpts[i+1] = face_kpts[i+1] / canvas_height
            
            # Normalize hand keypoints
            for hand_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                if hand_key in person:
                    hand_kpts = person[hand_key]
                    hand_len = len(hand_kpts)
                    for i in range(0, hand_len, 3):
                        if i+1 < hand_len:
                            hand_kpts[i] = hand_kpts[i] / canvas_width
                            hand_kpts[i+1] = hand_kpts[i+1] / canvas_height
        
        return pose_keypoint