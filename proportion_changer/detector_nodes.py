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


class ProportionChangerUniAnimateDWPoseDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_images": ("IMAGE", {"tooltip": "Pose images"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Stick width for drawing keypoints"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "body_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Body keypoint size"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw feet keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "hand_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Hand keypoint size"}),
                "colorspace": (["RGB", "BGR"], {"tooltip": "Color space for the output image"}),
                "handle_not_detected": (["empty", "repeat"], {"default": "empty", "tooltip": "How to handle undetected poses, empty inserts black and repeat inserts previous detection"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Draw head keypoints"}),
            },
            "optional": {
                "reference_pose_image": ("IMAGE", {"tooltip": "Reference pose image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("poses", "reference_pose",)
    FUNCTION = "process"
    CATEGORY = "ProportionChanger"

    def process(self, pose_images, score_threshold, stick_width, reference_pose_image=None, draw_body=True, body_keypoint_size=4, 
                draw_feet=True, draw_hands=True, hand_keypoint_size=4, colorspace="RGB", handle_not_detected="empty", draw_head=True):

        device = mm.get_torch_device()
        
        #model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "..", "unianimate", "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

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

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = DWposeDetector(self.det, self.pose) 

        #model inference
        height, width = pose_images.shape[1:3]
        
        pose_np = pose_images.cpu().numpy() * 255
        ref_np = None
        if reference_pose_image is not None:
            ref = reference_pose_image
            ref_np = ref.cpu().numpy() * 255

        poses, reference_pose = pose_extract(pose_np, ref_np, self.dwpose_detector, height, width, score_threshold, stick_width=stick_width,
                                             draw_body=draw_body, body_keypoint_size=body_keypoint_size, draw_feet=draw_feet, 
                                             draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size, handle_not_detected=handle_not_detected, draw_head=draw_head)
        poses = poses / 255.0
        if reference_pose_image is not None:
            reference_pose = reference_pose.unsqueeze(0) / 255.0
        else:
            reference_pose = torch.zeros(1, 64, 64, 3, device=torch.device("cpu"))

        if colorspace == "BGR":
            poses=torch.flip(poses, dims=[-1])

        return (poses, reference_pose, )


class ProportionChangerDWPoseDetectorForPoseKeypoint:
    """
    DWPose detector node that extracts pose keypoints from image and outputs POSE_KEYPOINT format.
    This node is designed to work with ProportionChangerUltimateUniAnimateDWPoseDetector.
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
                
            except Exception as e:
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
                for i in range(0, len(body_kpts), 3):
                    if i+1 < len(body_kpts):
                        body_kpts[i] = body_kpts[i] / canvas_width      # x coordinate
                        body_kpts[i+1] = body_kpts[i+1] / canvas_height  # y coordinate
            
            # Normalize face keypoints
            if "face_keypoints_2d" in person:
                face_kpts = person["face_keypoints_2d"]
                for i in range(0, len(face_kpts), 3):
                    if i+1 < len(face_kpts):
                        face_kpts[i] = face_kpts[i] / canvas_width
                        face_kpts[i+1] = face_kpts[i+1] / canvas_height
            
            # Normalize hand keypoints
            for hand_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                if hand_key in person:
                    hand_kpts = person[hand_key]
                    for i in range(0, len(hand_kpts), 3):
                        if i+1 < len(hand_kpts):
                            hand_kpts[i] = hand_kpts[i] / canvas_width
                            hand_kpts[i+1] = hand_kpts[i+1] / canvas_height
        
        return pose_keypoint