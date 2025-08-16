"""
ProportionChanger utility nodes
"""

from .utils import log
import json


class PoseJSONToPoseKeypoint:
    """
    Convert JSON string to POSE_KEYPOINT format for testing purposes
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json": ("STRING", {"multiline": True, "tooltip": "JSON string containing pose data"})
            }
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "convert"
    CATEGORY = "ProportionChanger"
    
    def convert(self, pose_json):
        """
        Convert JSON string to POSE_KEYPOINT format
        """
        if not pose_json or pose_json.strip() == "":
            # Return empty pose keypoint structure
            empty_person = {
                "pose_keypoints_2d": [0.0] * 75,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63
            }
            return ([{"people": [empty_person], "canvas_width": 512, "canvas_height": 768}],)
        
        try:
            # Parse JSON string
            parsed_data = json.loads(pose_json)
            
            # Validate basic structure
            if isinstance(parsed_data, dict):
                # Single frame format, wrap in list
                parsed_data = [parsed_data]
            elif not isinstance(parsed_data, list):
                raise ValueError("Invalid pose data format")
            
            # Basic validation for each frame
            for frame in parsed_data:
                if not isinstance(frame, dict):
                    raise ValueError("Each frame must be a dictionary")
                if "people" not in frame:
                    frame["people"] = []
                if "canvas_width" not in frame:
                    frame["canvas_width"] = 512
                if "canvas_height" not in frame:
                    frame["canvas_height"] = 768
            
            return (parsed_data,)
            
        except json.JSONDecodeError as e:
            log.error(f"JSON parsing error: {e}")
            # Return empty structure on error
            empty_person = {
                "pose_keypoints_2d": [0.0] * 75,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63
            }
            return ([{"people": [empty_person], "canvas_width": 512, "canvas_height": 768}],)
        except Exception as e:
            log.error(f"Pose conversion error: {e}")
            # Return empty structure on error
            empty_person = {
                "pose_keypoints_2d": [0.0] * 75,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63
            }
            return ([{"people": [empty_person], "canvas_width": 512, "canvas_height": 768}],)


class PoseKeypointPreview:
    """
    Convert POSE_KEYPOINT data to JSON string for debugging and copying to PoseJSONToPoseKeypoint
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "POSE_KEYPOINT data to convert to JSON"}),
                "pretty_format": ("BOOLEAN", {"default": True, "tooltip": "Format JSON with indentation for readability"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert_to_json"
    CATEGORY = "ProportionChanger"
    
    def convert_to_json(self, pose_keypoint, pretty_format):
        """
        Convert POSE_KEYPOINT data to JSON string that can be copied to PoseJSONToPoseKeypoint
        """
        try:
            if not pose_keypoint or len(pose_keypoint) == 0:
                text = '{"version": "1.0", "people": []}'
                return {"ui": {"text": (text,)}, "result": (text,)}
            
            # Convert to JSON string
            if pretty_format:
                text = json.dumps(pose_keypoint, indent=2, ensure_ascii=False)
            else:
                text = json.dumps(pose_keypoint, ensure_ascii=False)
            
            return {"ui": {"text": (text,)}, "result": (text,)}
            
        except Exception as e:
            log.error(f"Failed to convert POSE_KEYPOINT to JSON: {e}")
            text = f'{{"error": "Failed to convert POSE_KEYPOINT to JSON: {str(e)}"}}'
            return {"ui": {"text": (text,)}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "PoseJSONToPoseKeypoint": PoseJSONToPoseKeypoint,
    "PoseKeypointPreview": PoseKeypointPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseJSONToPoseKeypoint": "pose_keypoint input",
    "PoseKeypointPreview": "pose_keypoint preview",
}