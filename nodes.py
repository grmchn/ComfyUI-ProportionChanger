"""
ProportionChanger nodes - Main entry point
Backward compatibility maintained for existing workflows
"""

from .utils import log
import json

# Import all nodes from our organized structure
from .proportion_changer import (
    ProportionChangerReference,
    ProportionChangerDWPoseDetector,
    ProportionChangerDWPoseRender,
    ProportionChangerParams,
    ProportionChangerInterpolator
)


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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert_to_json"
    OUTPUT_NODE = True
    CATEGORY = "ProportionChanger"
    
    def convert_to_json(self, pose_keypoint, pretty_format, unique_id=None, extra_pnginfo=None):
        """
        Convert POSE_KEYPOINT data to JSON string that can be copied to PoseJSONToPoseKeypoint
        """
        try:
            if not pose_keypoint or len(pose_keypoint) == 0:
                text = '{"version": "1.0", "people": []}'
            else:
                # Convert to JSON string
                if pretty_format:
                    text = json.dumps(pose_keypoint, indent=2, ensure_ascii=False)
                else:
                    text = json.dumps(pose_keypoint, ensure_ascii=False)
            
            # Update workflow widgets_values to persist data across reloads (like Show Text)
            if unique_id is not None and extra_pnginfo is not None:
                # Ensure extra_pnginfo is a list
                if not isinstance(extra_pnginfo, list):
                    log.warning(f"extra_pnginfo is not a list (type: {type(extra_pnginfo)}), attempting to convert")
                    if isinstance(extra_pnginfo, dict):
                        extra_pnginfo = [extra_pnginfo]
                    else:
                        log.error(f"Cannot convert extra_pnginfo to list, skipping workflow update")
                        extra_pnginfo = None
                
                if extra_pnginfo and len(extra_pnginfo) > 0:
                    if (
                        not isinstance(extra_pnginfo[0], dict)
                        or "workflow" not in extra_pnginfo[0]
                    ):
                        log.error("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
                    else:
                        workflow = extra_pnginfo[0]["workflow"]
                    node = next(
                        (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                        None,
                    )
                    if node:
                        # Keep pretty_format value, update text display value
                        node["widgets_values"] = [pretty_format, text]
            
            return {"ui": {"text": (text,)}, "result": (text,)}
            
        except Exception as e:
            log.error(f"Failed to convert POSE_KEYPOINT to JSON: {e}")
            text = f'{{"error": "Failed to convert POSE_KEYPOINT to JSON: {str(e)}"}}'
            
            # Update workflow even on error
            if unique_id is not None and extra_pnginfo is not None:
                # Ensure extra_pnginfo is a list (error case)
                if not isinstance(extra_pnginfo, list):
                    if isinstance(extra_pnginfo, dict):
                        extra_pnginfo = [extra_pnginfo]
                    else:
                        extra_pnginfo = None
                
                if extra_pnginfo and len(extra_pnginfo) > 0:
                    if isinstance(extra_pnginfo[0], dict) and "workflow" in extra_pnginfo[0]:
                        workflow = extra_pnginfo[0]["workflow"]
                        node = next(
                            (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                            None,
                        )
                        if node:
                            node["widgets_values"] = [pretty_format, text]
            
            return {"ui": {"text": (text,)}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    # Main DWPose detector nodes
    "ProportionChangerReference": ProportionChangerReference,
    "ProportionChangerDWPoseDetector": ProportionChangerDWPoseDetector,
    "ProportionChangerDWPoseRender": ProportionChangerDWPoseRender,
    "ProportionChangerParams": ProportionChangerParams,
    "ProportionChangerInterpolator": ProportionChangerInterpolator,
    
    # Utility nodes
    "PoseJSONToPoseKeypoint": PoseJSONToPoseKeypoint,
    "PoseKeypointPreview": PoseKeypointPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main DWPose detector nodes
    "ProportionChangerReference": "ProportionChanger Reference",
    "ProportionChangerDWPoseDetector": "ProportionChanger DWPose Detector",
    "ProportionChangerDWPoseRender": "ProportionChanger DWPose Render",
    "ProportionChangerParams": "ProportionChanger Params",
    "ProportionChangerInterpolator": "ProportionChanger Interpolator",
    
    # Utility nodes
    "PoseJSONToPoseKeypoint": "pose_keypoint input",
    "PoseKeypointPreview": "pose_keypoint preview",
}