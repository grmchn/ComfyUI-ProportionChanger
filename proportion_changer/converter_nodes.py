"""
Converter nodes for bridging WanAnimate POSEDATA into POSE_KEYPOINT format.
"""

from __future__ import annotations

import numpy as np

try:  # Standard package import (ComfyUI runtime)
    from ..utils import log
except ImportError:  # Fallback for standalone/unit test environments
    from utils import log


def _to_array(value):
    """Convert nested list/tuple/ndarray to numpy array or return None."""
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=float)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def _looks_normalized(coords: np.ndarray) -> bool:
    """
    Heuristic: treat coordinates as normalized (0..1-ish) if their absolute max is small.
    WanAnimate/UniAnimate often uses normalized coordinates; some POSEDATA variants already store pixels.
    """
    if coords is None or coords.size == 0:
        return True
    finite = coords[np.isfinite(coords)]
    if finite.size == 0:
        return True
    return float(np.max(np.abs(finite))) <= 2.0


def _extract_keypoints(meta, coord_name, conf_name, target_points, width, height, default_conf=1.0):
    """
    Extract, scale, and pad keypoints to the desired target length.
    Missing coordinates or invalid values are zeroed with confidence 0.
    """
    coords_raw = getattr(meta, coord_name, None)
    conf_raw = getattr(meta, conf_name, None)

    # Fall back to dict-style access if present
    if coords_raw is None and isinstance(meta, dict):
        coords_raw = meta.get(coord_name)
    if conf_raw is None and isinstance(meta, dict):
        conf_raw = meta.get(conf_name)

    coords = _to_array(coords_raw)
    confs = _to_array(conf_raw)
    if confs is not None:
        confs = confs.reshape(-1)

    # Normalize coordinate shape to (N, 2)
    if coords is not None:
        if coords.ndim == 1:
            # Flattened x,y pairs
            if coords.size % 2 == 0:
                coords = coords.reshape(-1, 2)
            else:
                coords = None
        elif coords.shape[-1] >= 2:
            coords = coords.reshape(-1, coords.shape[-1])[:, :2]
        else:
            coords = None

    scale_from_normalized = _looks_normalized(coords) if coords is not None else True

    points = []
    max_points = coords.shape[0] if coords is not None else 0

    for idx in range(target_points):
        if coords is not None and idx < max_points:
            x, y = coords[idx]
            conf = confs[idx] if confs is not None and idx < confs.shape[0] else default_conf

            # Validate numerical values
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(conf):
                x = y = 0.0
                conf = 0.0
            else:
                x = float(x)
                y = float(y)
                if scale_from_normalized:
                    x *= width
                    y *= height
                conf = float(conf)
        else:
            x = y = 0.0
            conf = 0.0

        points.extend([x, y, conf])

    return points


class PoseDataToPoseKeypoint:
    """
    Convert WanAnimatePreprocess POSEDATA into POSE_KEYPOINT format used by ProportionChanger.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {"tooltip": "POSEDATA output from WanAnimate Pose & Face Detection."}),
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "convert"
    CATEGORY = "ProportionChanger"

    def convert(self, pose_data, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        pose_metas = None
        if isinstance(pose_data, dict):
            pose_metas = pose_data.get("pose_metas") or pose_data.get("pose_meta")
        if pose_metas is None:
            pose_metas = getattr(pose_data, "pose_metas", None)

        if not pose_metas:
            empty_frame = {
                "version": "1.0",
                "people": [],
                "canvas_width": width,
                "canvas_height": height,
            }
            return ([empty_frame],)

        frames = []
        for meta in pose_metas:
            try:
                body_points = _extract_keypoints(meta, "kps_body", "kps_body_p", 25, width, height, default_conf=1.0)
                # AAPoseMeta typically supplies 20 points; padder already set remaining to zeros
                face_points = _extract_keypoints(meta, "kps_face", "kps_face_p", 70, width, height, default_conf=1.0)
                lhand_points = _extract_keypoints(meta, "kps_lhand", "kps_lhand_p", 21, width, height, default_conf=1.0)
                rhand_points = _extract_keypoints(meta, "kps_rhand", "kps_rhand_p", 21, width, height, default_conf=1.0)

                # Determine if body has any confidence > 0
                has_body = any(body_points[i] > 0 for i in range(2, len(body_points), 3))

                frame = {
                    "version": "1.0",
                    "people": [],
                    "canvas_width": width,
                    "canvas_height": height,
                }

                if has_body:
                    person = {
                        "pose_keypoints_2d": body_points,
                        "face_keypoints_2d": face_points,
                        "hand_left_keypoints_2d": lhand_points,
                        "hand_right_keypoints_2d": rhand_points,
                    }
                    frame["people"].append(person)

                frames.append(frame)
            except Exception as exc:
                log.error(f"PoseDataToPoseKeypoint conversion failed for frame: {exc}")
                frames.append(
                    {
                        "version": "1.0",
                        "people": [],
                        "canvas_width": width,
                        "canvas_height": height,
                    }
                )

        return (frames,)
