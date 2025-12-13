"""
Converter nodes for bridging WanAnimate POSEDATA into POSE_KEYPOINT format.
"""

from __future__ import annotations

import copy
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


def _looks_normalized_xy_triplets(triplets: list[float]) -> bool:
    """
    Heuristic: treat POSE_KEYPOINT x/y as normalized if values are in 0..1-ish scale.
    """
    if not triplets:
        return True

    max_abs = 0.0
    # sample all points with confidence > 0
    for i in range(0, len(triplets) - 2, 3):
        conf = triplets[i + 2]
        if conf is None or conf <= 0:
            continue
        x = triplets[i]
        y = triplets[i + 1]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        max_abs = max(max_abs, float(abs(x)), float(abs(y)))

    return max_abs <= 2.0


def _resize_pose_triplets_with_pad(
    triplets: list[float],
    *,
    src_w: float,
    src_h: float,
    dst_w: float,
    dst_h: float,
    pad_x: float,
    pad_y: float,
    scale: float,
) -> list[float]:
    if not triplets:
        return triplets

    normalized = _looks_normalized_xy_triplets(triplets)
    out = list(triplets)

    for i in range(0, len(out) - 2, 3):
        x = out[i]
        y = out[i + 1]
        conf = out[i + 2]

        if conf is None or conf <= 0:
            out[i] = 0.0
            out[i + 1] = 0.0
            out[i + 2] = 0.0 if conf is None else conf
            continue

        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(conf):
            out[i] = 0.0
            out[i + 1] = 0.0
            out[i + 2] = 0.0
            continue

        x = float(x)
        y = float(y)
        conf = float(conf)

        if normalized:
            x *= float(src_w)
            y *= float(src_h)

        # pad to match target aspect, then scale to destination
        x = (x + pad_x) * scale
        y = (y + pad_y) * scale

        out[i] = x
        out[i + 1] = y
        out[i + 2] = conf

    return out


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


class PoseKeypointResize:
    """
    Resize POSE_KEYPOINT to desired width/height without stretching:
    - If aspect ratio matches: scale directly.
    - If aspect ratio differs: pad the shorter side (centered) then scale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Input POSE_KEYPOINT to resize."}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "resize"
    CATEGORY = "ProportionChanger"

    def resize(self, pose_keypoint, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        if not pose_keypoint:
            return (
                [
                    {
                        "version": "1.0",
                        "people": [],
                        "canvas_width": width,
                        "canvas_height": height,
                    }
                ],
            )

        dst_w = float(width)
        dst_h = float(height)
        dst_ratio = dst_w / dst_h

        frames_out = []
        for frame in pose_keypoint:
            frame_out = copy.deepcopy(frame) if isinstance(frame, dict) else {}

            src_w = float(frame_out.get("canvas_width", width) or width)
            src_h = float(frame_out.get("canvas_height", height) or height)
            if src_w <= 0 or src_h <= 0:
                src_w = float(width)
                src_h = float(height)

            src_ratio = src_w / src_h

            if abs(src_ratio - dst_ratio) < 1e-8:
                pad_x = 0.0
                pad_y = 0.0
                padded_w = src_w
                padded_h = src_h
            elif src_ratio < dst_ratio:
                # pad width (left/right)
                padded_h = src_h
                padded_w = src_h * dst_ratio
                pad_x = (padded_w - src_w) / 2.0
                pad_y = 0.0
            else:
                # pad height (top/bottom)
                padded_w = src_w
                padded_h = src_w / dst_ratio
                pad_x = 0.0
                pad_y = (padded_h - src_h) / 2.0

            scale = dst_w / padded_w if padded_w > 0 else 1.0

            people = frame_out.get("people", [])
            if not isinstance(people, list):
                people = []

            for person in people:
                if not isinstance(person, dict):
                    continue
                for key in (
                    "pose_keypoints_2d",
                    "face_keypoints_2d",
                    "hand_left_keypoints_2d",
                    "hand_right_keypoints_2d",
                ):
                    triplets = person.get(key)
                    if not isinstance(triplets, list):
                        continue
                    person[key] = _resize_pose_triplets_with_pad(
                        triplets,
                        src_w=src_w,
                        src_h=src_h,
                        dst_w=dst_w,
                        dst_h=dst_h,
                        pad_x=pad_x,
                        pad_y=pad_y,
                        scale=scale,
                    )

            frame_out["canvas_width"] = width
            frame_out["canvas_height"] = height
            frames_out.append(frame_out)

        return (frames_out,)
