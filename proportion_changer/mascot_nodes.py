"""
Mascot pose nodes for ProportionChanger.

The model artifacts are downloaded from HuggingFace and executed with
ONNXRuntime. This module intentionally has no runtime dependency on the
mascot_body_detect source repository.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

import cv2
import numpy as np

try:
    import folder_paths
except ImportError:  # pragma: no cover - ComfyUI provides this at runtime.
    folder_paths = None

try:
    from comfy.utils import ProgressBar
except ImportError:  # pragma: no cover - standalone import smoke tests.
    ProgressBar = None

try:
    from ..utils import log
except ImportError:  # pragma: no cover
    log = logging.getLogger(__name__)


DEFAULT_REPO_ID = "grmchn/mascot-pose-detect"
HF_BASE_URL = f"https://huggingface.co/{DEFAULT_REPO_ID}/resolve/main"
MASCOT_BBOX_MODEL_URLS = [
    f"{HF_BASE_URL}/bbox/model.onnx",
]
MASCOT_DWPOSE_MODEL_URLS = [
    f"{HF_BASE_URL}/keypoint/dinov2_vitpose_l/model.onnx",
]
POSE_URL_TO_VARIANT = {
    f"{HF_BASE_URL}/keypoint/dinov2_vitpose_l/model.onnx": "dinov2_vitpose_l",
}
BBOX_KEYS = (
    "full",
    "head",
    "body",
    "hand_left",
    "hand_right",
    "foot_left",
    "foot_right",
)

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

COCO17_TO_DWPOSE25 = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 8, 11, 9, 12, 10, 13)

NEUTRAL_HAND_TEMPLATE_RIGHT = np.array(
    [
        (0.00, 0.00),
        (0.18, 0.12),
        (0.32, 0.26),
        (0.42, 0.40),
        (0.50, 0.52),
        (0.20, 0.55),
        (0.22, 0.72),
        (0.23, 0.83),
        (0.24, 0.92),
        (0.06, 0.60),
        (0.07, 0.78),
        (0.07, 0.90),
        (0.08, 1.00),
        (-0.08, 0.58),
        (-0.09, 0.74),
        (-0.10, 0.86),
        (-0.11, 0.95),
        (-0.20, 0.52),
        (-0.22, 0.64),
        (-0.24, 0.74),
        (-0.26, 0.82),
    ],
    dtype=np.float32,
)
NEUTRAL_HAND_TEMPLATE_LEFT = NEUTRAL_HAND_TEMPLATE_RIGHT * np.array([-1.0, 1.0], dtype=np.float32)


def _models_root() -> str:
    if folder_paths is None:
        return os.path.abspath(os.path.join("models", "mascot_body_detect"))
    return os.path.join(folder_paths.models_dir, "mascot_body_detect")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _repo_marker_path(base: str) -> str:
    return os.path.join(base, ".repo_id")


def _local_repo_matches(base: str) -> bool:
    try:
        with open(_repo_marker_path(base), "r", encoding="utf-8") as handle:
            return handle.read().strip() == DEFAULT_REPO_ID
    except FileNotFoundError:
        return False


def _write_repo_marker(base: str) -> None:
    with open(_repo_marker_path(base), "w", encoding="utf-8") as handle:
        handle.write(DEFAULT_REPO_ID)


def _download_artifacts(url: str, *, include_bbox: bool, keypoint_variant: str | None = None) -> str:
    if include_bbox and url not in MASCOT_BBOX_MODEL_URLS and keypoint_variant is None:
        raise ValueError(f"URL {url} is not in the list of allowed Mascot BBox models.")
    if keypoint_variant is not None and url not in MASCOT_DWPOSE_MODEL_URLS:
        raise ValueError(f"URL {url} is not in the list of allowed Mascot Pose models.")

    from huggingface_hub import snapshot_download

    base = _models_root()
    os.makedirs(base, exist_ok=True)
    allow_patterns = []
    required_paths = []
    if include_bbox:
        allow_patterns.append("bbox/*")
        required_paths.extend(
            [
                os.path.join(base, "bbox", "model.onnx"),
                os.path.join(base, "bbox", "classes.json"),
                os.path.join(base, "bbox", "decode_params.json"),
            ]
        )
    if keypoint_variant is not None:
        allow_patterns.append(f"keypoint/{keypoint_variant}/*")
        required_paths.extend(
            [
                os.path.join(base, "keypoint", keypoint_variant, "model.onnx"),
                os.path.join(base, "keypoint", keypoint_variant, "meta.json"),
            ]
        )

    if not _local_repo_matches(base) or not all(os.path.exists(path) for path in required_paths):
        snapshot_download(
            repo_id=DEFAULT_REPO_ID,
            repo_type="model",
            allow_patterns=allow_patterns,
            local_dir=base,
        )
        _write_repo_marker(base)
    return base


def _providers(cuda: bool) -> list[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]


def _progress(total: int):
    if ProgressBar is None:
        return None
    return ProgressBar(total)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img_rgb: np.ndarray, size_wh: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    img_h, img_w = img_rgb.shape[:2]
    target_w, target_h = size_wh
    scale = min(target_w / img_w, target_h / img_h)
    resized_w = int(round(img_w * scale))
    resized_h = int(round(img_h * scale))
    resized = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return canvas, scale, (pad_x, pad_y)


def decode_rtmdet_scale(cls_logits: np.ndarray, bbox_dist: np.ndarray, stride: int, score_thr: float) -> np.ndarray:
    height, width = cls_logits.shape[-2:]
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    center_x = (xx + 0.5) * stride
    center_y = (yy + 0.5) * stride
    scores = sigmoid(cls_logits[0])
    cls_idx = scores.argmax(axis=0)
    cls_score = scores.max(axis=0)
    keep = cls_score >= score_thr
    if not keep.any():
        return np.zeros((0, 6), dtype=np.float32)

    # Exported RTMDet ONNX bbox distances are already in input-image pixels.
    # Do not multiply them by stride; doing so expands boxes to the whole image.
    left = bbox_dist[0, 0][keep]
    top = bbox_dist[0, 1][keep]
    right = bbox_dist[0, 2][keep]
    bottom = bbox_dist[0, 3][keep]
    kept_x = center_x[keep]
    kept_y = center_y[keep]
    return np.stack(
        [
            kept_x - left,
            kept_y - top,
            kept_x + right,
            kept_y + bottom,
            cls_score[keep],
            cls_idx[keep],
        ],
        axis=1,
    ).astype(np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms_per_class(boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return boxes

    keep_global: list[int] = []
    class_ids = boxes[:, 5].astype(np.int32)
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[indices]
        order = np.argsort(-class_boxes[:, 4])
        suppressed = np.zeros(len(order), dtype=bool)
        for pos, src_order_idx in enumerate(order):
            if suppressed[pos]:
                continue
            keep_global.append(int(indices[src_order_idx]))
            for later_pos in range(pos + 1, len(order)):
                if suppressed[later_pos]:
                    continue
                if iou_xyxy(class_boxes[src_order_idx, :4], class_boxes[order[later_pos], :4]) >= iou_thr:
                    suppressed[later_pos] = True
    return boxes[sorted(keep_global)]


def _run_bbox(session, img_rgb: np.ndarray, params: dict[str, Any], score_threshold: float) -> np.ndarray:
    input_size = tuple(params["input_size"])
    padded, scale, (pad_x, pad_y) = letterbox(img_rgb, input_size)
    inp = (padded.astype(np.float32) - np.asarray(params["mean"], dtype=np.float32)) / np.asarray(
        params["std"], dtype=np.float32
    )
    inp = inp.transpose(2, 0, 1)[None]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    cls_outputs = outputs[:3]
    bbox_outputs = outputs[3:]

    boxes_by_scale = [
        decode_rtmdet_scale(cls_out, bbox_out, int(stride), score_threshold)
        for cls_out, bbox_out, stride in zip(cls_outputs, bbox_outputs, params["strides"])
    ]
    boxes = np.concatenate(boxes_by_scale, axis=0) if boxes_by_scale else np.zeros((0, 6), dtype=np.float32)
    boxes = nms_per_class(boxes, float(params["nms_iou_threshold"]))
    if len(boxes) == 0:
        return boxes

    img_h, img_w = img_rgb.shape[:2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
    return boxes


def infer_bbox_raw(
    session,
    img_rgb: np.ndarray,
    score_threshold: float,
    params: dict[str, Any],
    classes_map: dict[str, str],
) -> list[dict[str, float | str]]:
    boxes = _run_bbox(session, img_rgb, params, score_threshold)
    out: list[dict[str, float | str]] = []
    for x1, y1, x2, y2, score, class_id_raw in boxes:
        class_name = classes_map.get(str(int(class_id_raw)))
        if class_name is None:
            continue
        out.append(
            {
                "x": float(x1),
                "y": float(y1),
                "width": float(max(0.0, x2 - x1)),
                "height": float(max(0.0, y2 - y1)),
                "label": class_name,
                "score": float(score),
            }
        )
    return out


def infer_bbox_dict(
    session,
    img_rgb: np.ndarray,
    score_threshold: float,
    params: dict[str, Any],
    classes_map: dict[str, str],
) -> dict[str, list[float] | None]:
    boxes = _run_bbox(session, img_rgb, params, score_threshold)
    img_h, img_w = img_rgb.shape[:2]
    out: dict[str, list[float] | None] = {name: None for name in BBOX_KEYS}
    if len(boxes) == 0:
        return out

    labelled_boxes = []
    for x1, y1, x2, y2, score, class_id_raw in boxes:
        class_name = classes_map.get(str(int(class_id_raw)))
        if class_name in BBOX_KEYS:
            labelled_boxes.append(
                {
                    "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(score),
                    "label": class_name,
                }
            )
    if not labelled_boxes:
        return out

    roi_candidates = [box for box in labelled_boxes if box["label"] == "full"]
    if not roi_candidates:
        roi_candidates = [box for box in labelled_boxes if box["label"] == "body"]
    if not roi_candidates:
        return out

    roi = max(roi_candidates, key=lambda box: box["score"])
    roi_x1, roi_y1, roi_x2, roi_y2 = roi["xyxy"]
    roi_w = max(1.0, roi_x2 - roi_x1)
    roi_h = max(1.0, roi_y2 - roi_y1)
    # Body boxes do not always enclose hands and feet; expand fallback ROIs.
    margin_x = roi_w * (0.35 if roi["label"] == "body" else 0.05)
    margin_y = roi_h * (0.35 if roi["label"] == "body" else 0.05)
    gate = (
        max(0.0, roi_x1 - margin_x),
        max(0.0, roi_y1 - margin_y),
        min(float(img_w), roi_x2 + margin_x),
        min(float(img_h), roi_y2 + margin_y),
    )

    def inside_gate(box: dict[str, Any]) -> bool:
        x1, y1, x2, y2 = box["xyxy"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gx1, gy1, gx2, gy2 = gate
        return gx1 <= cx <= gx2 and gy1 <= cy <= gy2

    def to_yxyx_1000(box: dict[str, Any]) -> list[float]:
        x1, y1, x2, y2 = box["xyxy"]
        return [
            float(y1) / img_h * 1000.0,
            float(x1) / img_w * 1000.0,
            float(y2) / img_h * 1000.0,
            float(x2) / img_w * 1000.0,
        ]

    out[roi["label"]] = to_yxyx_1000(roi)
    for class_name in BBOX_KEYS:
        if class_name == roi["label"]:
            continue
        candidates = [box for box in labelled_boxes if box["label"] == class_name and inside_gate(box)]
        if not candidates:
            continue
        out[class_name] = to_yxyx_1000(max(candidates, key=lambda box: box["score"]))
    return out


def bbox_yxyx_1000_to_xyxy_pixel(
    bbox: list[float] | None,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    ymin, xmin, ymax, xmax = bbox
    return (
        xmin / 1000.0 * img_w,
        ymin / 1000.0 * img_h,
        xmax / 1000.0 * img_w,
        ymax / 1000.0 * img_h,
    )


def topdown_affine(
    bbox_xyxy: tuple[float, float, float, float],
    input_size: tuple[int, int],
    pad_ratio: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    target_w, target_h = input_size
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    bbox_w = max(1.0, (x2 - x1) * pad_ratio)
    bbox_h = max(1.0, (y2 - y1) * pad_ratio)
    aspect = target_w / target_h
    if bbox_w / bbox_h > aspect:
        bbox_h = bbox_w / aspect
    else:
        bbox_w = bbox_h * aspect
    crop_x1 = center_x - bbox_w / 2.0
    crop_y1 = center_y - bbox_h / 2.0
    return np.array(
        [
            [target_w / bbox_w, 0.0, -crop_x1 * target_w / bbox_w],
            [0.0, target_h / bbox_h, -crop_y1 * target_h / bbox_h],
        ],
        dtype=np.float32,
    )


def run_topdown_crop(
    session,
    img_rgb: np.ndarray,
    bbox_yxyx_0_1000: list[float] | None,
    input_size: tuple[int, int],
    pad_ratio: float,
) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
    img_h, img_w = img_rgb.shape[:2]
    bbox_xyxy = bbox_yxyx_1000_to_xyxy_pixel(bbox_yxyx_0_1000, img_w, img_h)
    if bbox_xyxy is None:
        return None, None
    matrix = topdown_affine(bbox_xyxy, input_size, pad_ratio)
    crop = cv2.warpAffine(
        img_rgb,
        matrix,
        input_size,
        flags=cv2.INTER_LINEAR,
        borderValue=(114, 114, 114),
    )
    inp = (crop.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    inp = inp.transpose(2, 0, 1)[None]
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inp}), matrix


def decode_heatmap(heatmap: np.ndarray, heatmap_to_input_scale: float) -> np.ndarray:
    keypoints, heat_h, heat_w = heatmap.shape
    out = np.zeros((keypoints, 3), dtype=np.float32)
    for idx in range(keypoints):
        flat_idx = int(heatmap[idx].argmax())
        y, x = divmod(flat_idx, heat_w)
        score = float(heatmap[idx, y, x])
        dx = 0.0
        dy = 0.0
        if 1 <= x < heat_w - 1:
            dx = 0.25 * float(np.sign(heatmap[idx, y, x + 1] - heatmap[idx, y, x - 1]))
        if 1 <= y < heat_h - 1:
            dy = 0.25 * float(np.sign(heatmap[idx, y + 1, x] - heatmap[idx, y - 1, x]))
        out[idx] = [
            (x + dx) * heatmap_to_input_scale,
            (y + dy) * heatmap_to_input_scale,
            score,
        ]
    return out


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float) -> np.ndarray:
    x = simcc_x[0]
    y = simcc_y[0]
    keypoints = x.shape[0]
    out = np.zeros((keypoints, 3), dtype=np.float32)
    for idx in range(keypoints):
        x_idx = int(x[idx].argmax())
        y_idx = int(y[idx].argmax())
        score = min(float(x[idx, x_idx]), float(y[idx, y_idx]))
        out[idx] = [x_idx / split_ratio, y_idx / split_ratio, score]
    return out


def crop_to_image_points(kp_crop_xy_score: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    out = np.zeros_like(kp_crop_xy_score, dtype=np.float32)
    for idx, (x, y, score) in enumerate(kp_crop_xy_score):
        mapped = inv_matrix @ np.array([float(x), float(y), 1.0], dtype=np.float32)
        out[idx] = [float(mapped[0]), float(mapped[1]), float(score)]
    return out


def coco17_to_dwpose25(kp_17: list[list[float]], derive_neck: bool = True) -> list[list[float]]:
    arr = np.asarray(kp_17, dtype=np.float32)
    out = np.zeros((25, 3), dtype=np.float32)
    for coco_idx, dw_idx in enumerate(COCO17_TO_DWPOSE25):
        out[dw_idx] = arr[coco_idx]

    if derive_neck:
        nose = arr[0]
        left_shoulder = arr[5]
        right_shoulder = arr[6]
        if left_shoulder[2] > 0 and right_shoulder[2] > 0:
            midpoint = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
            if nose[2] > 0:
                centroid = (nose[:2] + left_shoulder[:2] + right_shoulder[:2]) / 3.0
                out[1, :2] = (midpoint + centroid) / 2.0
                out[1, 2] = float(min(nose[2], left_shoulder[2], right_shoulder[2]))
            else:
                out[1, :2] = midpoint
                out[1, 2] = float(min(left_shoulder[2], right_shoulder[2]))
    return out.tolist()


def derive_toes_from_bbox(
    kp_25_pixel: list[list[float]],
    foot_left: list[float] | None,
    foot_right: list[float] | None,
    img_w: int,
    img_h: int,
) -> list[list[float]]:
    arr = np.asarray(kp_25_pixel, dtype=np.float32)
    for toe_idx, ankle_idx, bbox in (
        (18, 13, foot_right),
        (19, 10, foot_left),
    ):
        arr[toe_idx] = [0.0, 0.0, 0.0]
        if bbox is None:
            continue
        ymin, xmin, ymax, xmax = bbox
        arr[toe_idx, 0] = (xmin + xmax) / 2.0 / 1000.0 * img_w
        arr[toe_idx, 1] = (ymin + ymax) / 2.0 / 1000.0 * img_h
        ankle_score = float(arr[ankle_idx, 2])
        arr[toe_idx, 2] = ankle_score if ankle_score > 0.0 else 0.5
    return arr.tolist()


def infer_keypoints_pixel(
    session,
    img_rgb: np.ndarray,
    bbox_dict: dict[str, list[float] | None],
    kp_meta: dict[str, Any],
    variant: str,
) -> list[list[float]]:
    img_h, img_w = img_rgb.shape[:2]
    roi = bbox_dict.get("full") or bbox_dict.get("body")
    input_size = tuple(kp_meta["input_size"])
    pad_ratio = float(kp_meta.get("topdown_bbox_pad_ratio", 1.25))
    outputs, matrix = run_topdown_crop(session, img_rgb, roi, input_size, pad_ratio)
    if outputs is None or matrix is None:
        return [[0.0, 0.0, 0.0] for _ in range(25)]

    if variant in {"vitpose_l", "dinov2_vitpose_l"}:
        scale = float(kp_meta.get("heatmap_to_input_scale", 4.0))
        kp_crop = decode_heatmap(outputs[0][0], scale)
        kp_image_17 = crop_to_image_points(kp_crop, matrix)
        kp_25 = coco17_to_dwpose25(kp_image_17.tolist(), derive_neck=True)
    elif variant == "rtmpose_s":
        kp_crop = decode_simcc(outputs[0], outputs[1], float(kp_meta["split_ratio"]))
        kp_image_25 = crop_to_image_points(kp_crop, matrix)
        kp_25 = kp_image_25[:25].tolist()
        if len(kp_25) < 25:
            kp_25.extend([[0.0, 0.0, 0.0] for _ in range(25 - len(kp_25))])
    else:
        raise ValueError(f"unknown keypoint variant: {variant}")

    kp_25 = derive_toes_from_bbox(
        kp_25,
        bbox_dict.get("foot_left"),
        bbox_dict.get("foot_right"),
        img_w=img_w,
        img_h=img_h,
    )
    for idx in range(20, 25):
        kp_25[idx] = [0.0, 0.0, 0.0]
    return kp_25


def _bbox_short_side_pixel(bbox: list[float], img_w: int, img_h: int) -> float:
    ymin, xmin, ymax, xmax = bbox
    width = max(1.0, (xmax - xmin) / 1000.0 * img_w)
    height = max(1.0, (ymax - ymin) / 1000.0 * img_h)
    return min(width, height)


def _build_hand_2d(
    hand_bbox: list[float] | None,
    kp_25_pixel: list[list[float]],
    *,
    elbow_idx: int,
    wrist_idx: int,
    side: str,
    img_w: int,
    img_h: int,
    canvas_w: int,
    canvas_h: int,
) -> list[float]:
    if hand_bbox is None:
        return [0.0] * (21 * 3)

    arr = np.asarray(kp_25_pixel, dtype=np.float32)
    wrist = arr[wrist_idx]
    elbow = arr[elbow_idx]
    if wrist[2] <= 0:
        ymin, xmin, ymax, xmax = hand_bbox
        wrist_xy = np.array(
            [
                (xmin + xmax) / 2.0 / 1000.0 * img_w,
                (ymin + ymax) / 2.0 / 1000.0 * img_h,
            ],
            dtype=np.float32,
        )
    else:
        wrist_xy = wrist[:2].astype(np.float32)

    forearm_len = 0.0
    if elbow[2] > 0 and wrist[2] > 0:
        direction = wrist[:2] - elbow[:2]
        forearm_len = float(np.linalg.norm(direction))
    else:
        direction = np.array([0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0], dtype=np.float32)
    else:
        direction = direction / norm

    dx, dy = float(direction[0]), float(direction[1])
    theta = math.atan2(-dx, dy)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    template = NEUTRAL_HAND_TEMPLATE_LEFT if side == "left" else NEUTRAL_HAND_TEMPLATE_RIGHT
    scale = _bbox_short_side_pixel(hand_bbox, img_w, img_h)
    caps = [max(8.0, min(float(img_w), float(img_h)) * 0.18)]
    if forearm_len > 1e-6:
        caps.append(max(8.0, forearm_len * 1.25))
    scale = min(scale, *caps)
    points = template @ rot.T * scale + wrist_xy

    out: list[float] = []
    for x, y in points:
        out.extend([float(x) / img_w * canvas_w, float(y) / img_h * canvas_h, 0.5])
    return out


def build_person_dict(
    stage1_bbox: dict[str, list[float] | None],
    kp_25_pixel: list[list[float]],
    canvas_width: int,
    canvas_height: int,
    img_width: int,
    img_height: int,
) -> dict[str, list[float]]:
    pose_kp: list[float] = []
    for idx in range(25):
        if 20 <= idx < 25:
            pose_kp.extend([0.0, 0.0, 0.0])
            continue
        x, y, score = kp_25_pixel[idx]
        pose_kp.extend(
            [
                float(x) / img_width * canvas_width,
                float(y) / img_height * canvas_height,
                float(score),
            ]
        )

    right_hand = _build_hand_2d(
        stage1_bbox.get("hand_right"),
        kp_25_pixel,
        elbow_idx=3,
        wrist_idx=4,
        side="right",
        img_w=img_width,
        img_h=img_height,
        canvas_w=canvas_width,
        canvas_h=canvas_height,
    )
    left_hand = _build_hand_2d(
        stage1_bbox.get("hand_left"),
        kp_25_pixel,
        elbow_idx=6,
        wrist_idx=7,
        side="left",
        img_w=img_width,
        img_h=img_height,
        canvas_w=canvas_width,
        canvas_h=canvas_height,
    )

    return {
        "pose_keypoints_2d": pose_kp,
        "face_keypoints_2d": [0.0] * (70 * 3),
        "hand_left_keypoints_2d": left_hand,
        "hand_right_keypoints_2d": right_hand,
    }


class DownloadAndLoadMascotDWPoseModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    MASCOT_DWPOSE_MODEL_URLS,
                    {"default": f"{HF_BASE_URL}/keypoint/dinov2_vitpose_l/model.onnx"},
                ),
                "cuda": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use CUDAExecutionProvider when available."},
                ),
            },
            "optional": {
                "warmup": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Run a small dummy inference after loading."},
                ),
            },
        }

    RETURN_TYPES = ("MASCOT_POSE_MODEL",)
    RETURN_NAMES = ("mascot_pose_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "ProportionChanger"
    DESCRIPTION = "Download and load a mascot pose ONNX model from HuggingFace."

    def loadmodel(self, url: str, cuda: bool, warmup: bool = True):
        if folder_paths is None:
            raise RuntimeError("folder_paths is required inside ComfyUI to locate the model directory")

        import onnxruntime as ort

        keypoint_variant = POSE_URL_TO_VARIANT.get(url)
        if keypoint_variant is None:
            raise ValueError(f"URL {url} is not in the list of allowed Mascot Pose models.")

        # The current top-down keypoint ONNX still needs a body ROI. Keep the bbox
        # sidecar inside this handle so the pose detector remains a single-input
        # node while exposing a separate bbox loader for bbox-only workflows.
        base = _download_artifacts(url, include_bbox=True, keypoint_variant=keypoint_variant)
        bbox_onnx = os.path.join(base, "bbox", "model.onnx")
        kp_onnx = os.path.join(base, "keypoint", keypoint_variant, "model.onnx")
        providers = _providers(bool(cuda))
        bbox_session = ort.InferenceSession(bbox_onnx, providers=providers)
        kp_session = ort.InferenceSession(kp_onnx, providers=providers)

        if warmup:
            log.info("Warming up Mascot Pose model...")
            bbox_input = bbox_session.get_inputs()[0]
            kp_input = kp_session.get_inputs()[0]
            bbox_shape = [1 if not isinstance(dim, int) else dim for dim in bbox_input.shape]
            kp_shape = [1 if not isinstance(dim, int) else dim for dim in kp_input.shape]
            bbox_session.run(None, {bbox_input.name: np.zeros(bbox_shape, dtype=np.float32)})
            kp_session.run(None, {kp_input.name: np.zeros(kp_shape, dtype=np.float32)})
            log.info("Mascot Pose model warmed up")

        return (
            {
                "variant": keypoint_variant,
                "url": url,
                "cuda": bool(cuda),
                "bbox_session": bbox_session,
                "bbox_decode": _read_json(os.path.join(base, "bbox", "decode_params.json")),
                "bbox_classes": _read_json(os.path.join(base, "bbox", "classes.json")),
                "kp_session": kp_session,
                "kp_meta": _read_json(os.path.join(base, "keypoint", keypoint_variant, "meta.json")),
            },
        )


class DownloadAndLoadMascotBBoxModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (MASCOT_BBOX_MODEL_URLS, {"default": f"{HF_BASE_URL}/bbox/model.onnx"}),
                "cuda": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use CUDAExecutionProvider when available."},
                ),
            },
            "optional": {
                "warmup": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Run a small dummy inference after loading."},
                ),
            },
        }

    RETURN_TYPES = ("MASCOT_BBOX_MODEL",)
    RETURN_NAMES = ("mascot_bbox_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "ProportionChanger"
    DESCRIPTION = "Download and load a mascot bbox ONNX model from HuggingFace."

    def loadmodel(self, url: str, cuda: bool, warmup: bool = True):
        if folder_paths is None:
            raise RuntimeError("folder_paths is required inside ComfyUI to locate the model directory")

        import onnxruntime as ort

        if url not in MASCOT_BBOX_MODEL_URLS:
            raise ValueError(f"URL {url} is not in the list of allowed Mascot BBox models.")

        base = _download_artifacts(url, include_bbox=True)
        bbox_onnx = os.path.join(base, "bbox", "model.onnx")
        session = ort.InferenceSession(bbox_onnx, providers=_providers(bool(cuda)))

        if warmup:
            log.info("Warming up Mascot BBox model...")
            input_info = session.get_inputs()[0]
            input_shape = [1 if not isinstance(dim, int) else dim for dim in input_info.shape]
            session.run(None, {input_info.name: np.zeros(input_shape, dtype=np.float32)})
            log.info("Mascot BBox model warmed up")

        return (
            {
                "url": url,
                "cuda": bool(cuda),
                "bbox_session": session,
                "bbox_decode": _read_json(os.path.join(base, "bbox", "decode_params.json")),
                "bbox_classes": _read_json(os.path.join(base, "bbox", "classes.json")),
            },
        )


class MascotDWPoseDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mascot_pose_model": ("MASCOT_POSE_MODEL", {"tooltip": "Output of (Down)Load Mascot Pose Model"}),
                "image": ("IMAGE", {"tooltip": "Input image, RGB float [0,1]."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "detect"
    CATEGORY = "ProportionChanger"
    DESCRIPTION = "Detect mascot body pose as ProportionChanger POSE_KEYPOINT."

    def detect(self, mascot_pose_model, image, width: int, height: int, threshold: float):
        out_frames = []
        pbar = _progress(len(image))

        for img_t in image:
            img_np = np.clip(img_t.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            img_h, img_w = img_np.shape[:2]
            bbox_dict = infer_bbox_dict(
                mascot_pose_model["bbox_session"],
                img_np,
                threshold,
                mascot_pose_model["bbox_decode"],
                mascot_pose_model["bbox_classes"],
            )
            person = None
            if bbox_dict.get("full") is not None or bbox_dict.get("body") is not None:
                kp_25_pixel = infer_keypoints_pixel(
                    mascot_pose_model["kp_session"],
                    img_np,
                    bbox_dict,
                    mascot_pose_model["kp_meta"],
                    mascot_pose_model["variant"],
                )
                person = build_person_dict(
                    stage1_bbox=bbox_dict,
                    kp_25_pixel=kp_25_pixel,
                    canvas_width=int(width),
                    canvas_height=int(height),
                    img_width=img_w,
                    img_height=img_h,
                )

            out_frames.append(
                {
                    "version": "1.0",
                    "people": [person] if person else [],
                    "canvas_width": int(width),
                    "canvas_height": int(height),
                }
            )
            if pbar is not None:
                pbar.update(1)

        return (out_frames,)


class MascotBBoxDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mascot_bbox_model": ("MASCOT_BBOX_MODEL", {"tooltip": "Output of (Down)Load Mascot BBox Model"}),
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "class_filter": (
                    ["all", "full", "head", "body", "hand_left", "hand_right", "foot_left", "foot_right"],
                    {"default": "all"},
                ),
                "max_detections": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "detect"
    CATEGORY = "ProportionChanger"
    DESCRIPTION = "Detect mascot part bounding boxes as comfy-core BOUNDING_BOX."

    def detect(self, mascot_bbox_model, image, threshold: float, class_filter: str, max_detections: int):
        out_frames = []
        pbar = _progress(len(image))
        for img_t in image:
            img_np = np.clip(img_t.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            boxes = infer_bbox_raw(
                mascot_bbox_model["bbox_session"],
                img_np,
                threshold,
                mascot_bbox_model["bbox_decode"],
                mascot_bbox_model["bbox_classes"],
            )
            if class_filter != "all":
                boxes = [box for box in boxes if box["label"] == class_filter]
            boxes = sorted(boxes, key=lambda box: -float(box["score"]))[: int(max_detections)]
            out_frames.append(boxes)
            if pbar is not None:
                pbar.update(1)
        return (out_frames,)


class ConvertToSCAILPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("scail_pose",)
    FUNCTION = "convert"
    CATEGORY = "ProportionChanger"
    DESCRIPTION = "Convert 25-point POSE_KEYPOINT to SCAIL-Pose data."

    @staticmethod
    def _normalized_xy_scores(raw, num_points, canvas_width, canvas_height, *, skip_points=0):
        if len(raw) < (skip_points + num_points) * 3:
            return None, None

        arr = np.asarray(raw, dtype=np.float32).reshape(-1, 3)[skip_points : skip_points + num_points]
        coords = arr[:, :2].copy()
        scores = arr[:, 2].copy()

        valid_x = np.isfinite(coords[:, 0])
        valid_y = np.isfinite(coords[:, 1])
        valid_score = np.isfinite(scores)
        invalid = ~(valid_x & valid_y & valid_score)

        visible = (scores > 0) & ~invalid
        visible_coords = coords[visible]
        already_normalized = True
        if visible_coords.size > 0:
            already_normalized = float(np.nanmax(np.abs(visible_coords))) <= 2.0

        if not already_normalized and canvas_width > 0:
            coords[:, 0] /= float(canvas_width)
        if not already_normalized and canvas_height > 0:
            coords[:, 1] /= float(canvas_height)

        coords[invalid] = 0.0
        scores[invalid] = 0.0
        return coords, scores

    def convert(self, keypoints):
        num_body = 18
        num_face = 68
        num_hand = 21
        out_frames = []

        for frame in keypoints or []:
            people = frame.get("people", []) if isinstance(frame, dict) else []
            people = people[:1] if isinstance(people, list) else []

            bodies = []
            body_scores = []
            hands = []
            hand_scores = []
            faces = []
            face_scores = []

            for person in people:
                if not isinstance(person, dict):
                    continue

                canvas_width = float(frame.get("canvas_width", 1) or 1)
                canvas_height = float(frame.get("canvas_height", 1) or 1)

                pose_raw = person.get("pose_keypoints_2d") or []
                body_xy, body_score = self._normalized_xy_scores(
                    pose_raw, num_body, canvas_width, canvas_height
                )
                if body_xy is None:
                    continue
                bodies.append(body_xy)
                body_scores.append(body_score)

                face_raw = person.get("face_keypoints_2d") or []
                face_skip = 1 if len(face_raw) >= (num_face + 1) * 3 else 0
                face_xy, face_score = self._normalized_xy_scores(
                    face_raw, num_face, canvas_width, canvas_height, skip_points=face_skip
                )
                if face_xy is not None:
                    faces.append(face_xy)
                    face_scores.append(face_score)
                else:
                    faces.append(np.zeros((num_face, 2), dtype=np.float32))
                    face_scores.append(np.zeros(num_face, dtype=np.float32))

                # SCAIL-Pose's VitPose converter stores hands as right, then left.
                for hand_key in ("hand_right_keypoints_2d", "hand_left_keypoints_2d"):
                    hand_raw = person.get(hand_key) or []
                    hand_xy, hand_score = self._normalized_xy_scores(
                        hand_raw, num_hand, canvas_width, canvas_height
                    )
                    if hand_xy is not None:
                        hands.append(hand_xy)
                        hand_scores.append(hand_score)

            out_frames.append(
                {
                    "bodies": {
                        "candidate": np.asarray(bodies, dtype=np.float32)
                        if bodies
                        else np.zeros((0, num_body, 2), dtype=np.float32),
                        "subset": np.asarray(
                            [
                                np.where(score > 0.3, np.arange(num_body), -1)
                                for score in body_scores
                            ],
                            dtype=np.float32,
                        )
                        if body_scores
                        else np.zeros((0, num_body), dtype=np.float32),
                    },
                    "hands": np.asarray(hands, dtype=np.float32)
                    if hands
                    else np.zeros((0, num_hand, 2), dtype=np.float32),
                    "faces": np.asarray(faces, dtype=np.float32)
                    if faces
                    else np.zeros((0, num_face, 2), dtype=np.float32),
                    "body_score": np.asarray(body_scores, dtype=np.float32)
                    if body_scores
                    else np.zeros((0, num_body), dtype=np.float32),
                    "hand_score": np.asarray(hand_scores, dtype=np.float32)
                    if hand_scores
                    else np.zeros((0, num_hand), dtype=np.float32),
                    "face_score": np.asarray(face_scores, dtype=np.float32)
                    if face_scores
                    else np.zeros((0, num_face), dtype=np.float32),
                }
            )

        return ({"poses": out_frames, "swap_hands": True},)
