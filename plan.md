# Plan: PoseData to PoseKeypoint converter node

## Goal
Add a ComfyUI node (e.g. `PoseDataToPoseKeypoint`) inside ProportionChanger that converts WanAnimatePreprocess `POSEDATA` into ProportionChanger/UniAnimate-style `POSE_KEYPOINT` so downstream nodes can reuse DWPose-format keypoints without needing WanAnimate code.

## Expected IO
- Inputs (match Draw ViT Pose for familiarity):
  - `pose_data` (POSEDATA) – produced by Pose and Face Detection.
  - `width` (INT) – canvas width to scale normalized coords to pixels.
  - `height` (INT) – canvas height.
- Output:
  - `pose_keypoint` (POSE_KEYPOINT) – list of frame dicts following UniAnimate/OpenPose style used in `ProportionChangerDWPoseDetector`.

## Source formats to bridge
- `pose_data["pose_metas"]`: list of `AAPoseMeta` or humanapi meta clones; body/hand/face keypoints are **normalized 0–1** and packed as numpy arrays with confidences.
- ProportionChanger expects per-frame dict:
  ```json
  {
    "version": "1.0",
    "people": [
      {
        "pose_keypoints_2d": [x,y,conf]*25,
        "face_keypoints_2d": [x,y,conf]*70,
        "hand_left_keypoints_2d": [x,y,conf]*21,
        "hand_right_keypoints_2d": [x,y,conf]*21
      }
    ],
    "canvas_width": W,
    "canvas_height": H
  }
  ```
- Body point count mismatch: AAPoseMeta body has 20 keypoints (19-22 are averaged joints, no toes). DWPose/UniAnimate uses 25 (includes toes idx 19–24). Plan to map available 20 into first 20 slots and pad toes (19–24) with zeros/conf=0.

## Conversion rules
1) For each `meta` in `pose_data["pose_metas"]`:
   - Scale all keypoints by `width`/`height` to pixels.
   - Body: take `meta.kps_body` (shape 20x2) + `meta.kps_body_p` to build 25*3 list; pad missing 5 joints (toe L/R and ankle variants) with 0s and conf 0.
- Face: `meta.kps_face` (69 pts) -> first 69; pad to 70 with zeros; confidences from `kps_face_p` if available else 1.
   - Hands: `meta.kps_lhand`, `kps_rhand` (21 pts each); pad if missing; use `kps_*_p` for conf else 1.
   - Build `people` entry; if no valid body points (all conf 0), emit empty `people` list.
2) Append `canvas_width/height` from input args.
3) Return list ordered to match frames.

## Node design
- New file: add class `PoseDataToPoseKeypoint` under `proportion_changer` package (e.g. `converter_nodes.py` or existing util file) and expose via `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` in `nodes.py`.
- Category: `ProportionChanger`.
- INPUT_TYPES same three fields; RETURN_TYPES (`POSE_KEYPOINT`,) RETURN_NAMES (`pose_keypoint`,).
- Keep tensor-free; operates on Python lists/numpy.
- Minimal deps: reuse numpy if already available; avoid torch.

## Edge cases to handle
- Missing `kps_face` or hands: pad with zeros/conf=0.
- NaN/None in keypoints: coerce to 0, conf=0.
- Empty `pose_metas`: return `[ {version:"1.0", people:[], canvas_width:W, canvas_height:H} ]`.
- Width/height ≤0: fail fast with ValueError.

## Tests / validation
- Add lightweight unit test under `test/` to feed fake `pose_data` with one meta and assert lengths (75/210/63/63) and canvas sizes.
- (Optional) round-trip check: feed output to `PoseKeypointPreview` to ensure JSON renders; manual workflow example in `example_workflows`.

## Integration steps
1) Implement converter node code.
2) Wire into `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`.
3) Add doc blurb to `README.md` (usage + expected inputs).
4) Add test and run `pytest` (or targeted script) if available.

## Reference file paths (relative to this plan)
- `../ComfyUI-WanAnimatePreprocess/nodes.py` — DrawViTPose input schema (`pose_data`, `width`, `height`), PoseAndFaceDetection return structure.
- `../ComfyUI-WanAnimatePreprocess/pose_utils/pose2d_utils.py` — `AAPoseMeta` structure, keypoint ordering, `load_from_meta` and related helpers.
- `../ComfyUI-WanAnimatePreprocess/pose_utils/human_visualization.py` (optional) — shows how body/hand indices map to drawing; useful to confirm joint ordering if needed.
