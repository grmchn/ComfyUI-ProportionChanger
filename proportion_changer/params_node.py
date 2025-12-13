"""
ProportionChangerParams node - Direct POSE_KEYPOINT editing node
Ported from ComfyUI-ultimate-openpose-editor-toyxyz

Adjusts pose_keypoint data with key parameters, performing only keypoint transformation without image generation
"""

import json
import copy
import math


class ProportionChangerParams:
    """
    Parameter adjustment node for direct POSE_KEYPOINT editing
    Performs only keypoint transformation without image rendering
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "pose_keypoint": ("POSE_KEYPOINT", {"default": None}),
                
                # === BODY KEYPOINTS (top to bottom order) ===
                "head_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "neck_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "shoulder_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "arm_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "torso_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pelvis_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "leg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # === FACE KEYPOINTS (face top to bottom order) ===
                "face_shape_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "eyebrow_height": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "left_eyebrow_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "right_eyebrow_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "eye_distance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "eye_height": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "left_eye_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "right_eye_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "nose_scale_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "mouth_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                
                # === HAND KEYPOINTS ===
                "hands_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                
                # === GLOBAL TRANSFORM ===
                "overall_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "rotate_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "translate_x": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "translate_y": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("changed_pose_keypoint",)
    FUNCTION = "adjust_pose_keypoint"
    CATEGORY = "ProportionChanger"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        パラメータが変更された場合は必ず再実行されるようにする
        ComfyUIのキャッシュ回避のため
        """
        # Hash parameter values to detect changes
        import hashlib
        param_str = str(sorted(kwargs.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def adjust_pose_keypoint(self, pose_keypoint=None, 
                            pelvis_scale=1.0, torso_scale=1.0, neck_scale=1.0, head_scale=1.0,
                            eye_distance_scale=1.0, eye_height=0.0, eyebrow_height=0.0,
                            left_eye_scale=1.0, right_eye_scale=1.0, left_eyebrow_scale=1.0,
                            right_eyebrow_scale=1.0, mouth_scale=1.0, nose_scale_face=1.0,
                            face_shape_scale=1.0, shoulder_scale=1.0, arm_scale=1.0,
                            leg_scale=1.0, hands_scale=1.0, overall_scale=1.0,
                            rotate_angle=0.0, translate_x=0.0, translate_y=0.0):
        """
        pose_keypointデータを各種パラメータで調整
        
        Args:
            pose_keypoint: 入力POSE_KEYPOINTデータ
            **params: 調整パラメータ
            
        Returns:
            adjusted_pose_keypoint: 調整後のPOSE_KEYPOINTデータ
        """
        # Convert parameters to dictionary format
        params = {
            "pelvis_scale": pelvis_scale,
            "torso_scale": torso_scale,
            "neck_scale": neck_scale,
            "head_scale": head_scale,
            "eye_distance_scale": eye_distance_scale,
            "eye_height": eye_height,
            "eyebrow_height": eyebrow_height,
            "left_eye_scale": left_eye_scale,
            "right_eye_scale": right_eye_scale,
            "left_eyebrow_scale": left_eyebrow_scale,
            "right_eyebrow_scale": right_eyebrow_scale,
            "mouth_scale": mouth_scale,
            "nose_scale_face": nose_scale_face,
            "face_shape_scale": face_shape_scale,
            "shoulder_scale": shoulder_scale,
            "arm_scale": arm_scale,
            "leg_scale": leg_scale,
            "hands_scale": hands_scale,
            "overall_scale": overall_scale,
            "rotate_angle": rotate_angle,
            "translate_x": translate_x,
            "translate_y": translate_y
        }
        
        if pose_keypoint:
            
            # Check input data details
            if isinstance(pose_keypoint, dict):
                frame_data = pose_keypoint
            elif isinstance(pose_keypoint, list) and len(pose_keypoint) > 0:
                frame_data = pose_keypoint[0]
            else:
                frame_data = None
                
            if frame_data and "people" in frame_data and len(frame_data["people"]) > 0:
                person = frame_data["people"][0]
                if "pose_keypoints_2d" in person and len(person["pose_keypoints_2d"]) >= 18:
                    # 肩幅をチェック（Right Shoulder: index 2, Left Shoulder: index 5）
                    kps = person["pose_keypoints_2d"]
                    if len(kps) >= 18:
                        r_shoulder_x, r_shoulder_y = kps[6], kps[7]  # index 2 * 3
                        l_shoulder_x, l_shoulder_y = kps[15], kps[16]  # index 5 * 3
                        shoulder_width = ((r_shoulder_x - l_shoulder_x)**2 + (r_shoulder_y - l_shoulder_y)**2)**0.5
        
        if not pose_keypoint:
            return (pose_keypoint,)
        
        # Do nothing if all parameters are default values
        default_values = {
            "pelvis_scale": 1.0, "torso_scale": 1.0, "neck_scale": 1.0, "head_scale": 1.0,
            "eye_distance_scale": 1.0, "eye_height": 0.0, "eyebrow_height": 0.0,
            "left_eye_scale": 1.0, "right_eye_scale": 1.0, "left_eyebrow_scale": 1.0,
            "right_eyebrow_scale": 1.0, "mouth_scale": 1.0, "nose_scale_face": 1.0,
            "face_shape_scale": 1.0, "shoulder_scale": 1.0, "arm_scale": 1.0,
            "leg_scale": 1.0, "hands_scale": 1.0, "overall_scale": 1.0,
            "rotate_angle": 0.0, "translate_x": 0.0, "translate_y": 0.0
        }
        
        # Check if all values are default
        all_default = True
        for key, default_val in default_values.items():
            if abs(params.get(key, default_val) - default_val) > 0.001:  # Allow small numerical errors
                all_default = False
                break
        
        if all_default:
            # Execute processing even with default values (process from original data, not cached modified data)
            pass
        
        # POSE_KEYPOINTの形式を正規化 (dict形式の場合はリストに変換)
        # Important: Always create deep copy to never modify the original object
        if isinstance(pose_keypoint, dict):
            adjusted_pose = [copy.deepcopy(pose_keypoint)]
        elif isinstance(pose_keypoint, list):
            adjusted_pose = copy.deepcopy(pose_keypoint)
        else:
            # Always return a copy even for unknown types
            return (copy.deepcopy(pose_keypoint),)
            
        
        # Process each frame data
        for frame_data in adjusted_pose:
            if not isinstance(frame_data, dict) or "people" not in frame_data:
                continue
                
            canvas_width = frame_data.get("canvas_width", 512)
            canvas_height = frame_data.get("canvas_height", 768)
            
            # Adjust pose data for each person
            for person in frame_data["people"]:
                self._adjust_person_pose(person, canvas_width, canvas_height, **params)
        
        
        # If input was dict format, return as single dict
        # Important: Always return as new object to break references to original data
        if isinstance(pose_keypoint, dict):
            return (adjusted_pose[0],)
        else:
            return (adjusted_pose,)
    
    def _adjust_person_pose(self, person, canvas_width, canvas_height, **params):
        """
        単一人物のポーズデータを調整
        """
        # Get basic parameters (set default values too)
        overall_scale = params.get("overall_scale", 1.0)
        rotate_angle = params.get("rotate_angle", 0.0)
        translate_x = params.get("translate_x", 0.0)
        translate_y = params.get("translate_y", 0.0)
        
        # 手首の移動量を記録するため、Body keypoints調整前の位置を保存
        original_wrist_positions = {}
        if "pose_keypoints_2d" in person and person["pose_keypoints_2d"]:
            kps = person["pose_keypoints_2d"]
            # 手首の位置を記録: Left Wrist (index 7) and Right Wrist (index 4)
            if len(kps) >= 24:  # 8 * 3 = 24 (index 7まで確保)
                original_wrist_positions["left"] = [kps[21], kps[22]]  # index 7 * 3
            if len(kps) >= 15:  # 5 * 3 = 15 (index 4まで確保)
                original_wrist_positions["right"] = [kps[12], kps[13]]  # index 4 * 3
        
        # Body keypoints調整
        if "pose_keypoints_2d" in person and person["pose_keypoints_2d"]:
            person["pose_keypoints_2d"] = self._adjust_body_keypoints(
                person["pose_keypoints_2d"], canvas_width, canvas_height, **params
            )
        
        # 手首の移動量を計算
        wrist_movements = {}
        if "pose_keypoints_2d" in person and person["pose_keypoints_2d"]:
            kps = person["pose_keypoints_2d"]
            # 調整後の手首位置と比較
            if "left" in original_wrist_positions and len(kps) >= 24:
                new_left_wrist = [kps[21], kps[22]]
                wrist_movements["left"] = [
                    new_left_wrist[0] - original_wrist_positions["left"][0],
                    new_left_wrist[1] - original_wrist_positions["left"][1]
                ]
            if "right" in original_wrist_positions and len(kps) >= 15:
                new_right_wrist = [kps[12], kps[13]]
                wrist_movements["right"] = [
                    new_right_wrist[0] - original_wrist_positions["right"][0],
                    new_right_wrist[1] - original_wrist_positions["right"][1]
                ]
        
        # Face keypoints調整
        if "face_keypoints_2d" in person and person["face_keypoints_2d"]:
            person["face_keypoints_2d"] = self._adjust_face_keypoints(
                person["face_keypoints_2d"], canvas_width, canvas_height, **params
            )
        
        # Hand keypoints調整（手首の移動量も考慮）
        if "hand_left_keypoints_2d" in person and person["hand_left_keypoints_2d"]:
            person["hand_left_keypoints_2d"] = self._adjust_hand_keypoints(
                person["hand_left_keypoints_2d"], canvas_width, canvas_height, 
                wrist_movement=wrist_movements.get("left", [0.0, 0.0]), **params
            )
        
        if "hand_right_keypoints_2d" in person and person["hand_right_keypoints_2d"]:
            person["hand_right_keypoints_2d"] = self._adjust_hand_keypoints(
                person["hand_right_keypoints_2d"], canvas_width, canvas_height,
                wrist_movement=wrist_movements.get("right", [0.0, 0.0]), **params
            )
        
        # 全体変換を最後に適用
        self._apply_global_transform(
            person, canvas_width, canvas_height, 
            overall_scale, rotate_angle, translate_x, translate_y
        )
    
    def _adjust_body_keypoints(self, keypoints, canvas_width, canvas_height, **params):
        """
        Body keypointsの調整処理
        toyxyzの実装を参考にした基本的な部位別スケーリング
        全ての変換を元のキーポイントを基準に独立して計算
        """
        if not keypoints or len(keypoints) < 54:  # 18 keypoints * 3 = 54
            return keypoints
            
        # 元のキーポイントをコピー（参照用）
        original = keypoints.copy()
        adjusted = keypoints.copy()
        
        # DWPose keypoint definitions (corrected based on actual data structure)
        KP = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "LHip": 8, "LKnee": 9,
            "LAnkle": 10, "RHip": 11, "RKnee": 12, "RAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17
        }
        
        def get_point(kps, index):
            if index * 3 + 2 >= len(kps) or kps[index * 3 + 2] == 0:
                return None
            return [kps[index * 3], kps[index * 3 + 1]]
        
        def set_point(kps, index, point):
            if index * 3 + 1 < len(kps) and point is not None:
                kps[index * 3] = point[0]
                kps[index * 3 + 1] = point[1]
        
        # Get scaling parameters
        pelvis_scale = params.get("pelvis_scale", 1.0)
        torso_scale = params.get("torso_scale", 1.0)
        neck_scale = params.get("neck_scale", 1.0)
        head_scale = params.get("head_scale", 1.0)
        shoulder_scale = params.get("shoulder_scale", 1.0)
        arm_scale = params.get("arm_scale", 1.0)
        leg_scale = params.get("leg_scale", 1.0)
        
        # Get original reference points (always from original keypoints)
        orig_r_hip = get_point(original, KP["RHip"])
        orig_l_hip = get_point(original, KP["LHip"])
        orig_neck = get_point(original, KP["Neck"])
        orig_nose = get_point(original, KP["Nose"])
        
        if not all([orig_r_hip, orig_l_hip, orig_neck, orig_nose]):
            return adjusted  # Return unchanged if essential keypoints are missing
        
        # 階層構造で変換を適用：hip_center -> neck -> shoulders -> arms, legs
        # 全て元の座標を基準に計算し、依存関係を考慮
        
        # 1. 骨盤の調整（元の座標基準）
        orig_hip_center = [(orig_r_hip[0] + orig_l_hip[0]) / 2, (orig_r_hip[1] + orig_l_hip[1]) / 2]
        
        if pelvis_scale != 1.0:
            new_r_hip = [
                orig_hip_center[0] + (orig_r_hip[0] - orig_hip_center[0]) * pelvis_scale,
                orig_hip_center[1] + (orig_r_hip[1] - orig_hip_center[1]) * pelvis_scale
            ]
            new_l_hip = [
                orig_hip_center[0] + (orig_l_hip[0] - orig_hip_center[0]) * pelvis_scale,
                orig_hip_center[1] + (orig_l_hip[1] - orig_hip_center[1]) * pelvis_scale
            ]
            set_point(adjusted, KP["RHip"], new_r_hip)
            set_point(adjusted, KP["LHip"], new_l_hip)
            # 新しいhip_centerを計算
            current_hip_center = [(new_r_hip[0] + new_l_hip[0]) / 2, (new_r_hip[1] + new_l_hip[1]) / 2]
        else:
            current_hip_center = orig_hip_center
        
        # 2. Torso adjustment (from original neck, based on hip_center after pelvis adjustment)
        if torso_scale != 1.0:
            new_neck = [
                current_hip_center[0] + (orig_neck[0] - orig_hip_center[0]) * torso_scale,
                current_hip_center[1] + (orig_neck[1] - orig_hip_center[1]) * torso_scale
            ]
            set_point(adjusted, KP["Neck"], new_neck)
            current_neck = new_neck
        else:
            # pelvis調整による影響のみ適用
            neck_offset = [current_hip_center[0] - orig_hip_center[0], current_hip_center[1] - orig_hip_center[1]]
            current_neck = [orig_neck[0] + neck_offset[0], orig_neck[1] + neck_offset[1]]
            set_point(adjusted, KP["Neck"], current_neck)
        
        # 3. Shoulder adjustment (from original shoulder, based on neck after torso adjustment)
        orig_r_shoulder = get_point(original, KP["RShoulder"])
        orig_l_shoulder = get_point(original, KP["LShoulder"])
        
        if orig_r_shoulder and orig_l_shoulder and shoulder_scale != 1.0:
            new_r_shoulder = [
                current_neck[0] + (orig_r_shoulder[0] - orig_neck[0]) * shoulder_scale,
                current_neck[1] + (orig_r_shoulder[1] - orig_neck[1]) * shoulder_scale
            ]
            new_l_shoulder = [
                current_neck[0] + (orig_l_shoulder[0] - orig_neck[0]) * shoulder_scale,
                current_neck[1] + (orig_l_shoulder[1] - orig_neck[1]) * shoulder_scale
            ]
            set_point(adjusted, KP["RShoulder"], new_r_shoulder)
            set_point(adjusted, KP["LShoulder"], new_l_shoulder)
            current_r_shoulder, current_l_shoulder = new_r_shoulder, new_l_shoulder
        else:
            # torso/pelvisによる影響のみ適用
            neck_movement = [current_neck[0] - orig_neck[0], current_neck[1] - orig_neck[1]]
            current_r_shoulder = [orig_r_shoulder[0] + neck_movement[0], orig_r_shoulder[1] + neck_movement[1]] if orig_r_shoulder else None
            current_l_shoulder = [orig_l_shoulder[0] + neck_movement[0], orig_l_shoulder[1] + neck_movement[1]] if orig_l_shoulder else None
            if current_r_shoulder: set_point(adjusted, KP["RShoulder"], current_r_shoulder)
            if current_l_shoulder: set_point(adjusted, KP["LShoulder"], current_l_shoulder)
        
        # 4. Neck/head adjustment (from original nose/head parts, based on neck after torso adjustment)
        if neck_scale != 1.0 or head_scale != 1.0:
            # neck_scaleでnoseの位置を調整
            if neck_scale != 1.0:
                new_nose = [
                    current_neck[0] + (orig_nose[0] - orig_neck[0]) * neck_scale,
                    current_neck[1] + (orig_nose[1] - orig_neck[1]) * neck_scale
                ]
            else:
                # neck移動の影響のみ適用
                neck_movement = [current_neck[0] - orig_neck[0], current_neck[1] - orig_neck[1]]
                new_nose = [orig_nose[0] + neck_movement[0], orig_nose[1] + neck_movement[1]]
            
            set_point(adjusted, KP["Nose"], new_nose)
            
            # 頭部の他のパーツも調整
            for head_kp in ["REye", "LEye", "REar", "LEar"]:
                if head_kp in KP:
                    orig_head_point = get_point(original, KP[head_kp])
                    if orig_head_point:
                        # neck_scaleとhead_scaleを組み合わせて適用
                        # まずneck_scaleを適用（noseと同様に）
                        if neck_scale != 1.0:
                            neck_scaled = [
                                current_neck[0] + (orig_head_point[0] - orig_neck[0]) * neck_scale,
                                current_neck[1] + (orig_head_point[1] - orig_neck[1]) * neck_scale
                            ]
                        else:
                            # neck移動の影響のみ
                            neck_movement = [current_neck[0] - orig_neck[0], current_neck[1] - orig_neck[1]]
                            neck_scaled = [orig_head_point[0] + neck_movement[0], orig_head_point[1] + neck_movement[1]]
                        
                        # 次にhead_scaleを適用（new_noseを基準に）
                        if head_scale != 1.0:
                            head_scaled = [
                                new_nose[0] + (neck_scaled[0] - new_nose[0]) * head_scale,
                                new_nose[1] + (neck_scaled[1] - new_nose[1]) * head_scale
                            ]
                            set_point(adjusted, KP[head_kp], head_scaled)
                        else:
                            set_point(adjusted, KP[head_kp], neck_scaled)
        else:
            # neck_scale=1.0 and head_scale=1.0の場合、neck移動の影響のみ適用
            neck_movement = [current_neck[0] - orig_neck[0], current_neck[1] - orig_neck[1]]
            new_nose = [orig_nose[0] + neck_movement[0], orig_nose[1] + neck_movement[1]]
            set_point(adjusted, KP["Nose"], new_nose)
            
            for head_kp in ["REye", "LEye", "REar", "LEar"]:
                if head_kp in KP:
                    orig_head_point = get_point(original, KP[head_kp])
                    if orig_head_point:
                        head_moved = [orig_head_point[0] + neck_movement[0], orig_head_point[1] + neck_movement[1]]
                        set_point(adjusted, KP[head_kp], head_moved)
        
        # 5. Arm adjustment (from original arm joints, based on position after shoulder adjustment)
        if arm_scale != 1.0 and current_r_shoulder and current_l_shoulder:
            for side_prefix, current_shoulder, orig_shoulder in [
                ("R", current_r_shoulder, orig_r_shoulder), 
                ("L", current_l_shoulder, orig_l_shoulder)
            ]:
                for joint in ["Elbow", "Wrist"]:
                    kp_name = f"{side_prefix}{joint}"
                    if kp_name in KP:
                        orig_joint = get_point(original, KP[kp_name])
                        if orig_joint and orig_shoulder:
                            new_joint = [
                                current_shoulder[0] + (orig_joint[0] - orig_shoulder[0]) * arm_scale,
                                current_shoulder[1] + (orig_joint[1] - orig_shoulder[1]) * arm_scale
                            ]
                            set_point(adjusted, KP[kp_name], new_joint)
        elif current_r_shoulder and current_l_shoulder:
            # arm_scale=1.0の場合、shoulder移動の影響のみ適用
            for side_prefix, current_shoulder, orig_shoulder in [
                ("R", current_r_shoulder, orig_r_shoulder), 
                ("L", current_l_shoulder, orig_l_shoulder)
            ]:
                if orig_shoulder:
                    shoulder_movement = [current_shoulder[0] - orig_shoulder[0], current_shoulder[1] - orig_shoulder[1]]
                    for joint in ["Elbow", "Wrist"]:
                        kp_name = f"{side_prefix}{joint}"
                        if kp_name in KP:
                            orig_joint = get_point(original, KP[kp_name])
                            if orig_joint:
                                new_joint = [orig_joint[0] + shoulder_movement[0], orig_joint[1] + shoulder_movement[1]]
                                set_point(adjusted, KP[kp_name], new_joint)
        
        # 6. Leg adjustment (from original leg joints, based on position after pelvis adjustment)
        current_r_hip = get_point(adjusted, KP["RHip"])
        current_l_hip = get_point(adjusted, KP["LHip"])
        
        if leg_scale != 1.0:
            for side_prefix, current_hip, orig_hip in [("R", current_r_hip, orig_r_hip), ("L", current_l_hip, orig_l_hip)]:
                for joint in ["Knee", "Ankle"]:
                    kp_name = f"{side_prefix}{joint}"
                    if kp_name in KP:
                        orig_joint = get_point(original, KP[kp_name])
                        if orig_joint and orig_hip:
                            new_joint = [
                                current_hip[0] + (orig_joint[0] - orig_hip[0]) * leg_scale,
                                current_hip[1] + (orig_joint[1] - orig_hip[1]) * leg_scale
                            ]
                            set_point(adjusted, KP[kp_name], new_joint)
        else:
            # leg_scale=1.0の場合、pelvis移動の影響のみ適用
            for side_prefix, current_hip, orig_hip in [("R", current_r_hip, orig_r_hip), ("L", current_l_hip, orig_l_hip)]:
                if orig_hip:
                    hip_movement = [current_hip[0] - orig_hip[0], current_hip[1] - orig_hip[1]]
                    for joint in ["Knee", "Ankle"]:
                        kp_name = f"{side_prefix}{joint}"
                        if kp_name in KP:
                            orig_joint = get_point(original, KP[kp_name])
                            if orig_joint:
                                new_joint = [orig_joint[0] + hip_movement[0], orig_joint[1] + hip_movement[1]]
                                set_point(adjusted, KP[kp_name], new_joint)
        
        # 7. Toe/heel adjustment (ankle-based leg_scale scaling)
        current_l_ankle = get_point(adjusted, KP["LAnkle"])
        current_r_ankle = get_point(adjusted, KP["RAnkle"])
        orig_l_ankle = get_point(original, KP["LAnkle"]) 
        orig_r_ankle = get_point(original, KP["RAnkle"])
        
        if len(adjusted) >= 60:  # Check if toe/heel keypoints exist
            # Toe/heel keypoint definitions (based on actual data structure)
            # 18: RBigToe (right foot toe), 19: LBigToe (left foot toe)
            TOE_HEEL_MAPPING = {
                18: ("RBigToe", current_r_ankle, orig_r_ankle),  # Right toe -> right ankle reference
                19: ("LBigToe", current_l_ankle, orig_l_ankle),  # Left toe -> left ankle reference
            }
            
            for idx, (name, current_ankle, orig_ankle) in TOE_HEEL_MAPPING.items():
                keypoint_start = idx * 3
                if (keypoint_start + 2 < len(original) and original[keypoint_start + 2] > 0 and 
                    current_ankle and orig_ankle):  # Only if both ankle and toe exist
                    
                    orig_toe_heel = [original[keypoint_start], original[keypoint_start + 1]]
                    confidence = original[keypoint_start + 2]
                    
                    if leg_scale != 1.0:
                        # Maintain direction vector from ankle to toe, scale distance only
                        orig_relative = [orig_toe_heel[0] - orig_ankle[0], orig_toe_heel[1] - orig_ankle[1]]
                        scaled_relative = [orig_relative[0] * leg_scale, orig_relative[1] * leg_scale]
                        new_toe_heel = [current_ankle[0] + scaled_relative[0], current_ankle[1] + scaled_relative[1]]
                    else:
                        # leg_scale=1.0: follow ankle movement
                        ankle_movement = [current_ankle[0] - orig_ankle[0], current_ankle[1] - orig_ankle[1]]
                        new_toe_heel = [orig_toe_heel[0] + ankle_movement[0], orig_toe_heel[1] + ankle_movement[1]]
                    
                    # Update toe/heel position
                    adjusted[keypoint_start] = new_toe_heel[0]
                    adjusted[keypoint_start + 1] = new_toe_heel[1]
                    # Keep confidence unchanged
        
        return adjusted
    
    def _adjust_face_keypoints(self, keypoints, canvas_width, canvas_height, **params):
        """
        Face keypointsの調整処理
        顔の各パーツ（目、眉、口、鼻）の調整
        """
        if not keypoints or len(keypoints) < 210:  # 70 face keypoints * 3 = 210
            return keypoints
            
        # Get parameters
        eye_distance_scale = params.get("eye_distance_scale", 1.0)
        eye_height = params.get("eye_height", 0.0)
        eyebrow_height = params.get("eyebrow_height", 0.0)
        left_eye_scale = params.get("left_eye_scale", 1.0)
        right_eye_scale = params.get("right_eye_scale", 1.0)
        left_eyebrow_scale = params.get("left_eyebrow_scale", 1.0)
        right_eyebrow_scale = params.get("right_eyebrow_scale", 1.0)
        mouth_scale = params.get("mouth_scale", 1.0)
        nose_scale_face = params.get("nose_scale_face", 1.0)
        face_shape_scale = params.get("face_shape_scale", 1.0)
        
        # Do nothing if all are default values
        if (eye_distance_scale == 1.0 and eye_height == 0.0 and eyebrow_height == 0.0 and
            left_eye_scale == 1.0 and right_eye_scale == 1.0 and 
            left_eyebrow_scale == 1.0 and right_eyebrow_scale == 1.0 and
            mouth_scale == 1.0 and nose_scale_face == 1.0 and face_shape_scale == 1.0):
            return keypoints
            
        adjusted = keypoints.copy()
        
        # OpenPose face keypoint groups (standard definition)
        FACE_KP_GROUPS = {
            "Left_Eye": list(range(42, 48)),        # 左目 (42-47)
            "Right_Eye": list(range(36, 42)),       # 右目 (36-41)
            "Left_Eyebrow": list(range(22, 27)),    # 左眉 (22-26)
            "Right_Eyebrow": list(range(17, 22)),   # 右眉 (17-21)
            "Mouth": list(range(48, 68)),           # 口 (48-67)
            "Nose_Face": list(range(27, 36)),       # 鼻 (27-35)
            "Face_Shape": list(range(0, 17))        # 顔の輪郭 (0-16)
        }
        
        GROUP_SCALES = {
            "Left_Eye": left_eye_scale,
            "Right_Eye": right_eye_scale,
            "Left_Eyebrow": left_eyebrow_scale,
            "Right_Eyebrow": right_eyebrow_scale,
            "Mouth": mouth_scale,
            "Nose_Face": nose_scale_face,
            "Face_Shape": face_shape_scale
        }
        
        # 顔の中心を計算 (鼻の先端付近を使用)
        if len(adjusted) >= 93:  # nose tip around index 30 * 3 = 90
            face_center_x = adjusted[90]   # nose tip x
            face_center_y = adjusted[91]   # nose tip y
        else:
            face_center_x = canvas_width / 2
            face_center_y = canvas_height / 2
        
        # 各顔パーツのスケーリング
        for group_name, indices in FACE_KP_GROUPS.items():
            scale_factor = GROUP_SCALES.get(group_name, 1.0)
            if scale_factor != 1.0:
                # Eye groups: use their own face keypoint center for scaling
                if group_name in ["Left_Eye", "Right_Eye"]:
                    # Calculate current face eye center (centroid of eye keypoints)
                    valid_points = []
                    for idx in indices:
                        if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                            valid_points.append((adjusted[idx * 3], adjusted[idx * 3 + 1]))
                    
                    if valid_points:
                        # Eye center as scaling origin
                        eye_center_x = sum(p[0] for p in valid_points) / len(valid_points)
                        eye_center_y = sum(p[1] for p in valid_points) / len(valid_points)
                        
                        # Scale around the eye's own center
                        for idx in indices:
                            if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                                x, y = adjusted[idx * 3], adjusted[idx * 3 + 1]
                                
                                # Scale from eye center
                                rel_x = (x - eye_center_x) * scale_factor
                                rel_y = (y - eye_center_y) * scale_factor
                                
                                # Keep centered at eye center
                                adjusted[idx * 3] = eye_center_x + rel_x
                                adjusted[idx * 3 + 1] = eye_center_y + rel_y
                else:
                    # Other face parts use nose tip as center
                    for idx in indices:
                        if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                            x, y = adjusted[idx * 3], adjusted[idx * 3 + 1]
                            
                            # Scale from face center
                            rel_x = (x - face_center_x) * scale_factor
                            rel_y = (y - face_center_y) * scale_factor
                            
                            adjusted[idx * 3] = face_center_x + rel_x
                            adjusted[idx * 3 + 1] = face_center_y + rel_y
        
        # eye_height調整（目の位置を上下に移動）
        if eye_height != 0.0:
            for group_name in ["Left_Eye", "Right_Eye"]:
                for idx in FACE_KP_GROUPS[group_name]:
                    if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                        adjusted[idx * 3 + 1] += eye_height
        
        # eyebrow_height調整（眉の位置を上下に移動）
        if eyebrow_height != 0.0:
            for group_name in ["Left_Eyebrow", "Right_Eyebrow"]:
                for idx in FACE_KP_GROUPS[group_name]:
                    if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                        adjusted[idx * 3 + 1] += eyebrow_height
        
        # eye_distance_scale調整（目の間隔を調整）
        if eye_distance_scale != 1.0:
            # 左右の目の中心を計算
            left_eye_center_x = left_eye_center_y = 0
            right_eye_center_x = right_eye_center_y = 0
            left_count = right_count = 0
            
            for idx in FACE_KP_GROUPS["Left_Eye"]:
                if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                    left_eye_center_x += adjusted[idx * 3]
                    left_eye_center_y += adjusted[idx * 3 + 1]
                    left_count += 1
            
            for idx in FACE_KP_GROUPS["Right_Eye"]:
                if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                    right_eye_center_x += adjusted[idx * 3]
                    right_eye_center_y += adjusted[idx * 3 + 1]
                    right_count += 1
            
            if left_count > 0 and right_count > 0:
                left_eye_center_x /= left_count
                left_eye_center_y /= left_count
                right_eye_center_x /= right_count
                right_eye_center_y /= right_count
                
                # 目の間の中点
                eyes_center_x = (left_eye_center_x + right_eye_center_x) / 2
                eyes_center_y = (left_eye_center_y + right_eye_center_y) / 2
                
                # Scale each eye away from center point
                for group_name, center_x, center_y in [
                    ("Left_Eye", left_eye_center_x, left_eye_center_y),
                    ("Right_Eye", right_eye_center_x, right_eye_center_y)
                ]:
                    for idx in FACE_KP_GROUPS[group_name]:
                        if idx * 3 + 2 < len(adjusted) and adjusted[idx * 3 + 2] > 0:
                            # Scale direction from eye center to overall center
                            rel_x = (center_x - eyes_center_x) * eye_distance_scale
                            rel_y = (center_y - eyes_center_y) * eye_distance_scale
                            
                            # 個々のキーポイントも同じ方向に移動
                            point_rel_x = adjusted[idx * 3] - center_x
                            point_rel_y = adjusted[idx * 3 + 1] - center_y
                            
                            adjusted[idx * 3] = eyes_center_x + rel_x + point_rel_x
                            adjusted[idx * 3 + 1] = eyes_center_y + rel_y + point_rel_y
        
        return adjusted
    
    def _adjust_hand_keypoints(self, keypoints, canvas_width, canvas_height, wrist_movement=None, **params):
        """
        Hand keypointsの調整処理
        手首を基準としたスケーリング + 腕の動きに連動した位置調整
        
        Args:
            wrist_movement: [dx, dy] 手首の移動量（腕のスケーリングによる）
        """
        if not keypoints or len(keypoints) < 63:  # 21 keypoints * 3 = 63
            return keypoints
            
        hands_scale = params.get("hands_scale", 1.0)
        
        # Set default value if wrist_movement is None
        if wrist_movement is None:
            wrist_movement = [0.0, 0.0]
            
        # Do nothing if hands_scale is 1.0 and wrist_movement is 0
        if hands_scale == 1.0 and wrist_movement[0] == 0.0 and wrist_movement[1] == 0.0:
            return keypoints
            
        adjusted = keypoints.copy()
        
        # 手首は index 0 (OpenPose hand keypoint standard)
        wrist_x = adjusted[0]
        wrist_y = adjusted[1]
        wrist_conf = adjusted[2]
        
        if wrist_conf <= 0:
            # Move entire hand even when wrist is not detected
            if wrist_movement[0] != 0.0 or wrist_movement[1] != 0.0:
                adjusted_len = len(adjusted)
                for i in range(0, adjusted_len, 3):
                    if i + 2 < adjusted_len and adjusted[i + 2] > 0:
                        adjusted[i] += wrist_movement[0]
                        adjusted[i + 1] += wrist_movement[1]
            return adjusted
        
        # Step 1: 手首を基準として hands_scale でスケーリング
        if hands_scale != 1.0:
            adjusted_len = len(adjusted)
            for i in range(3, adjusted_len, 3):  # Start from joints other than wrist
                if i + 2 < adjusted_len and adjusted[i + 2] > 0:
                    x, y = adjusted[i], adjusted[i + 1]
                    
                    # Scale by relative position from wrist
                    scaled_x = wrist_x + (x - wrist_x) * hands_scale
                    scaled_y = wrist_y + (y - wrist_y) * hands_scale
                    
                    adjusted[i] = scaled_x
                    adjusted[i + 1] = scaled_y
        
        # Step 2: 腕の動きに連動して手全体を移動
        if wrist_movement[0] != 0.0 or wrist_movement[1] != 0.0:
            for i in range(0, len(adjusted), 3):  # 手首も含めて全てのキーポイントを移動
                if i + 2 < len(adjusted) and adjusted[i + 2] > 0:  # 信頼度チェック
                    adjusted[i] += wrist_movement[0]
                    adjusted[i + 1] += wrist_movement[1]
        
        return adjusted
    
    def _apply_global_transform(self, person, canvas_width, canvas_height, 
                               overall_scale, rotate_angle, translate_x, translate_y):
        """
        全体変換（回転・移動）を適用
        """
        if overall_scale == 1.0 and rotate_angle == 0.0 and translate_x == 0.0 and translate_y == 0.0:
            return  # 変換不要
        
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        
        # 回転角度をラジアンに変換
        angle_rad = math.radians(rotate_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 各キーポイント配列に変換を適用
        for keypoint_type in ["pose_keypoints_2d", "face_keypoints_2d", 
                              "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            if keypoint_type in person and person[keypoint_type]:
                keypoints = person[keypoint_type]
                
                for i in range(0, len(keypoints), 3):
                    if i + 2 < len(keypoints) and keypoints[i + 2] > 0:
                        x, y = keypoints[i], keypoints[i + 1]
                        
                        # 中心を基準とした座標に変換
                        rel_x = x - center_x
                        rel_y = y - center_y
                        
                        # overall_scaleを適用
                        if overall_scale != 1.0:
                            rel_x *= overall_scale
                            rel_y *= overall_scale
                        
                        # 回転変換
                        if rotate_angle != 0.0:
                            new_x = rel_x * cos_a - rel_y * sin_a
                            new_y = rel_x * sin_a + rel_y * cos_a
                        else:
                            new_x = rel_x
                            new_y = rel_y
                        
                        # 移動変換と中心座標の復元
                        keypoints[i] = center_x + new_x + translate_x
                        keypoints[i + 1] = center_y + new_y + translate_y


# ノードクラスのエイリアス（ComfyUIでの参照用）
NODE_CLASS_MAPPINGS = {
    "ProportionChangerParams": ProportionChangerParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProportionChangerParams": "ProportionChanger Params"
}
