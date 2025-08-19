"""
ProportionChanger Reference node
Contains the complex proportion changing algorithm for pose transformation
"""

import numpy as np

# Import utilities from our utils package
from ..utils import (
    pose_keypoint_to_dwpose_format,
    dwpose_format_to_pose_keypoint
)


class ProportionChangerReference:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Target pose keypoints"}),
            },
            "optional": {
                "reference_pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Reference pose keypoint"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("changed_pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ProportionChanger"

    def process(self, pose_keypoint, reference_pose_keypoint=None):
        """
        Process POSE_KEYPOINT data using proportion changing algorithms
        Supports batch processing for multiple frames
        """
        
        if not pose_keypoint or len(pose_keypoint) == 0:
            # Return empty keypoint data
            empty_person = {
                "pose_keypoints_2d": [0.0] * 75,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63
            }
            return ([{"people": [empty_person], "canvas_width": 512, "canvas_height": 768}],)
        
        # Get canvas dimensions from first frame
        frame_data = pose_keypoint[0]
        canvas_width = frame_data.get('canvas_width', 512)
        canvas_height = frame_data.get('canvas_height', 768)
        
        # Prepare reference data if provided
        ref_data = None
        ref_canvas_width, ref_canvas_height = canvas_width, canvas_height
        if reference_pose_keypoint is not None:
            # Get reference canvas dimensions
            ref_frame_data = reference_pose_keypoint[0]
            ref_canvas_width = ref_frame_data.get('canvas_width', 512)
            ref_canvas_height = ref_frame_data.get('canvas_height', 768)
            # Convert reference using its own canvas dimensions
            ref_data = pose_keypoint_to_dwpose_format(reference_pose_keypoint, ref_canvas_width, ref_canvas_height)
        
        # Convert all frames to DWPose format for batch processing
        batch_pose_data = []
        for frame in pose_keypoint:
            single_frame_data = [frame]
            pose_data = pose_keypoint_to_dwpose_format(single_frame_data, canvas_width, canvas_height)
            batch_pose_data.append(pose_data)
        
        # Apply WanVideo-style batch proportion changes with unified ratios
        processed_batch = self.apply_batch_proportion_changes(
            batch_pose_data, ref_data, 
            canvas_width, canvas_height, ref_canvas_width, ref_canvas_height
        )
        
        # Convert back to POSE_KEYPOINT format
        result_frames = []
        for processed_pose in processed_batch:
            result_keypoint = dwpose_format_to_pose_keypoint(
                processed_pose['bodies']['candidate'],
                processed_pose['faces'],
                processed_pose['hands'],
                canvas_width,
                canvas_height
            )
            
            # Add the processed frame to results
            if result_keypoint and isinstance(result_keypoint, dict):
                result_frames.append(result_keypoint)
            else:
                # Fallback empty frame
                empty_person = {
                    "pose_keypoints_2d": [0.0] * 75,
                    "face_keypoints_2d": [0.0] * 210,
                    "hand_left_keypoints_2d": [0.0] * 63,
                    "hand_right_keypoints_2d": [0.0] * 63
                }
                result_frames.append({"people": [empty_person], "canvas_width": canvas_width, "canvas_height": canvas_height})
        
        return (result_frames,)
    
    def apply_batch_proportion_changes(self, batch_pose_data, ref_data,
                                     canvas_width, canvas_height, ref_canvas_width, ref_canvas_height):
        """
        COMPLETE WanVideo pose_extract port (lines 203-700)
        Direct 1:1 implementation of WanVideo logic with all corrections and noise reduction
        """
        if not batch_pose_data or not ref_data:
            return batch_pose_data
        
        # Simulate results_vis structure like WanVideo
        results_vis = batch_pose_data
        
        # Get reference data (canvas scaling is handled by coordinate conversion functions)
        ref_candidate = ref_data['bodies']['candidate'].copy()
        ref_faces = ref_data['faces'].copy() 
        ref_hands = ref_data['hands'].copy()
        
        # NOTE: Canvas size scaling is NOT needed here because:
        # 1. pose_keypoint_to_dwpose_format already converts to normalized coordinates (0-1)
        # 2. Both target and reference are in the same normalized coordinate space
        # 3. dwpose_format_to_pose_keypoint handles final canvas scaling automatically
        # Adding canvas scaling here would result in double scaling!
        
        # Get first frame data
        bodies = results_vis[0]['bodies']
        faces = results_vis[0]['faces']
        hands = results_vis[0]['hands']
        candidate = bodies['candidate']
        
        # EXACT WanVideo implementation starts here (lines 241-500)
        ref_2_x = ref_candidate[2][0]
        ref_2_y = ref_candidate[2][1]
        ref_5_x = ref_candidate[5][0]
        ref_5_y = ref_candidate[5][1]
        ref_8_x = ref_candidate[8][0]
        ref_8_y = ref_candidate[8][1]
        ref_11_x = ref_candidate[11][0]
        ref_11_y = ref_candidate[11][1]
        ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
        ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

        zero_2_x = candidate[2][0]
        zero_2_y = candidate[2][1]
        zero_5_x = candidate[5][0]
        zero_5_y = candidate[5][1]
        zero_8_x = candidate[8][0]
        zero_8_y = candidate[8][1]
        zero_11_x = candidate[11][0]
        zero_11_y = candidate[11][1]
        zero_center1 = 0.5*(candidate[2]+candidate[5])
        zero_center2 = 0.5*(candidate[8]+candidate[11])

        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x) if (zero_5_x-zero_2_x) != 0 else 1.0
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1]) if (zero_center2[1]-zero_center1[1]) != 0 else 1.0

        results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[0]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[0]['hands'][:,:,0] *= x_ratio
        results_vis[0]['hands'][:,:,1] *= y_ratio
        
        # Store original data for face processing
        original_candidate = candidate.copy()
        original_faces = results_vis[0]['faces'].copy()
        
        # Advanced face scaling with independent X and Y scaling based on reference proportions
        faces = results_vis[0]['faces']
        if len(candidate) >= 16 and len(ref_candidate) >= 16 and len(faces) > 0 and faces.shape[1] > 30:
            # Calculate reference and target proportions
            # Use ACTUAL ears (keypoints 3,4) for proper face scaling
            ref_ear_distance = ((ref_candidate[3][0] - ref_candidate[4][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[4][1]) ** 2) ** 0.5
            # Use ORIGINAL candidate (before body scaling) for face calculations
            target_ear_distance_original = ((original_candidate[3][0] - original_candidate[4][0]) ** 2 + (original_candidate[3][1] - original_candidate[4][1]) ** 2) ** 0.5
            
            if target_ear_distance_original > 0:
                # Calculate original face contour width (before any scaling)
                face_left_idx = 0   # Face contour left
                face_right_idx = 16 # Face contour right
                original_face_width = ((original_faces[0, face_right_idx, 0] - original_faces[0, face_left_idx, 0]) ** 2 + 
                                      (original_faces[0, face_right_idx, 1] - original_faces[0, face_left_idx, 1]) ** 2) ** 0.5
                
                # Reference face contour width
                if ref_faces is not None and len(ref_faces) > 0 and ref_faces.shape[1] > 16:
                    ref_face_width = ((ref_faces[0, face_right_idx, 0] - ref_faces[0, face_left_idx, 0]) ** 2 + 
                                     (ref_faces[0, face_right_idx, 1] - ref_faces[0, face_left_idx, 1]) ** 2) ** 0.5
                    
                    if ref_face_width > 0 and original_face_width > 0:
                        # X scaling: match reference face proportion
                        face_scale_ratio_x = ref_face_width / original_face_width
                        
                        # Y scaling: match reference face height proportion
                        # Use nose tip (30) to chin (8) distance for face height
                        nose_idx = 30
                        chin_idx = 8
                        
                        # Calculate original face height
                        if original_faces.shape[1] > max(nose_idx, chin_idx):
                            original_face_height = ((original_faces[0, nose_idx, 1] - original_faces[0, chin_idx, 1]) ** 2 + 
                                                   (original_faces[0, nose_idx, 0] - original_faces[0, chin_idx, 0]) ** 2) ** 0.5
                            
                            # Calculate reference face height
                            ref_face_height = ((ref_faces[0, nose_idx, 1] - ref_faces[0, chin_idx, 1]) ** 2 + 
                                             (ref_faces[0, nose_idx, 0] - ref_faces[0, chin_idx, 0]) ** 2) ** 0.5
                            
                            if original_face_height > 0 and ref_face_height > 0:
                                face_scale_ratio_y = ref_face_height / original_face_height
                            else:
                                # Fallback to ear distance ratio
                                face_scale_ratio_y = ref_ear_distance / target_ear_distance_original
                        else:
                            # Fallback to ear distance ratio
                            face_scale_ratio_y = ref_ear_distance / target_ear_distance_original
                    else:
                        # Fallback to body ratios
                        face_scale_ratio_x = x_ratio
                        face_scale_ratio_y = y_ratio
                else:
                    # Fallback to body ratios
                    face_scale_ratio_x = x_ratio
                    face_scale_ratio_y = y_ratio
                
                # Use face nose tip (keypoint 30) as reference for alignment
                face_nose_tip_idx = 30
                
                # Get current face nose position (before scaling)
                current_face_nose = original_faces[0, face_nose_tip_idx, :]
                
                # Scale faces relative to current face nose position with different X/Y ratios
                faces_centered = original_faces - current_face_nose[np.newaxis, np.newaxis, :]
                faces_centered[:, :, 0] *= face_scale_ratio_x  # X scaling
                faces_centered[:, :, 1] *= face_scale_ratio_y  # Y scaling
                
                # Apply scaling first (relative to original nose position)
                faces[:, :, :] = faces_centered + current_face_nose[np.newaxis, np.newaxis, :]
                
                # Update candidate reference after body scaling
                candidate = results_vis[0]['bodies']['candidate']
                
                # Align face with body nose position (most important for visual accuracy)
                body_nose_position = candidate[0]  # Body nose after scaling (keypoint 0)
                scaled_face_nose = faces[0, face_nose_tip_idx, :]  # Face nose after scaling
                
                # Primary alignment: nose to nose (both X and Y)
                x_offset_face = body_nose_position[0] - scaled_face_nose[0]
                y_offset_face = body_nose_position[1] - scaled_face_nose[1]
                
                # Apply primary nose alignment
                faces[:, :, 0] += x_offset_face
                faces[:, :, 1] += y_offset_face
                
            else:
                # Fallback to body scaling if ear distance calculation fails
                faces[:,:,0] *= x_ratio
                faces[:,:,1] *= y_ratio
        else:
            # Fallback to body scaling if ear keypoints or face data are not available
            faces[:,:,0] *= x_ratio
            faces[:,:,1] *= y_ratio
        
        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']
        
        ########neck########
        l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
        neck_ratio = l_neck_ref / l_neck_0 if l_neck_0 > 0 else 1.0

        x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
        y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

        results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck
        
        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']
        
        ########shoulder2########
        l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5
        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0 if l_shoulder2_0 > 0 else 1.0

        x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

        results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[0]['hands'][1,:,1] += y_offset_shoulder2

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########shoulder5########
        l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5
        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0 if l_shoulder5_0 > 0 else 1.0

        x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

        results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

        # Continue with ALL 15 sections - COMPLETE WanVideo implementation
        
        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########arm3########
        l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
        l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5
        arm3_ratio = l_arm3_ref / l_arm3_0 if l_arm3_0 > 0 else 1.0

        x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

        results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[0]['hands'][1,:,0] += x_offset_arm3
        results_vis[0]['hands'][1,:,1] += y_offset_arm3

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########arm4########
        l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
        l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5
        arm4_ratio = l_arm4_ref / l_arm4_0 if l_arm4_0 > 0 else 1.0

        x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[0]['hands'][1,:,0] += x_offset_arm4
        results_vis[0]['hands'][1,:,1] += y_offset_arm4

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########arm6########
        l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
        l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5
        arm6_ratio = l_arm6_ref / l_arm6_0 if l_arm6_0 > 0 else 1.0

        x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

        results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[0]['hands'][0,:,0] += x_offset_arm6
        results_vis[0]['hands'][0,:,1] += y_offset_arm6

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########arm7########
        l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
        l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5
        arm7_ratio = l_arm7_ref / l_arm7_0 if l_arm7_0 > 0 else 1.0

        x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[0]['hands'][0,:,0] += x_offset_arm7
        results_vis[0]['hands'][0,:,1] += y_offset_arm7

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########head14########
        l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5
        head14_ratio = l_head14_ref / l_head14_0 if l_head14_0 > 0 else 1.0

        x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
        y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

        results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########head15########
        l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5
        head15_ratio = l_head15_ref / l_head15_0 if l_head15_0 > 0 else 1.0

        x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
        y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

        results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########head16########
        l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
        l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5
        head16_ratio = l_head16_ref / l_head16_0 if l_head16_0 > 0 else 1.0

        x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
        y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        ########head17########
        l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
        l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5
        head17_ratio = l_head17_ref / l_head17_0 if l_head17_0 > 0 else 1.0

        x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
        y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']
        
        ########left leg########
        l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
        l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
        ll1_ratio = l_ll1_ref / l_ll1_0 if l_ll1_0 > 0 else 1.0

        x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

        results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
        if len(candidate) > 19:
            results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
            results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
        l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
        ll2_ratio = l_ll2_ref / l_ll2_0 if l_ll2_0 > 0 else 1.0

        x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
        if len(candidate) > 19:
            results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
            results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2

        # Update candidate reference  
        candidate = results_vis[0]['bodies']['candidate']

        ########right leg########
        l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
        l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
        rl1_ratio = l_rl1_ref / l_rl1_0 if l_rl1_0 > 0 else 1.0

        x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

        results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
        if len(candidate) > 18:
            results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
            results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1

        # Update candidate reference
        candidate = results_vis[0]['bodies']['candidate']

        l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
        l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
        rl2_ratio = l_rl2_ref / l_rl2_0 if l_rl2_0 > 0 else 1.0

        x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
        if len(candidate) > 18:
            results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
            results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2
        
        # IMPORTANT: Final position offset for first frame
        candidate = results_vis[0]['bodies']['candidate']
        offset = ref_candidate[1] - candidate[1]
        results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]

        # Now apply to all other frames (WanVideo lines 502-676)
        for i in range(1, len(results_vis)):
            results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
            results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
            results_vis[i]['hands'][:,:,0] *= x_ratio
            results_vis[i]['hands'][:,:,1] *= y_ratio
            
            # Apply same face processing to other frames
            frame_faces = results_vis[i]['faces']
            if len(frame_faces) > 0 and frame_faces.shape[1] > 30:
                # Apply the same face scaling ratios calculated from first frame
                if 'face_scale_ratio_x' in locals() and 'face_scale_ratio_y' in locals():
                    # Store original face data for this frame
                    original_frame_faces = frame_faces.copy()
                    
                    # Use face nose tip (keypoint 30) as reference for alignment
                    face_nose_tip_idx = 30
                    current_face_nose = original_frame_faces[0, face_nose_tip_idx, :]
                    
                    # Scale faces relative to current face nose position
                    faces_centered = original_frame_faces - current_face_nose[np.newaxis, np.newaxis, :]
                    faces_centered[:, :, 0] *= face_scale_ratio_x  # X scaling
                    faces_centered[:, :, 1] *= face_scale_ratio_y  # Y scaling
                    
                    # Apply scaling
                    frame_faces[:, :, :] = faces_centered + current_face_nose[np.newaxis, np.newaxis, :]
                    
                    # Align with body after body adjustments are complete
                    # This will be done after all body adjustments below
                else:
                    # Fallback to body scaling
                    frame_faces[:,:,0] *= x_ratio
                    frame_faces[:,:,1] *= y_ratio
            else:
                # Fallback to body scaling
                if len(frame_faces) > 0:
                    frame_faces[:,:,0] *= x_ratio
                    frame_faces[:,:,1] *= y_ratio

            ########neck########
            x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
            y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

            results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

            ########shoulder2########
            x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

            results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
            results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
            results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

            ########shoulder5########
            x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

            results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
            results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
            results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

            ########arm3########
            x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

            results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
            results_vis[i]['hands'][1,:,0] += x_offset_arm3
            results_vis[i]['hands'][1,:,1] += y_offset_arm3

            ########arm4########
            x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
            results_vis[i]['hands'][1,:,0] += x_offset_arm4
            results_vis[i]['hands'][1,:,1] += y_offset_arm4

            ########arm6########
            x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

            results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
            results_vis[i]['hands'][0,:,0] += x_offset_arm6
            results_vis[i]['hands'][0,:,1] += y_offset_arm6

            ########arm7########
            x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
            results_vis[i]['hands'][0,:,0] += x_offset_arm7
            results_vis[i]['hands'][0,:,1] += y_offset_arm7

            ########head14########
            x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
            y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

            results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

            ########head15########
            x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
            y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

            results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

            ########head16########
            x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
            y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

            ########head17########
            x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
            y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

            ########left leg########
            x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

            results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
            if len(results_vis[i]['bodies']['candidate']) > 19:
                results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
                results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1

            x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
            if len(results_vis[i]['bodies']['candidate']) > 19:
                results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
                results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

            ########right leg########
            x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

            results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
            if len(results_vis[i]['bodies']['candidate']) > 18:
                results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
                results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1

            x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
            if len(results_vis[i]['bodies']['candidate']) > 18:
                results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
                results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

            # Apply face alignment after all body adjustments
            frame_faces = results_vis[i]['faces']
            if len(frame_faces) > 0 and frame_faces.shape[1] > 30:
                if 'face_scale_ratio_x' in locals() and 'face_scale_ratio_y' in locals():
                    # Align face with body nose position (keypoint 0)
                    body_nose_position = results_vis[i]['bodies']['candidate'][0]  # Body nose after scaling
                    face_nose_tip_idx = 30
                    scaled_face_nose = frame_faces[0, face_nose_tip_idx, :]  # Face nose after scaling
                    
                    # Primary alignment: nose to nose (both X and Y)
                    x_offset_face = body_nose_position[0] - scaled_face_nose[0]
                    y_offset_face = body_nose_position[1] - scaled_face_nose[1]
                    
                    # Apply face alignment offsets
                    frame_faces[:, :, 0] += x_offset_face
                    frame_faces[:, :, 1] += y_offset_face

            # Final position offset for this frame
            results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
            results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
            results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]
    
        return results_vis
    
    def apply_proportion_changes(self, pose_data, ref_data, 
                                canvas_width, canvas_height, ref_canvas_width, ref_canvas_height):
        """
        Apply proportion changing algorithms from the original DWPose detector
        Complete 1:1 port from pose_extract function (lines 241-500+) in WanVideoWrapper
        
        DWPose Body Keypoint Structure:
        0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
        5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
        9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
        13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle,
        17-24: Foot keypoints (refer to CLAUDE.md for detailed mapping)
        """
        
        # Get candidate and subset from pose data
        candidate = pose_data['bodies']['candidate']
        faces = pose_data['faces']
        hands = pose_data['hands']
        
        if len(candidate) == 0:
            return pose_data
        
        # If reference data is provided, apply proportion changes
        if ref_data is not None and len(ref_data['bodies']['candidate']) > 0:
            ref_candidate = ref_data['bodies']['candidate']
            ref_faces = ref_data['faces']
            ref_hands = ref_data['hands']
            
            # NOTE: Canvas size scaling is handled automatically by the coordinate conversion functions
            # No manual canvas scaling needed in proportion processing
            
            # Complete algorithm port from original lines 241-500+
            # Note: ref_candidate, ref_hands, ref_faces are already canvas-size-scaled if needed
            ref_2_x = ref_candidate[2][0]
            ref_2_y = ref_candidate[2][1]
            ref_5_x = ref_candidate[5][0]
            ref_5_y = ref_candidate[5][1]
            ref_8_x = ref_candidate[8][0]
            ref_8_y = ref_candidate[8][1]
            ref_11_x = ref_candidate[11][0]
            ref_11_y = ref_candidate[11][1]
            ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
            ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

            zero_2_x = candidate[2][0]
            zero_2_y = candidate[2][1]
            zero_5_x = candidate[5][0]
            zero_5_y = candidate[5][1]
            zero_8_x = candidate[8][0]
            zero_8_y = candidate[8][1]
            zero_11_x = candidate[11][0]
            zero_11_y = candidate[11][1]
            zero_center1 = 0.5*(candidate[2]+candidate[5])
            zero_center2 = 0.5*(candidate[8]+candidate[11])

            # Calculate proportion ratios WITHOUT canvas size scaling (will be applied uniformly at the end)
            ref_proportion_x = (ref_5_x-ref_2_x)
            ref_proportion_y = (ref_center2[1]-ref_center1[1])
            target_proportion_x = (zero_5_x-zero_2_x)
            target_proportion_y = (zero_center2[1]-zero_center1[1])
            
            x_ratio = ref_proportion_x / target_proportion_x if target_proportion_x != 0 else 1.0
            y_ratio = ref_proportion_y / target_proportion_y if target_proportion_y != 0 else 1.0

            # Store original candidate before scaling for face calculations
            original_candidate = candidate.copy()
            
            candidate[:,0] *= x_ratio
            candidate[:,1] *= y_ratio
            hands[:,:,0] *= x_ratio
            hands[:,:,1] *= y_ratio
            
            # Face scaling with independent X and Y scaling based on reference proportions
            if len(candidate) >= 16 and len(ref_candidate) >= 16 and len(faces) > 0 and faces.shape[1] > 30:
                # Store original face data before any scaling
                original_faces = faces.copy()
                
                # Calculate reference and target proportions
                # Use ACTUAL ears (keypoints 3,4) not knees!
                # Use original reference measurements (no canvas scaling for face calculations)
                ref_ear_distance = ((ref_candidate[3][0] - ref_candidate[4][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[4][1]) ** 2) ** 0.5
                # Use ORIGINAL candidate (before body scaling) for face calculations
                target_ear_distance_original = ((original_candidate[3][0] - original_candidate[4][0]) ** 2 + (original_candidate[3][1] - original_candidate[4][1]) ** 2) ** 0.5
                
                if target_ear_distance_original > 0:
                    # Calculate original face contour width (before any scaling)
                    face_left_idx = 0   # Face contour left
                    face_right_idx = 16 # Face contour right
                    original_face_width = ((original_faces[0, face_right_idx, 0] - original_faces[0, face_left_idx, 0]) ** 2 + 
                                          (original_faces[0, face_right_idx, 1] - original_faces[0, face_left_idx, 1]) ** 2) ** 0.5
                    
                    # Reference face contour width (use original measurements, canvas scaling applied later)
                    ref_faces = ref_data['faces'] if ref_data else None
                    if ref_faces is not None and len(ref_faces) > 0 and ref_faces.shape[1] > 16:
                        # Use original reference face measurements (no canvas scaling here)
                        ref_face_width = ((ref_faces[0, face_right_idx, 0] - ref_faces[0, face_left_idx, 0]) ** 2 + 
                                         (ref_faces[0, face_right_idx, 1] - ref_faces[0, face_left_idx, 1]) ** 2) ** 0.5
                        
                        if ref_face_width > 0 and original_face_width > 0:
                            # X scaling: match reference face proportion (no canvas scaling)
                            face_scale_ratio_x = ref_face_width / original_face_width
                            
                            # Y scaling: match reference face height proportion
                            # Use nose tip (30) to chin (8) distance for face height
                            nose_idx = 30
                            chin_idx = 8
                            
                            # Calculate original face height
                            if original_faces.shape[1] > max(nose_idx, chin_idx):
                                original_face_height = ((original_faces[0, nose_idx, 1] - original_faces[0, chin_idx, 1]) ** 2 + 
                                                       (original_faces[0, nose_idx, 0] - original_faces[0, chin_idx, 0]) ** 2) ** 0.5
                                
                                # Calculate reference face height (use original measurements)
                                ref_face_height = ((ref_faces[0, nose_idx, 1] - ref_faces[0, chin_idx, 1]) ** 2 + 
                                                 (ref_faces[0, nose_idx, 0] - ref_faces[0, chin_idx, 0]) ** 2) ** 0.5
                                
                                if original_face_height > 0 and ref_face_height > 0:
                                    face_scale_ratio_y = ref_face_height / original_face_height
                                else:
                                    # Fallback to ear distance ratio
                                    face_scale_ratio_y = ref_ear_distance / target_ear_distance_original
                            else:
                                # Fallback to ear distance ratio
                                face_scale_ratio_y = ref_ear_distance / target_ear_distance_original
                        else:
                            # Fallback to body ratios
                            face_scale_ratio_x = x_ratio
                            face_scale_ratio_y = y_ratio
                    else:
                        # Fallback to body ratios
                        face_scale_ratio_x = x_ratio
                        face_scale_ratio_y = y_ratio
                    
                    # Use face nose tip (keypoint 30) as reference for alignment
                    face_nose_tip_idx = 30
                    
                    # Get current face nose position (before scaling)
                    current_face_nose = original_faces[0, face_nose_tip_idx, :]
                    
                    # Scale faces relative to current face nose position with different X/Y ratios
                    faces_centered = original_faces - current_face_nose[np.newaxis, np.newaxis, :]
                    faces_centered[:, :, 0] *= face_scale_ratio_x  # X scaling
                    faces_centered[:, :, 1] *= face_scale_ratio_y  # Y scaling
                    
                    # Apply scaling first (relative to original nose position)
                    faces[:, :, :] = faces_centered + current_face_nose[np.newaxis, np.newaxis, :]
                    
                    # Align based on eye positions (most important for visual accuracy)
                    body_nose_position = candidate[0]  # Body nose after scaling
                    scaled_face_nose = faces[0, face_nose_tip_idx, :]  # Face nose after scaling
                    
                    # Calculate X offset to align nose positions
                    x_offset_face = body_nose_position[0] - scaled_face_nose[0]
                    
                    # Calculate Y offset based on eye level alignment (most important!)
                    # Use ACTUAL body eyes (keypoints 1,2) - not ear approximation!
                    body_left_eye = candidate[1]   # Left Eye
                    body_right_eye = candidate[2]  # Right Eye
                    body_eye_center_y = (body_left_eye[1] + body_right_eye[1]) / 2.0
                    
                    # Calculate face eye center Y position (using eye keypoints 36-47 region average)
                    if faces.shape[1] > 47:
                        # Use left eye (36-41) and right eye (42-47) center
                        left_eye_center_y = np.mean(faces[0, 36:42, 1])
                        right_eye_center_y = np.mean(faces[0, 42:48, 1])
                        face_eye_center_y = (left_eye_center_y + right_eye_center_y) / 2.0
                        
                        # Align face eye level with body eye level (EXACT match!)
                        y_offset_face = body_eye_center_y - face_eye_center_y
                    else:
                        # Fallback to nose alignment if eye keypoints not available
                        y_offset_face = body_nose_position[1] - scaled_face_nose[1]
                    
                    # Apply X and Y offsets independently
                    faces[:, :, 0] += x_offset_face
                    faces[:, :, 1] += y_offset_face
                    
                else:
                    # Fallback to body scaling if ear distance calculation fails
                    faces[:,:,0] *= x_ratio
                    faces[:,:,1] *= y_ratio
            else:
                # Fallback to body scaling if ear keypoints or face data are not available
                faces[:,:,0] *= x_ratio
                faces[:,:,1] *= y_ratio
            
            ########neck########
            l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
            l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
            neck_ratio = l_neck_ref / l_neck_0

            x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
            y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

            candidate[0,0] += x_offset_neck
            candidate[0,1] += y_offset_neck
            candidate[14,0] += x_offset_neck
            candidate[14,1] += y_offset_neck
            candidate[15,0] += x_offset_neck
            candidate[15,1] += y_offset_neck
            candidate[16,0] += x_offset_neck
            candidate[16,1] += y_offset_neck
            candidate[17,0] += x_offset_neck
            candidate[17,1] += y_offset_neck
            
            ########shoulder2########
            l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
            l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

            shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

            x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

            candidate[2,0] += x_offset_shoulder2
            candidate[2,1] += y_offset_shoulder2
            candidate[3,0] += x_offset_shoulder2
            candidate[3,1] += y_offset_shoulder2
            candidate[4,0] += x_offset_shoulder2
            candidate[4,1] += y_offset_shoulder2
            hands[1,:,0] += x_offset_shoulder2
            hands[1,:,1] += y_offset_shoulder2

            ########shoulder5########
            l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
            l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

            shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

            x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

            candidate[5,0] += x_offset_shoulder5
            candidate[5,1] += y_offset_shoulder5
            candidate[6,0] += x_offset_shoulder5
            candidate[6,1] += y_offset_shoulder5
            candidate[7,0] += x_offset_shoulder5
            candidate[7,1] += y_offset_shoulder5
            hands[0,:,0] += x_offset_shoulder5
            hands[0,:,1] += y_offset_shoulder5

            ########arm3########
            l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
            l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

            arm3_ratio = l_arm3_ref / l_arm3_0

            x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

            candidate[3,0] += x_offset_arm3
            candidate[3,1] += y_offset_arm3
            candidate[4,0] += x_offset_arm3
            candidate[4,1] += y_offset_arm3
            hands[1,:,0] += x_offset_arm3
            hands[1,:,1] += y_offset_arm3

            ########arm4########
            l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
            l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

            arm4_ratio = l_arm4_ref / l_arm4_0

            x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

            candidate[4,0] += x_offset_arm4
            candidate[4,1] += y_offset_arm4
            hands[1,:,0] += x_offset_arm4
            hands[1,:,1] += y_offset_arm4

            ########arm6########
            l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
            l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

            arm6_ratio = l_arm6_ref / l_arm6_0

            x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

            candidate[6,0] += x_offset_arm6
            candidate[6,1] += y_offset_arm6
            candidate[7,0] += x_offset_arm6
            candidate[7,1] += y_offset_arm6
            hands[0,:,0] += x_offset_arm6
            hands[0,:,1] += y_offset_arm6

            ########arm7########
            l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
            l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

            arm7_ratio = l_arm7_ref / l_arm7_0

            x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

            candidate[7,0] += x_offset_arm7
            candidate[7,1] += y_offset_arm7
            hands[0,:,0] += x_offset_arm7
            hands[0,:,1] += y_offset_arm7

            ########head14########
            l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
            l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

            head14_ratio = l_head14_ref / l_head14_0

            x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
            y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

            candidate[14,0] += x_offset_head14
            candidate[14,1] += y_offset_head14
            candidate[16,0] += x_offset_head14
            candidate[16,1] += y_offset_head14

            ########head15########
            l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
            l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

            head15_ratio = l_head15_ref / l_head15_0

            x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
            y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

            candidate[15,0] += x_offset_head15
            candidate[15,1] += y_offset_head15
            candidate[17,0] += x_offset_head15
            candidate[17,1] += y_offset_head15

            ########head16########
            l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
            l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

            head16_ratio = l_head16_ref / l_head16_0

            x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
            y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

            candidate[16,0] += x_offset_head16
            candidate[16,1] += y_offset_head16

            ########head17########
            l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
            l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

            head17_ratio = l_head17_ref / l_head17_0

            x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
            y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

            candidate[17,0] += x_offset_head17
            candidate[17,1] += y_offset_head17
            
            ########left leg########
            l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
            l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
            ll1_ratio = l_ll1_ref / l_ll1_0

            x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

            candidate[9,0] += x_offset_ll1
            candidate[9,1] += y_offset_ll1
            candidate[10,0] += x_offset_ll1
            candidate[10,1] += y_offset_ll1
            candidate[19,0] += x_offset_ll1
            candidate[19,1] += y_offset_ll1

            l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
            l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
            ll2_ratio = l_ll2_ref / l_ll2_0

            x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

            candidate[10,0] += x_offset_ll2
            candidate[10,1] += y_offset_ll2
            candidate[19,0] += x_offset_ll2
            candidate[19,1] += y_offset_ll2

            ########right leg########
            l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
            l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
            rl1_ratio = l_rl1_ref / l_rl1_0

            x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

            candidate[12,0] += x_offset_rl1
            candidate[12,1] += y_offset_rl1
            candidate[13,0] += x_offset_rl1
            candidate[13,1] += y_offset_rl1
            candidate[18,0] += x_offset_rl1
            candidate[18,1] += y_offset_rl1

            l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
            l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
            rl2_ratio = l_rl2_ref / l_rl2_0

            x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

            candidate[13,0] += x_offset_rl2
            candidate[13,1] += y_offset_rl2
            candidate[18,0] += x_offset_rl2
            candidate[18,1] += y_offset_rl2

            # Final offset to align neck positions (line 496 in original)
            offset = ref_candidate[1] - candidate[1]

            candidate += offset[np.newaxis, :]
            hands += offset[np.newaxis, np.newaxis, :]
            
            # Face offset: maintain nose alignment instead of neck alignment
            if len(faces) > 0 and faces.shape[1] > 30:
                # Calculate offset to keep face nose aligned with body nose
                face_nose_tip_idx = 30
                current_face_nose = faces[0, face_nose_tip_idx, :]
                body_nose_after_offset = candidate[0]  # Body nose after offset
                face_offset = body_nose_after_offset - current_face_nose
                faces += face_offset[np.newaxis, np.newaxis, :]
            else:
                # Fallback: apply same offset as body
                faces += offset[np.newaxis, np.newaxis, :]
        
        # NOTE: Canvas scaling is NOT needed here because:
        # 1. pose_keypoint_to_dwpose_format converts to normalized coordinates (0-1)
        # 2. proportion processing works on normalized coordinates
        # 3. dwpose_format_to_pose_keypoint converts back to target canvas coordinates
        # Adding canvas scaling here would result in double scaling!
        
        return {
            'bodies': {
                'candidate': candidate,
                'subset': pose_data['bodies']['subset']
            },
            'faces': faces,
            'hands': hands
        }