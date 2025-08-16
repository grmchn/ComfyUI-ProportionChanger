"""
Pose data processing and conversion utilities
Handles conversion between POSE_KEYPOINT format and DWPose internal format
"""

import numpy as np
import copy


def pose_keypoint_to_dwpose_format(pose_keypoint, canvas_width, canvas_height):
    """
    Convert POSE_KEYPOINT format to DWPose internal format
    
    Args:
        pose_keypoint: POSE_KEYPOINT data (list of dicts)
        canvas_width: Canvas width for coordinate conversion
        canvas_height: Canvas height for coordinate conversion
    
    Returns:
        dict: DWPose format with 'bodies', 'faces', 'hands' keys
    """
    
    if not pose_keypoint or len(pose_keypoint) == 0:
        return {'bodies': {'candidate': np.array([]), 'subset': np.array([])}, 'faces': np.array([]), 'hands': np.array([])}
    
    frame_data = pose_keypoint[0]  # Use first frame
    people = frame_data.get('people', [])
    
    if len(people) == 0:
        return {'bodies': {'candidate': np.array([]), 'subset': np.array([])}, 'faces': np.array([]), 'hands': np.array([])}
    
    person = people[0]  # Use first person
    
    # Extract keypoints
    body_kpts = person.get('pose_keypoints_2d', [])
    face_kpts = person.get('face_keypoints_2d', [])
    lhand_kpts = person.get('hand_left_keypoints_2d', [])
    rhand_kpts = person.get('hand_right_keypoints_2d', [])
    
    # Convert body keypoints to candidate format
    candidates = []
    subset = []
    
    if body_kpts and len(body_kpts) >= 75:  # 25 points * 3 (x,y,conf)
        for i in range(25):  # Support full 25 points including toes
            x = body_kpts[i*3] / canvas_width if canvas_width > 0 else body_kpts[i*3]
            y = body_kpts[i*3+1] / canvas_height if canvas_height > 0 else body_kpts[i*3+1]
            conf = body_kpts[i*3+2]
            candidates.append([x, y, conf])
        
        # Create subset (which keypoints are valid)
        subset_row = []
        for i in range(25):
            if body_kpts[i*3+2] > 0:  # confidence > 0
                subset_row.append(i)
            else:
                subset_row.append(-1)
        subset.append(subset_row)
    
    candidate_array = np.array(candidates) if candidates else np.array([])
    subset_array = np.array(subset) if subset else np.array([])
    
    # Convert face keypoints
    faces = []
    if face_kpts and len(face_kpts) >= 210:  # 70 points * 3
        face_points = []
        for i in range(70):
            x = face_kpts[i*3] / canvas_width if canvas_width > 0 else face_kpts[i*3]
            y = face_kpts[i*3+1] / canvas_height if canvas_height > 0 else face_kpts[i*3+1]
            conf = face_kpts[i*3+2]
            face_points.append([x, y, conf])
        faces.append(face_points)
    
    # Convert hand keypoints
    hands = []
    if lhand_kpts and len(lhand_kpts) >= 63:  # 21 points * 3
        lhand_points = []
        for i in range(21):
            x = lhand_kpts[i*3] / canvas_width if canvas_width > 0 else lhand_kpts[i*3]
            y = lhand_kpts[i*3+1] / canvas_height if canvas_height > 0 else lhand_kpts[i*3+1]
            conf = lhand_kpts[i*3+2]
            lhand_points.append([x, y, conf])
        hands.append(lhand_points)
    
    if rhand_kpts and len(rhand_kpts) >= 63:  # 21 points * 3
        rhand_points = []
        for i in range(21):
            x = rhand_kpts[i*3] / canvas_width if canvas_width > 0 else rhand_kpts[i*3]
            y = rhand_kpts[i*3+1] / canvas_height if canvas_height > 0 else rhand_kpts[i*3+1]
            conf = rhand_kpts[i*3+2]
            rhand_points.append([x, y, conf])
        hands.append(rhand_points)
    
    faces_array = np.array(faces) if faces else np.array([])
    hands_array = np.array(hands) if hands else np.array([])
    
    return {
        'bodies': {
            'candidate': candidate_array,
            'subset': subset_array
        },
        'faces': faces_array,
        'hands': hands_array
    }


def dwpose_format_to_pose_keypoint(candidate, faces, hands, canvas_width, canvas_height):
    """
    Convert DWPose internal format back to POSE_KEYPOINT format
    
    Args:
        candidate: Body keypoints in DWPose format
        faces: Face keypoints in DWPose format  
        hands: Hand keypoints in DWPose format
        canvas_width: Canvas width for coordinate conversion
        canvas_height: Canvas height for coordinate conversion
    
    Returns:
        list: POSE_KEYPOINT format data
    """
    
    # Convert body keypoints
    body_keypoints = []
    if len(candidate) > 0:
        for i in range(min(25, len(candidate))):  # Support up to 25 points including toes
            x = candidate[i][0] * canvas_width if canvas_width > 0 else candidate[i][0]
            y = candidate[i][1] * canvas_height if canvas_height > 0 else candidate[i][1]
            conf = candidate[i][2] if len(candidate[i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
            body_keypoints.extend([x, y, conf])
    
    # Pad to 25 points if needed
    while len(body_keypoints) < 75:  # 25 * 3
        body_keypoints.extend([0.0, 0.0, 0.0])
    
    # Convert face keypoints
    face_keypoints = []
    if len(faces) > 0 and len(faces[0]) > 0:
        for i in range(min(70, len(faces[0]))):
            x = faces[0][i][0] * canvas_width if canvas_width > 0 else faces[0][i][0]
            y = faces[0][i][1] * canvas_height if canvas_height > 0 else faces[0][i][1]
            conf = faces[0][i][2] if len(faces[0][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
            face_keypoints.extend([x, y, conf])
    
    # Pad to 70 points if needed
    while len(face_keypoints) < 210:  # 70 * 3
        face_keypoints.extend([0.0, 0.0, 0.0])
    
    # Convert hand keypoints
    lhand_keypoints = []
    rhand_keypoints = []
    
    if len(hands) > 0:
        # Left hand
        if len(hands[0]) > 0:
            for i in range(min(21, len(hands[0]))):
                x = hands[0][i][0] * canvas_width if canvas_width > 0 else hands[0][i][0]
                y = hands[0][i][1] * canvas_height if canvas_height > 0 else hands[0][i][1]
                conf = hands[0][i][2] if len(hands[0][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
                lhand_keypoints.extend([x, y, conf])
        
        # Right hand
        if len(hands) > 1 and len(hands[1]) > 0:
            for i in range(min(21, len(hands[1]))):
                x = hands[1][i][0] * canvas_width if canvas_width > 0 else hands[1][i][0]
                y = hands[1][i][1] * canvas_height if canvas_height > 0 else hands[1][i][1]
                conf = hands[1][i][2] if len(hands[1][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
                rhand_keypoints.extend([x, y, conf])
    
    # Pad hand keypoints to 21 points if needed
    while len(lhand_keypoints) < 63:  # 21 * 3
        lhand_keypoints.extend([0.0, 0.0, 0.0])
    while len(rhand_keypoints) < 63:  # 21 * 3
        rhand_keypoints.extend([0.0, 0.0, 0.0])
    
    # Create POSE_KEYPOINT structure
    person_data = {
        "pose_keypoints_2d": body_keypoints,
        "face_keypoints_2d": face_keypoints,
        "hand_left_keypoints_2d": lhand_keypoints,
        "hand_right_keypoints_2d": rhand_keypoints
    }
    
    frame_data = {
        "version": "1.0",
        "people": [person_data] if len(candidate) > 0 else [],
        "canvas_width": canvas_width,
        "canvas_height": canvas_height
    }
    
    return frame_data