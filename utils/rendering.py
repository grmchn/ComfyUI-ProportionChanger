"""
DWPose rendering utilities for drawing poses on canvas
Handles body, hand, and face keypoint visualization with DWPose styling
"""

import cv2
import math
import numpy as np
import colorsys
import os
import logging

# Debug mode control via environment variable (local implementation to avoid imports)
DEBUG_MODE = os.getenv('PROPORTION_CHANGER_DEBUG', 'false').lower() in ('true', '1', 'yes')
log = logging.getLogger(__name__)

def debug_log(message):
    """Conditional debug logging - only outputs if DEBUG_MODE is enabled"""
    if DEBUG_MODE:
        log.debug(f"🔍 {message}")


def draw_dwpose_render(pose_keypoint, resolution_x, show_body, show_face, show_hands, show_feet, 
                       pose_marker_size, face_marker_size, hand_marker_size):
    """
    Render POSE_KEYPOINT data using DWPose style with 25-point support including toe keypoints
    Compatible with ultimate-openpose-render parameters but using DWPose rendering algorithms
    """
    
    if not pose_keypoint or len(pose_keypoint) == 0:
        return []
    
    pose_imgs = []
    
    # Handle single frame vs multi-frame input
    if isinstance(pose_keypoint, dict):
        frames = [pose_keypoint]
    else:
        frames = pose_keypoint
    
    for frame_data in frames:
        if 'people' not in frame_data or len(frame_data['people']) == 0:
            # Create empty image for frames with no people
            H = frame_data.get('canvas_height', 768)
            W = frame_data.get('canvas_width', 512)
            if resolution_x > 0:
                W = resolution_x
                H = int(frame_data.get('canvas_height', 768) * (W / frame_data.get('canvas_width', 512)))
            pose_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
            
        # Get canvas dimensions
        H = frame_data.get('canvas_height', 768)
        W = frame_data.get('canvas_width', 512)
        
        # Apply resolution scaling
        if resolution_x > 0:
            W_scaled = resolution_x
            H_scaled = int(H * (W_scaled / W))
        else:
            W_scaled, H_scaled = W, H
        
        # Create canvas
        canvas = np.zeros((H_scaled, W_scaled, 3), dtype=np.uint8)
        
        # Process each person in the frame
        for person in frame_data['people']:
            # Draw body keypoints
            if show_body and 'pose_keypoints_2d' in person:
                canvas = draw_dwpose_body_and_foot(canvas, person['pose_keypoints_2d'], 
                                                   W_scaled, H_scaled, pose_marker_size, show_feet)
            
            # Draw hand keypoints
            if show_hands:
                if 'hand_left_keypoints_2d' in person:
                    canvas = draw_dwpose_handpose(canvas, person['hand_left_keypoints_2d'], 
                                                  W_scaled, H_scaled, hand_marker_size)
                if 'hand_right_keypoints_2d' in person:
                    canvas = draw_dwpose_handpose(canvas, person['hand_right_keypoints_2d'], 
                                                  W_scaled, H_scaled, hand_marker_size)
            
            # Draw face keypoints
            if show_face and 'face_keypoints_2d' in person:
                canvas = draw_dwpose_facepose(canvas, person['face_keypoints_2d'], 
                                              W_scaled, H_scaled, face_marker_size)
        
        pose_imgs.append(canvas)
    
    return pose_imgs


def draw_dwpose_body_and_foot(canvas, body_keypoints, W, H, pose_marker_size, show_feet):
    """
    Draw body and foot keypoints using DWPose style (25-point support)
    Based on WanVideo DWPose draw_body_and_foot function
    """
    
    if not body_keypoints or len(body_keypoints) < 54:  # At least 18 points * 3
        return canvas
    
    # Define limb connections (bone structure)
    if show_feet and len(body_keypoints) >= 75:  # 25 points * 3
        # DWPose 25-point with toe connections
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], 
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], 
                   [1, 16], [16, 18], [14, 19], [11, 20]]  # Added toe connections
    else:
        # Standard 18-point OpenPose
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], 
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], 
                   [1, 16], [16, 18]]
    
    # Color palette for bones
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [170, 255, 255], [255, 255, 0]]
    
    # First pass: determine coordinate system by checking all coordinates
    max_coord = 0
    for i in range(0, len(body_keypoints), 3):
        if i + 2 < len(body_keypoints):
            x_raw = body_keypoints[i]
            y_raw = body_keypoints[i + 1]
            if x_raw > 0 and y_raw > 0:
                max_coord = max(max_coord, x_raw, y_raw)
    
    # Determine if coordinates are normalized or already in pixel space
    is_normalized = max_coord <= 2.0
    
    # Convert keypoints to coordinate pairs with consistent coordinate system
    keypoints = []
    confidences = []
    
    for i in range(0, len(body_keypoints), 3):
        if i + 2 < len(body_keypoints):
            x_raw = body_keypoints[i]
            y_raw = body_keypoints[i + 1]
            conf = body_keypoints[i + 2]
            
            # Apply consistent coordinate transformation
            if is_normalized:
                # Data is normalized (0-1), convert to pixel coordinates
                x = x_raw * W
                y = y_raw * H
            else:
                # Data is already in pixel coordinates, use as-is
                x = x_raw
                y = y_raw
            
            keypoints.append([x, y])
            confidences.append(conf)
    
    # Debug: check if we have valid keypoints
    valid_keypoints = sum(1 for conf in confidences if conf > 0.0)
    debug_log(f"Body Debug - Canvas size: {W}x{H}, Max coord: {max_coord:.4f}, Normalized: {is_normalized}")
    debug_log(f"Body Debug - Valid keypoints: {valid_keypoints}/{len(confidences)}")
    debug_log(f"Body Debug - First 3 keypoints: {[(i, f'{keypoints[i][0]:.1f}, {keypoints[i][1]:.1f}', f'{confidences[i]:.3f}') for i in range(min(3, len(keypoints)))]}")
    
    if valid_keypoints == 0:
        debug_log("Body Debug - No valid keypoints, returning black screen")
        # Return black screen for failed pose detection
        return canvas
    
    # Draw limb connections (bones)
    bones_drawn = 0
    for i, limb in enumerate(limbSeq):
        if len(keypoints) >= max(limb[0], limb[1]):
            pt1_idx, pt2_idx = limb[0] - 1, limb[1] - 1  # Convert to 0-based indexing
            
            # Use more lenient confidence threshold
            if (pt1_idx < len(confidences) and pt2_idx < len(confidences) and 
                confidences[pt1_idx] > 0.0 and confidences[pt2_idx] > 0.0):
                
                x1, y1 = keypoints[pt1_idx]
                x2, y2 = keypoints[pt2_idx]
                
                # More lenient coordinate check
                if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                    # Calculate bone properties (following WanVideo DWPose convention)
                    # Note: WanVideo uses X=height, Y=width convention
                    X1, Y1 = y1, x1  # Convert to WanVideo coordinate convention
                    X2, Y2 = y2, x2
                    mX = (X1 + X2) / 2  # Mean height coordinate
                    mY = (Y1 + Y2) / 2  # Mean width coordinate
                    length = ((X1 - X2) ** 2 + (Y1 - Y2) ** 2) ** 0.5
                    
                    if length > 1:  # Only draw if bone has reasonable length
                        # Use WanVideo's angle calculation
                        angle = math.degrees(math.atan2(X1 - X2, Y1 - Y2))
                        
                        # Draw bone as ellipse polygon (note coordinate order: Y, X)
                        stick_width = max(1, pose_marker_size)
                        polygon = cv2.ellipse2Poly((int(mY), int(mX)), 
                                                   (int(length / 2), stick_width), 
                                                   int(angle), 0, 360, 1)
                        color_idx = min(i, len(colors) - 1)
                        cv2.fillConvexPoly(canvas, polygon, colors[color_idx])
                        bones_drawn += 1
    
    # Apply transparency to bones only if bones were drawn
    if bones_drawn > 0:
        canvas = (canvas * 0.6).astype(np.uint8)
    
    # Draw keypoint markers
    if pose_marker_size > 0:
        max_points = min(len(keypoints), 25 if show_feet else 18)
        points_drawn = 0
        for i in range(max_points):
            if i < len(confidences) and confidences[i] > 0.0:
                x, y = keypoints[i]
                if x >= 0 and y >= 0 and x < W and y < H:
                    color_idx = min(i, len(colors) - 1)
                    cv2.circle(canvas, (int(x), int(y)), pose_marker_size, colors[color_idx], thickness=-1)
                    points_drawn += 1
        
        # No debug points - let failed poses show as black screen
    
    return canvas


def draw_dwpose_handpose(canvas, hand_keypoints, W, H, hand_marker_size):
    """
    Draw hand keypoints using DWPose style
    Based on WanVideo DWPose draw_handpose function
    """
    
    if not hand_keypoints or len(hand_keypoints) < 63:  # 21 points * 3
        return canvas
    
    # Hand bone connections
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # First pass: determine coordinate system
    max_coord = 0
    for i in range(0, len(hand_keypoints), 3):
        if i + 2 < len(hand_keypoints):
            x_raw = hand_keypoints[i]
            y_raw = hand_keypoints[i + 1]
            if x_raw > 0 and y_raw > 0:
                max_coord = max(max_coord, x_raw, y_raw)
    
    is_normalized = max_coord <= 2.0
    
    # Convert keypoints to coordinate pairs with consistent coordinate system
    keypoints = []
    confidences = []
    
    for i in range(0, len(hand_keypoints), 3):
        if i + 2 < len(hand_keypoints):
            x_raw = hand_keypoints[i]
            y_raw = hand_keypoints[i + 1]
            conf = hand_keypoints[i + 2]
            
            # Apply consistent coordinate transformation
            if is_normalized:
                x = x_raw * W
                y = y_raw * H
            else:
                x = x_raw
                y = y_raw
                
            keypoints.append([x, y])
            confidences.append(conf)
    
    # Draw hand connections
    if hand_marker_size > 0:
        for ie, edge in enumerate(edges):
            if (len(keypoints) > max(edge) and 
                confidences[edge[0]] > 0.0 and confidences[edge[1]] > 0.0):
                
                x1, y1 = keypoints[edge[0]]
                x2, y2 = keypoints[edge[1]]
                
                if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                    # Generate color using HSV
                    h = (ie / float(len(edges))) % 1.0
                    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
                    color = (int(255 * r), int(255 * g), int(255 * b))
                    cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
    
    # Draw hand keypoints
    if hand_marker_size > 0:
        for i, (x, y) in enumerate(keypoints):
            if i < len(confidences) and confidences[i] > 0.0 and x >= 0 and y >= 0:
                cv2.circle(canvas, (int(x), int(y)), hand_marker_size, (0, 0, 255), thickness=-1)
    
    return canvas


def draw_dwpose_facepose(canvas, face_keypoints, W, H, face_marker_size):
    """
    Draw face keypoints using DWPose style
    Based on WanVideo DWPose draw_facepose function
    """
    
    if not face_keypoints or len(face_keypoints) < 210:  # 70 points * 3
        return canvas
    
    # First pass: determine coordinate system
    max_coord = 0
    for i in range(0, len(face_keypoints), 3):
        if i + 2 < len(face_keypoints):
            x_raw = face_keypoints[i]
            y_raw = face_keypoints[i + 1]
            if x_raw > 0 and y_raw > 0:
                max_coord = max(max_coord, x_raw, y_raw)
    
    is_normalized = max_coord <= 2.0
    
    # Convert keypoints to coordinate pairs with consistent coordinate system
    keypoints = []
    confidences = []
    
    for i in range(0, len(face_keypoints), 3):
        if i + 2 < len(face_keypoints):
            x_raw = face_keypoints[i]
            y_raw = face_keypoints[i + 1]
            conf = face_keypoints[i + 2]
            
            # Apply consistent coordinate transformation
            if is_normalized:
                x = x_raw * W
                y = y_raw * H
            else:
                x = x_raw
                y = y_raw
                
            keypoints.append([x, y])
            confidences.append(conf)
    
    # Draw face keypoints
    if face_marker_size > 0:
        for i, (x, y) in enumerate(keypoints):
            if i < len(confidences) and confidences[i] > 0.0 and x >= 0 and y >= 0:
                cv2.circle(canvas, (int(x), int(y)), face_marker_size, (255, 255, 255), thickness=-1)
    
    return canvas