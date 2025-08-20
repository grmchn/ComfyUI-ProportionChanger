"""
ProportionChanger interpolation node class
Provides keypoint interpolation for creating smooth animations between poses
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union


class ProportionChangerInterpolator:
    """
    Interpolates between keypoint sequences to create smooth animation frames
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),  # Batch processing supported (singular notation but accepts lists)
                "interpolation_frames": ("INT", {"default": 0, "min": 0, "max": 5}),  # 0=no interpolation, 1=add 1 frame, 2=add 2 frames...
                "method": (["linear", "momentum", "bezier", "ease_in_out"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("interpolated_sequence",)
    FUNCTION = "process"
    CATEGORY = "ProportionChanger"

    def process(self, pose_keypoint, interpolation_frames: int = 0, method: str = "linear"):
        """
        Process keypoint interpolation with new simplified specification
        
        Args:
            pose_keypoint: Input keypoint sequence (can be single frame or sequence)
            interpolation_frames: Number of frames to interpolate between each pair (0-5)
            method: Interpolation method to use
            
        Returns:
            interpolated_sequence: Processed sequence with interpolated frames
        """
        
        # Handle single frame input - return as-is
        if interpolation_frames == 0:
            return (pose_keypoint,)
        
        # Convert input to sequence format if needed
        if isinstance(pose_keypoint, dict):
            # Single frame input - cannot interpolate
            return (pose_keypoint,)
        elif isinstance(pose_keypoint, list) and len(pose_keypoint) < 2:
            # Insufficient frames for interpolation
            return (pose_keypoint,)
        
        # Process sequence interpolation
        try:
            interpolated_result = self._process_sequence_interpolation(
                pose_keypoint, interpolation_frames, method
            )
            return (interpolated_result,)
        except Exception as e:
            print(f"Error during interpolation: {e}")
            return (pose_keypoint,)  # Return original on error
    
    def _process_sequence_interpolation(self, sequence: List[Dict], interpolation_frames: int, method: str) -> List[Dict]:
        """
        Process sequence interpolation with batch processing
        
        Args:
            sequence: List of keypoint frames
            interpolation_frames: Number of frames to interpolate between each pair
            method: Interpolation method
            
        Returns:
            Interpolated sequence with additional frames
        """
        if len(sequence) < 2:
            return sequence
        
        interpolated_sequence = []
        
        # Process each consecutive frame pair
        for i in range(len(sequence) - 1):
            current_frame = sequence[i]
            next_frame = sequence[i + 1]
            
            # Add current frame
            interpolated_sequence.append(current_frame)
            
            # Generate interpolated frames between current and next
            for j in range(interpolation_frames):
                t = (j + 1) / (interpolation_frames + 1)
                
                # Choose interpolation method
                if method == "linear":
                    interp_frame = self._linear_interpolation(current_frame, next_frame, t)
                elif method == "ease_in_out":
                    interp_frame = self._ease_in_out_interpolation(current_frame, next_frame, t)
                elif method == "bezier":
                    interp_frame = self._bezier_interpolation(current_frame, next_frame, t)
                elif method == "momentum":
                    # For momentum, use previous frame if available
                    prev_frame = sequence[i - 1] if i > 0 else None
                    next_next_frame = sequence[i + 2] if i + 2 < len(sequence) else None
                    interp_frame = self._momentum_interpolation(
                        prev_frame, current_frame, next_frame, next_next_frame, t
                    )
                else:
                    # Fallback to linear
                    interp_frame = self._linear_interpolation(current_frame, next_frame, t)
                
                interpolated_sequence.append(interp_frame)
        
        # Add final frame
        interpolated_sequence.append(sequence[-1])
        
        return interpolated_sequence
    
    def _linear_interpolation(self, frame_a: Dict, frame_b: Dict, t: float) -> Dict:
        """
        Linear interpolation between two keypoint frames
        """
        result_frame = {
            'people': [],
            'canvas_width': frame_a.get('canvas_width', 512),
            'canvas_height': frame_a.get('canvas_height', 768)
        }
        
        # Interpolate each person
        max_people = max(len(frame_a.get('people', [])), len(frame_b.get('people', [])))
        
        for person_idx in range(max_people):
            person_a = frame_a.get('people', [{}])[min(person_idx, len(frame_a.get('people', [])) - 1)] if frame_a.get('people') else {}
            person_b = frame_b.get('people', [{}])[min(person_idx, len(frame_b.get('people', [])) - 1)] if frame_b.get('people') else {}
            
            interpolated_person = {}
            
            # Interpolate each keypoint type
            for keypoint_type in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                if keypoint_type in person_a or keypoint_type in person_b:
                    kps_a = person_a.get(keypoint_type, [])
                    kps_b = person_b.get(keypoint_type, [])
                    interpolated_person[keypoint_type] = self._interpolate_keypoint_array(kps_a, kps_b, t)
            
            result_frame['people'].append(interpolated_person)
        
        return result_frame
    
    def _interpolate_keypoint_array(self, kps_a: List[float], kps_b: List[float], t: float) -> List[float]:
        """
        Interpolate keypoint arrays with proper confidence handling
        """
        if not kps_a and not kps_b:
            return []
        
        # Use non-empty array or pad shorter array
        max_len = max(len(kps_a), len(kps_b))
        kps_a_padded = kps_a + [0.0] * (max_len - len(kps_a))
        kps_b_padded = kps_b + [0.0] * (max_len - len(kps_b))
        
        interpolated = []
        for i in range(0, max_len, 3):  # Process x, y, confidence triplets
            if i + 2 < max_len:
                # Linear interpolation for x, y coordinates
                x = kps_a_padded[i] + t * (kps_b_padded[i] - kps_a_padded[i])
                y = kps_a_padded[i + 1] + t * (kps_b_padded[i + 1] - kps_a_padded[i + 1])
                
                # Use minimum confidence (conservative approach)
                conf = min(kps_a_padded[i + 2], kps_b_padded[i + 2])
                
                interpolated.extend([x, y, conf])
        
        return interpolated
    
    def _ease_in_out_interpolation(self, frame_a: Dict, frame_b: Dict, t: float) -> Dict:
        """
        Ease-in-out interpolation using cubic curves
        """
        # Apply cubic ease-in-out to t
        if t < 0.5:
            eased_t = 4 * t * t * t
        else:
            eased_t = 1 - 4 * (1 - t) * (1 - t) * (1 - t)
        
        return self._linear_interpolation(frame_a, frame_b, eased_t)
    
    def _bezier_interpolation(self, frame_a: Dict, frame_b: Dict, t: float) -> Dict:
        """
        Bezier curve interpolation with automatic control point generation
        """
        # Generate automatic control point (simple upward curve)
        result_frame = {
            'people': [],
            'canvas_width': frame_a.get('canvas_width', 512),
            'canvas_height': frame_a.get('canvas_height', 768)
        }
        
        max_people = max(len(frame_a.get('people', [])), len(frame_b.get('people', [])))
        
        for person_idx in range(max_people):
            person_a = frame_a.get('people', [{}])[min(person_idx, len(frame_a.get('people', [])) - 1)] if frame_a.get('people') else {}
            person_b = frame_b.get('people', [{}])[min(person_idx, len(frame_b.get('people', [])) - 1)] if frame_b.get('people') else {}
            
            interpolated_person = {}
            
            for keypoint_type in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                if keypoint_type in person_a or keypoint_type in person_b:
                    kps_a = person_a.get(keypoint_type, [])
                    kps_b = person_b.get(keypoint_type, [])
                    interpolated_person[keypoint_type] = self._bezier_interpolate_keypoint_array(kps_a, kps_b, t)
            
            result_frame['people'].append(interpolated_person)
        
        return result_frame
    
    def _bezier_interpolate_keypoint_array(self, kps_a: List[float], kps_b: List[float], t: float) -> List[float]:
        """
        Bezier interpolation for keypoint arrays
        """
        if not kps_a and not kps_b:
            return []
        
        max_len = max(len(kps_a), len(kps_b))
        kps_a_padded = kps_a + [0.0] * (max_len - len(kps_a))
        kps_b_padded = kps_b + [0.0] * (max_len - len(kps_b))
        
        interpolated = []
        for i in range(0, max_len, 3):
            if i + 2 < max_len:
                # Calculate control point (automatic curve generation)
                x_start, y_start = kps_a_padded[i], kps_a_padded[i + 1]
                x_end, y_end = kps_b_padded[i], kps_b_padded[i + 1]
                
                # Control point: middle x, upward y offset
                x_control = (x_start + x_end) / 2
                y_offset = abs(x_end - x_start) * 0.25  # Proportional to distance
                y_control = (y_start + y_end) / 2 - y_offset * 0.5  # Upward curve
                
                # 3-point Bezier: B(t) = (1-t)²*P0 + 2*(1-t)*t*P1 + t²*P2
                x = (1-t)**2 * x_start + 2*(1-t)*t * x_control + t**2 * x_end
                y = (1-t)**2 * y_start + 2*(1-t)*t * y_control + t**2 * y_end
                
                # Use minimum confidence
                conf = min(kps_a_padded[i + 2], kps_b_padded[i + 2])
                
                interpolated.extend([x, y, conf])
        
        return interpolated
    
    def _momentum_interpolation(self, prev_frame: Optional[Dict], current_frame: Dict, 
                              next_frame: Dict, next_next_frame: Optional[Dict], t: float) -> Dict:
        """
        Momentum-based interpolation considering velocity vectors
        """
        # Fixed momentum weight for internal use
        momentum_weight = 0.5
        
        if not prev_frame:
            # Fallback to bezier if no previous frame
            return self._bezier_interpolation(current_frame, next_frame, t)
        
        result_frame = {
            'people': [],
            'canvas_width': current_frame.get('canvas_width', 512),
            'canvas_height': current_frame.get('canvas_height', 768)
        }
        
        max_people = max(len(current_frame.get('people', [])), len(next_frame.get('people', [])))
        
        for person_idx in range(max_people):
            person_prev = prev_frame.get('people', [{}])[min(person_idx, len(prev_frame.get('people', [])) - 1)] if prev_frame.get('people') else {}
            person_curr = current_frame.get('people', [{}])[min(person_idx, len(current_frame.get('people', [])) - 1)] if current_frame.get('people') else {}
            person_next = next_frame.get('people', [{}])[min(person_idx, len(next_frame.get('people', [])) - 1)] if next_frame.get('people') else {}
            
            interpolated_person = {}
            
            for keypoint_type in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                if keypoint_type in person_curr or keypoint_type in person_next:
                    kps_prev = person_prev.get(keypoint_type, [])
                    kps_curr = person_curr.get(keypoint_type, [])
                    kps_next = person_next.get(keypoint_type, [])
                    interpolated_person[keypoint_type] = self._momentum_interpolate_keypoint_array(
                        kps_prev, kps_curr, kps_next, t, momentum_weight
                    )
            
            result_frame['people'].append(interpolated_person)
        
        return result_frame
    
    def _momentum_interpolate_keypoint_array(self, kps_prev: List[float], kps_curr: List[float], 
                                           kps_next: List[float], t: float, momentum_weight: float = 0.5) -> List[float]:
        """
        Momentum-based interpolation using velocity vectors
        """
        if not kps_curr and not kps_next:
            return []
        
        max_len = max(len(kps_prev), len(kps_curr), len(kps_next))
        kps_prev_padded = kps_prev + [0.0] * (max_len - len(kps_prev)) if kps_prev else [0.0] * max_len
        kps_curr_padded = kps_curr + [0.0] * (max_len - len(kps_curr))
        kps_next_padded = kps_next + [0.0] * (max_len - len(kps_next))
        
        interpolated = []
        for i in range(0, max_len, 3):
            if i + 2 < max_len:
                # Calculate velocity vectors
                vel_in_x = kps_curr_padded[i] - kps_prev_padded[i]
                vel_in_y = kps_curr_padded[i + 1] - kps_prev_padded[i + 1]
                vel_out_x = kps_next_padded[i] - kps_curr_padded[i]
                vel_out_y = kps_next_padded[i + 1] - kps_curr_padded[i + 1]
                
                # Momentum vector calculation
                momentum_x = momentum_weight * vel_in_x + (1 - momentum_weight) * vel_out_x
                momentum_y = momentum_weight * vel_in_y + (1 - momentum_weight) * vel_out_y
                
                # Bezier control point from momentum
                control_x = kps_curr_padded[i] + momentum_x * 0.5
                control_y = kps_curr_padded[i + 1] + momentum_y * 0.5
                
                # 3-point Bezier with momentum-generated control point
                x = (1-t)**2 * kps_curr_padded[i] + 2*(1-t)*t * control_x + t**2 * kps_next_padded[i]
                y = (1-t)**2 * kps_curr_padded[i + 1] + 2*(1-t)*t * control_y + t**2 * kps_next_padded[i + 1]
                
                # Use minimum confidence
                conf = min(kps_curr_padded[i + 2], kps_next_padded[i + 2])
                
                interpolated.extend([x, y, conf])
        
        return interpolated