"""
KeyPoint Denoiser Module
カルマンフィルタベースの時系列KeyPointデノイザー実装

Based on expert AI review and comprehensive algorithm design in Issue024
外部AI専門家レビューを統合した最新アルゴリズム実装
"""

# Core components export
from .config import DenoiserConfig, PRECISION_CONFIG, PERFORMANCE_CONFIG, DANCE_CONFIG
from .kalman_filter import KeypointKalmanFilter
from .body_analysis import calculate_enhanced_orientation_score, calculate_joint_distances_enhanced
from .core_algorithm import denoise_pose_keypoints_kalman

__all__ = [
    'DenoiserConfig',
    'PRECISION_CONFIG', 
    'PERFORMANCE_CONFIG',
    'DANCE_CONFIG',
    'KeypointKalmanFilter',
    'calculate_enhanced_orientation_score',
    'calculate_joint_distances_enhanced', 
    'denoise_pose_keypoints_kalman'
]

__version__ = "1.0.0"
__author__ = "ProportionChanger Team"