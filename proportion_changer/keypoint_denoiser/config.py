"""
Configuration and Parameters for KeyPoint Denoiser
デノイザーの設定とパラメータ管理
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DenoiserConfig:
    """デノイザー設定クラス"""
    
    # カルマンフィルタパラメータ
    qv_base: float = 0.003**2                # 基本速度ノイズ
    qv_high_speed_factor: float = 2.0        # 高速動作時の増加係数
    qv_orientation_factor: float = 0.5       # 側面時の増加係数
    
    r_base: float = 0.004**2                 # 基本観測ノイズ
    r_confidence_gamma: float = 1.5          # 信頼度依存性指数
    r_orientation_factor: float = 0.3        # 側面時の増加係数
    
    # ゲーティング閾値
    gate_threshold: float = 9.21             # χ²(df=2, p=0.99)
    gate_threshold_recovery: float = 12.0    # 復帰時の緩和閾値
    consecutive_reject_limit: int = 5        # 連続棄却上限
    
    # 信頼度・検出パラメータ
    conf_min: float = 0.35                   # 欠測判定閾値
    orientation_threshold: float = 0.7       # 正面フレーム判定閾値
    min_shoulder_width: float = 0.02         # 最小肩幅（正規化座標）
    min_eye_distance: float = 0.015          # 最小目間距離
    
    # 体スケール推定
    min_samples_for_stats: int = 3           # 統計計算最小サンプル数
    outlier_fence_multiplier: float = 1.5    # Tukeyフェンス係数
    fallback_body_scale: float = 0.15        # フォールバック体スケール
    
    # ダンス動作保護パラメータ
    spin_speed_threshold_factor: float = 0.15      # 体スケール比での高速閾値
    spin_direction_threshold: float = -0.5         # 逆方向判定（cosθ）
    spin_speed_similarity: float = 0.6             # 左右速度類似度
    spin_protection_factor: float = 2.0            # 保護時の閾値緩和
    
    jump_y_speed_threshold_factor: float = 0.1     # Y方向速度閾値
    jump_consistency_threshold: float = 0.5        # 速度一貫性（std/mean）
    jump_min_keypoints: int = 4                    # 最小検証キーポイント数
    jump_protection_factor: float = 1.8            # 保護時の閾値緩和
    
    coordination_correlation_threshold: float = 0.6    # 親子相関閾値
    coordination_symmetric_threshold: float = 0.5      # 対称相関閾値
    coordination_min_speed_factor: float = 0.05        # 最小速度（体スケール比）
    coordination_protection_factor: float = 1.5       # 保護時の閾値緩和
    
    # 構造投影パラメータ
    max_iterations: int = 2                  # 最大投影反復数
    adjustment_ratio: float = 0.5            # 段階調整係数
    convergence_threshold: float = 0.001     # 収束判定閾値
    projection_interval: int = 3             # 投影適用間隔（フレーム）
    
    # 骨長制約重み
    bone_constraint_weights: Dict[str, float] = None
    orientation_factors: Dict[str, float] = None
    
    # スムージングパラメータ
    light_window_size: int = 3               # 軽量ウィンドウサイズ
    short_gap_threshold: int = 4             # 短ギャップ上限（線形補間）
    medium_gap_threshold: int = 12           # 中ギャップ上限（スプライン）
    
    # 処理制御フラグ
    use_rts_smoother: bool = True            # RTSスムーザ使用フラグ
    enable_dance_protection: bool = True     # ダンス動作保護フラグ
    enable_bone_constraints: bool = True     # 構造投影フラグ
    verbose_logging: bool = False            # 詳細ログ出力フラグ
    
    def __post_init__(self):
        """初期化後の設定"""
        if self.bone_constraint_weights is None:
            self.bone_constraint_weights = {
                'torso': 1.0,         # 胴体（強制約）
                'upper_limbs': 0.8,   # 上肢
                'lower_limbs': 0.8,   # 下肢
                'extremities': 0.4,   # 末端（弱制約）
            }
        
        if self.orientation_factors is None:
            self.orientation_factors = {
                'shoulder_width_min': 0.25,    # 肩幅最小（側面時）
                'upper_limb_min': 0.5,         # 腕最小（側面時）
                'general_min': 0.8,            # 一般骨最小
                'general_max': 1.2,            # 一般骨最大
            }

# プリセット設定
PRECISION_CONFIG = DenoiserConfig(
    qv_base=0.002**2,
    gate_threshold=7.5,
    max_iterations=3,
    projection_interval=1,
    light_window_size=5,
    use_rts_smoother=True,
    verbose_logging=True
)

PERFORMANCE_CONFIG = DenoiserConfig(
    qv_base=0.005**2,
    gate_threshold=12.0,
    max_iterations=1,
    projection_interval=5,
    light_window_size=1,
    use_rts_smoother=False,
    enable_dance_protection=False,
    enable_bone_constraints=False,
    verbose_logging=False
)

DANCE_CONFIG = DenoiserConfig(
    gate_threshold_recovery=15.0,
    spin_protection_factor=3.0,
    jump_protection_factor=2.5,
    coordination_protection_factor=2.0,
    max_iterations=1,
    projection_interval=5,
    enable_dance_protection=True
)

# DWPose骨格構造定義
SKELETON_PAIRS = {
    'torso_length': (1, 8),      # 首 -> 左腰 (代表)
    'upper_arm_r': (2, 3),       # 右肩 -> 右肘
    'upper_arm_l': (5, 6),       # 左肩 -> 左肘
    'forearm_r': (3, 4),         # 右肘 -> 右手首
    'forearm_l': (6, 7),         # 左肘 -> 左手首
    'thigh_r': (11, 12),         # 右腰 -> 右膝
    'thigh_l': (8, 9),           # 左腰 -> 左膝
    'calf_r': (12, 13),          # 右膝 -> 右足首
    'calf_l': (9, 10),           # 左膝 -> 左足首
}

# キーポイント分類
FACE_KEYPOINT_INDICES = [0, 14, 15, 16, 17]  # 鼻、両目、両耳
EXTREMITY_INDICES = [4, 7, 10, 13]           # 手首、足首

def get_config_by_name(config_name: str) -> DenoiserConfig:
    """設定名から設定オブジェクトを取得"""
    config_map = {
        'precision': PRECISION_CONFIG,
        'performance': PERFORMANCE_CONFIG, 
        'dance_optimized': DANCE_CONFIG,
        'balanced': DenoiserConfig()  # デフォルト
    }
    
    return config_map.get(config_name.lower(), DenoiserConfig())

def create_custom_config(**kwargs) -> DenoiserConfig:
    """カスタム設定を作成"""
    base_config = DenoiserConfig()
    
    # 辞書として取得してアップデート
    config_dict = base_config.__dict__.copy()
    config_dict.update(kwargs)
    
    return DenoiserConfig(**config_dict)