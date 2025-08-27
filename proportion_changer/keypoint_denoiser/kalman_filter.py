"""
Kalman Filter Implementation for KeyPoint Tracking
キーポイント追跡用カルマンフィルタ実装
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
try:
    from .config import DenoiserConfig
except ImportError:
    # スタンドアロンテスト用
    from config import DenoiserConfig

class KeypointKalmanFilter:
    """
    各キーポイント用の2D定速度カルマンフィルタ
    状態: [x, y, vx, vy] - 位置と速度を同時推定
    """
    
    def __init__(self, initial_pos: np.ndarray, conf: float, body_scale: float, config: DenoiserConfig):
        # 状態ベクトル [x, y, vx, vy]
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32)
        
        # 状態遷移行列 (減衰付き等速度モデル)
        velocity_decay = 0.8  # 速度減衰係数
        self.A = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, velocity_decay, 0],  # vx' = vx * decay (速度減衰)
            [0, 0, 0, velocity_decay]   # vy' = vy * decay
        ], dtype=np.float32)
        
        # 観測行列 (位置のみ観測)
        self.H = np.array([
            [1, 0, 0, 0],  # observe x
            [0, 1, 0, 0]   # observe y
        ], dtype=np.float32)
        
        # 初期共分散行列（より保守的な初期化）
        self.P = np.diag([1.0, 1.0, 0.1, 0.1]).astype(np.float32)
        
        # パラメータ保存
        self.body_scale = body_scale
        self.config = config
        self.last_confidence = conf
        self.consecutive_rejects = 0
        
        # ヒストリー（デバッグ用）
        self.prediction_history = []
        self.measurement_history = []
        self.mahalanobis_history = []
        
    def predict(self, current_speed: float = 0.0, orientation: float = 0.5) -> np.ndarray:
        """予測ステップ - プロセスノイズQを動的調整"""
        # 予測
        self.x = self.A @ self.x
        
        # プロセスノイズQ（速度・向き・動作速度に応じて調整）
        qv_base = self.config.qv_base
        
        # 高速動作時はプロセスノイズを増加（急激な方向転換を許容）
        speed_factor = 1.0 + min(current_speed / (self.body_scale * 0.2 + 1e-8), 2.0)
        
        # 向きが不明確な時はプロセスノイズを増加（予測優先）
        orientation_factor = 1.0 + (1.0 - orientation) * self.config.qv_orientation_factor
        
        qv = qv_base * speed_factor * orientation_factor
        
        Q = np.diag([0, 0, qv, qv]).astype(np.float32)
        self.P = self.A @ self.P @ self.A.T + Q
        
        predicted_pos = self.x[:2].copy()
        
        # ヒストリー記録
        if self.config.verbose_logging:
            self.prediction_history.append(predicted_pos.copy())
        
        return predicted_pos
    
    def update(self, measurement: np.ndarray, confidence: float, orientation: float = 0.5) -> Tuple[np.ndarray, bool, float]:
        """
        更新ステップ - カイ二乗ゲーティング + 動的測定ノイズR調整
        Returns: (estimated_position, accepted, mahalanobis_distance_squared)
        """
        # 測定ノイズR（信頼度・向きに応じて調整）
        r_base = self.config.r_base
        conf_factor = 1.0 / (0.1 + confidence**self.config.r_confidence_gamma)
        orientation_factor = 1.0 + (1.0 - orientation) * self.config.r_orientation_factor
        
        r = r_base * conf_factor * orientation_factor
        R = np.diag([r, r]).astype(np.float32)
        
        # イノベーション（予測誤差）
        y = measurement - self.H @ self.x
        
        # イノベーション共分散
        S = self.H @ self.P @ self.H.T + R
        
        # カイ二乗ゲーティング（df=2, p=0.99 -> 閾値≈9.21）
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist2 = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            # 特異行列の場合は棄却
            if self.config.verbose_logging:
                pass
            return self.x[:2].copy(), False, float('inf')
        
        gate_threshold = self.config.gate_threshold
        
        # ヒステリシス（復帰時は少し緩く）
        if self.consecutive_rejects > 0:
            gate_threshold = self.config.gate_threshold_recovery
        
        # ヒストリー記録
        if self.config.verbose_logging:
            self.measurement_history.append(measurement.copy())
            self.mahalanobis_history.append(mahalanobis_dist2)
        
        # デバッグ情報（最初の数回のみ）
        if self.config.verbose_logging and len(self.measurement_history) < 5:
            pass
        
        if mahalanobis_dist2 < gate_threshold:
            # 観測受け入れ：更新実行
            K = self.P @ self.H.T @ S_inv
            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
            
            # 連続棄却カウンタリセット
            self.consecutive_rejects = 0
            self.last_confidence = confidence
            
            return self.x[:2].copy(), True, mahalanobis_dist2
        else:
            # 観測棄却：予測値を使用
            self.consecutive_rejects += 1
            
            return self.x[:2].copy(), False, mahalanobis_dist2
    
    def get_velocity(self) -> np.ndarray:
        """現在の推定速度を取得"""
        return self.x[2:4].copy()
    
    def get_speed(self) -> float:
        """現在の推定速度の大きさを取得"""
        return np.linalg.norm(self.x[2:4])
    
    def get_state_summary(self) -> Dict:
        """フィルタの状態サマリーを取得"""
        return {
            'position': self.x[:2].tolist(),
            'velocity': self.x[2:4].tolist(),
            'speed': self.get_speed(),
            'consecutive_rejects': self.consecutive_rejects,
            'last_confidence': self.last_confidence,
            'covariance_trace': np.trace(self.P)
        }
    
    def reset_to_measurement(self, measurement: np.ndarray, confidence: float):
        """測定値でフィルタをリセット（長期間の棄却後の復帰用）"""
        self.x[:2] = measurement
        self.x[2:4] = 0.0  # 速度をリセット
        
        # 共分散も初期化
        self.P = np.diag([0.001, 0.001, 0.01, 0.01]).astype(np.float32)
        
        self.consecutive_rejects = 0
        self.last_confidence = confidence
        
        if self.config.verbose_logging:
            pass

def initialize_kalman_filters(first_valid_frame: List[List[float]], 
                             body_scale: float, 
                             config: DenoiserConfig) -> Dict[int, KeypointKalmanFilter]:
    """
    全キーポイント用のカルマンフィルタを初期化
    """
    filters = {}
    
    for keypoint_idx in range(25):  # DWPose 25 body keypoints
        keypoint = first_valid_frame[keypoint_idx]
        
        if keypoint[2] > config.conf_min:  # 信頼度チェック
            kf = KeypointKalmanFilter(
                initial_pos=np.array(keypoint[:2], dtype=np.float32),
                conf=keypoint[2],
                body_scale=body_scale,
                config=config
            )
            kf.keypoint_id = keypoint_idx  # デバッグ用ID追加
            filters[keypoint_idx] = kf
            
            if config.verbose_logging:
                pass
    
    if config.verbose_logging:
        pass
    
    return filters

def update_kalman_filters_batch(filters: Dict[int, KeypointKalmanFilter],
                               keypoints: List[List[float]],
                               orientation_score: float,
                               protection_factors: Optional[Dict[int, float]] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, bool], Dict[int, float]]:
    """
    全フィルタのバッチ更新処理
    
    Returns:
        positions: {keypoint_idx: estimated_position}
        acceptances: {keypoint_idx: observation_accepted}  
        mahalanobis_distances: {keypoint_idx: distance_squared}
    """
    positions = {}
    acceptances = {}
    mahalanobis_distances = {}
    
    for kp_idx, kf in filters.items():
        if kp_idx >= len(keypoints):
            continue
            
        keypoint = keypoints[kp_idx]
        confidence = keypoint[2]
        
        # 予測ステップ
        current_speed = kf.get_speed()
        predicted_pos = kf.predict(current_speed, orientation_score)
        
        if confidence >= kf.config.conf_min:
            # 測定更新
            measurement = np.array(keypoint[:2], dtype=np.float32)
            updated_pos, accepted, mahal_dist = kf.update(measurement, confidence, orientation_score)
            
            # ダンス動作保護による再判定
            if not accepted and protection_factors and kp_idx in protection_factors:
                protection_factor = protection_factors[kp_idx]
                if protection_factor > 1.0:
                    # 弱い信頼度で強制更新
                    weak_confidence = confidence * 0.5
                    updated_pos, accepted, mahal_dist = kf.update(measurement, weak_confidence, orientation_score)
                    
                    if kf.config.verbose_logging and accepted:
                        pass
            
            positions[kp_idx] = updated_pos
            acceptances[kp_idx] = accepted
            mahalanobis_distances[kp_idx] = mahal_dist
            
        else:
            # 信頼度不足：予測値を使用
            positions[kp_idx] = predicted_pos
            acceptances[kp_idx] = False
            mahalanobis_distances[kp_idx] = 0.0
    
    return positions, acceptances, mahalanobis_distances