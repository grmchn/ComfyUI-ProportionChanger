"""
KeyPoint Denoiser Node for ComfyUI
カルマンフィルタベースKeyPointデノイザーのComfyUIノード実装
"""

import copy
import traceback
from typing import List, Dict, Any, Tuple, Optional

# Support both relative and absolute imports for ComfyUI compatibility
try:
    from .keypoint_denoiser import (
        denoise_pose_keypoints_kalman,
        DenoiserConfig,
        get_config_by_name,
        create_custom_config
    )
except ImportError:
    try:
        from keypoint_denoiser import (
            denoise_pose_keypoints_kalman,
            DenoiserConfig,
            get_config_by_name,
            create_custom_config
        )
    except ImportError:
        # Fallback for standalone testing
        import sys
        import os
        current_dir = os.path.dirname(__file__)
        keypoint_denoiser_dir = os.path.join(current_dir, 'keypoint_denoiser')
        sys.path.insert(0, keypoint_denoiser_dir)
        
        from core_algorithm import denoise_pose_keypoints_kalman
        from config import DenoiserConfig, get_config_by_name, create_custom_config

class ProportionChangerKeypointDenoiser:
    """
    KeyPointデノイザーノード
    
    カルマンフィルタベースの時系列ノイズ除去により、
    ダンス動画などの高速動作でのKeyPoint飛びや検出ミスを修正
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "processing_mode": (
                    ["balanced", "precision", "performance", "dance_optimized"], 
                    {"default": "balanced"}
                ),
                "denoising_strength": (
                    "FLOAT", 
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}
                ),
            },
            "optional": {
                "confidence_threshold": (
                    "FLOAT", 
                    {"default": 0.35, "min": 0.1, "max": 0.8, "step": 0.05}
                ),
                "gate_threshold_multiplier": (
                    "FLOAT", 
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}
                ),
                "enable_bone_constraints": (
                    "BOOLEAN", 
                    {"default": True}
                ),
                "enable_rts_smoother": (
                    "BOOLEAN", 
                    {"default": True}
                ),
                "enable_verbose_logging": (
                    "BOOLEAN", 
                    {"default": False}
                ),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("denoised_pose_keypoint",)
    FUNCTION = "denoise_keypoints"
    CATEGORY = "ProportionChanger"

    def denoise_keypoints(self, 
                         pose_keypoint: List[Dict],
                         processing_mode: str = "balanced",
                         denoising_strength: float = 1.0,
                         confidence_threshold: float = 0.35,
                         gate_threshold_multiplier: float = 1.0,
                         enable_bone_constraints: bool = True,
                         enable_rts_smoother: bool = True,
                         enable_verbose_logging: bool = False) -> Tuple[List[Dict]]:
        """
        KeyPointデノイザーのメイン処理
        
        Args:
            pose_keypoint: POSE_KEYPOINT形式のバッチデータ
            processing_mode: 処理モード（balanced/precision/performance/dance_optimized）
            denoising_strength: デノイジング強度（0.0-3.0）
            confidence_threshold: 信頼度閾値
            gate_threshold_multiplier: ゲート閾値調整倍率
            enable_bone_constraints: 構造制約有効化
            enable_rts_smoother: RTSスムーザ有効化
            enable_verbose_logging: 詳細ログ有効化
            
        Returns:
            Tuple[List[Dict]]: ノイズ除去済みPOSE_KEYPOINTバッチ
        """
        
        try:
            # 入力チェック
            if not pose_keypoint or len(pose_keypoint) == 0:
                if enable_verbose_logging:
                    print("⚠️ KeyPoint Denoiser: 入力データが空です")
                return (pose_keypoint,)
            
            if len(pose_keypoint) < 3:
                if enable_verbose_logging:
                    print(f"⚠️ KeyPoint Denoiser: バッチサイズ不足（{len(pose_keypoint)}フレーム）- 最小3フレーム必要")
                return (pose_keypoint,)
            
            # 設定作成
            base_config = get_config_by_name(processing_mode)
            
            # カスタムパラメータで設定を調整
            custom_config = create_custom_config(
                # ベース設定から開始
                qv_base=base_config.qv_base * (denoising_strength ** 0.5),
                gate_threshold=base_config.gate_threshold * gate_threshold_multiplier,
                conf_min=confidence_threshold,
                enable_bone_constraints=enable_bone_constraints,
                use_rts_smoother=enable_rts_smoother,
                enable_dance_protection=base_config.enable_dance_protection,
                verbose_logging=enable_verbose_logging,
                
                # デノイジング強度による調整
                r_base=base_config.r_base / (denoising_strength + 0.5),
                max_iterations=min(base_config.max_iterations, max(1, int(denoising_strength * 2))),
                light_window_size=min(7, max(1, int(base_config.light_window_size * denoising_strength))),
            )
            
            if enable_verbose_logging:
                print(f"\n🚀 KeyPoint Denoiser 開始")
                print(f"📥 入力フレーム数: {len(pose_keypoint)}")
                print(f"⚙️ 処理モード: {processing_mode}")
                print(f"💪 デノイジング強度: {denoising_strength}")
                print(f"🎯 ゲート閾値: {custom_config.gate_threshold:.2f}")
                print(f"🔍 信頼度閾値: {custom_config.conf_min}")
                print(f"🦴 構造制約: {'ON' if enable_bone_constraints else 'OFF'}")
                print(f"🎢 RTSスムーザ: {'ON' if enable_rts_smoother else 'OFF'}")
                print("-" * 60)
            
            # メインデノイジング処理実行
            denoised_result = denoise_pose_keypoints_kalman(
                pose_keypoint_batch=pose_keypoint,
                config=custom_config
            )
            
            if enable_verbose_logging:
                print(f"\n✅ KeyPoint Denoiser 完了")
                print(f"📤 出力フレーム数: {len(denoised_result)}")
                if len(denoised_result) == len(pose_keypoint):
                    print("✅ フレーム数保持: OK")
                else:
                    print(f"⚠️ フレーム数変化: {len(pose_keypoint)} → {len(denoised_result)}")
                print("=" * 60)
            
            return (denoised_result,)
            
        except Exception as e:
            error_msg = f"KeyPoint Denoiser エラー: {str(e)}"
            print(f"\n💥 {error_msg}")
            
            if enable_verbose_logging:
                print("\n🔍 詳細エラー情報:")
                traceback.print_exc()
                print("-" * 60)
            
            # エラー時は元データを返す
            return (pose_keypoint,)

    @classmethod 
    def IS_CHANGED(cls, **kwargs):
        """ノードの変更検出（常に実行するためにランダム値を返す）"""
        import random
        return random.random()

class ProportionChangerKeypointDenoiserAdvanced:
    """
    KeyPointデノイザー上級者向けノード
    
    より詳細なパラメータ制御が可能な版
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
            },
            "optional": {
                # カルマンフィルタパラメータ
                "kalman_process_noise": (
                    "FLOAT", 
                    {"default": 0.003, "min": 0.001, "max": 0.01, "step": 0.001}
                ),
                "kalman_measurement_noise": (
                    "FLOAT", 
                    {"default": 0.004, "min": 0.001, "max": 0.02, "step": 0.001}
                ),
                "gate_threshold": (
                    "FLOAT", 
                    {"default": 9.21, "min": 5.0, "max": 20.0, "step": 0.5}
                ),
                "gate_recovery_threshold": (
                    "FLOAT", 
                    {"default": 12.0, "min": 8.0, "max": 25.0, "step": 0.5}
                ),
                
                # 信頼度・検出パラメータ
                "confidence_threshold": (
                    "FLOAT", 
                    {"default": 0.35, "min": 0.1, "max": 0.8, "step": 0.05}
                ),
                "orientation_threshold": (
                    "FLOAT", 
                    {"default": 0.7, "min": 0.5, "max": 0.95, "step": 0.05}
                ),
                
                # ダンス動作保護
                "enable_spin_protection": ("BOOLEAN", {"default": True}),
                "enable_jump_protection": ("BOOLEAN", {"default": True}),
                "enable_coordination_protection": ("BOOLEAN", {"default": True}),
                
                # 構造投影パラメータ
                "enable_bone_constraints": ("BOOLEAN", {"default": True}),
                "bone_constraint_iterations": ("INT", {"default": 2, "min": 1, "max": 5}),
                "projection_interval": ("INT", {"default": 3, "min": 1, "max": 10}),
                
                # スムージングパラメータ
                "enable_rts_smoother": ("BOOLEAN", {"default": True}),
                "smoothing_window_size": ("INT", {"default": 3, "min": 1, "max": 7}),
                
                # システム設定
                "enable_verbose_logging": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("denoised_pose_keypoint",)
    FUNCTION = "denoise_keypoints_advanced"
    CATEGORY = "ProportionChanger"

    def denoise_keypoints_advanced(self,
                                 pose_keypoint: List[Dict],
                                 kalman_process_noise: float = 0.003,
                                 kalman_measurement_noise: float = 0.004,
                                 gate_threshold: float = 9.21,
                                 gate_recovery_threshold: float = 12.0,
                                 confidence_threshold: float = 0.35,
                                 orientation_threshold: float = 0.7,
                                 enable_spin_protection: bool = True,
                                 enable_jump_protection: bool = True,
                                 enable_coordination_protection: bool = True,
                                 enable_bone_constraints: bool = True,
                                 bone_constraint_iterations: int = 2,
                                 projection_interval: int = 3,
                                 enable_rts_smoother: bool = True,
                                 smoothing_window_size: int = 3,
                                 enable_verbose_logging: bool = False) -> Tuple[List[Dict]]:
        """
        上級者向けKeyPointデノイザー処理
        """
        
        try:
            # 入力チェック
            if not pose_keypoint or len(pose_keypoint) < 3:
                if enable_verbose_logging:
                    print(f"⚠️ Advanced Denoiser: バッチサイズ不足（{len(pose_keypoint) if pose_keypoint else 0}フレーム）")
                return (pose_keypoint,)
            
            # カスタム詳細設定作成
            advanced_config = create_custom_config(
                # カルマンフィルタ
                qv_base=kalman_process_noise**2,
                r_base=kalman_measurement_noise**2,
                gate_threshold=gate_threshold,
                gate_threshold_recovery=gate_recovery_threshold,
                
                # 検出パラメータ
                conf_min=confidence_threshold,
                orientation_threshold=orientation_threshold,
                
                # ダンス保護
                enable_dance_protection=(enable_spin_protection or enable_jump_protection or enable_coordination_protection),
                spin_protection_factor=2.0 if enable_spin_protection else 1.0,
                jump_protection_factor=1.8 if enable_jump_protection else 1.0,
                coordination_protection_factor=1.5 if enable_coordination_protection else 1.0,
                
                # 構造投影
                enable_bone_constraints=enable_bone_constraints,
                max_iterations=bone_constraint_iterations,
                projection_interval=projection_interval,
                
                # スムージング
                use_rts_smoother=enable_rts_smoother,
                light_window_size=smoothing_window_size,
                
                # システム
                verbose_logging=enable_verbose_logging
            )
            
            if enable_verbose_logging:
                print(f"\n🔧 Advanced KeyPoint Denoiser 開始")
                print(f"📥 入力フレーム数: {len(pose_keypoint)}")
                print(f"🔮 カルマンプロセスノイズ: {kalman_process_noise}")
                print(f"🔮 カルマン測定ノイズ: {kalman_measurement_noise}")
                print(f"🎯 ゲート閾値: {gate_threshold}")
                print(f"🛡️ ダンス保護: スピン{enable_spin_protection}, ジャンプ{enable_jump_protection}, 協調{enable_coordination_protection}")
                print(f"🦴 構造制約: {enable_bone_constraints} (反復{bone_constraint_iterations}回)")
                print("-" * 70)
            
            # メイン処理実行
            denoised_result = denoise_pose_keypoints_kalman(
                pose_keypoint_batch=pose_keypoint,
                config=advanced_config
            )
            
            if enable_verbose_logging:
                print(f"\n✅ Advanced KeyPoint Denoiser 完了")
                print(f"📤 出力フレーム数: {len(denoised_result)}")
                print("=" * 70)
            
            return (denoised_result,)
            
        except Exception as e:
            error_msg = f"Advanced KeyPoint Denoiser エラー: {str(e)}"
            print(f"\n💥 {error_msg}")
            
            if enable_verbose_logging:
                traceback.print_exc()
            
            return (pose_keypoint,)

    @classmethod 
    def IS_CHANGED(cls, **kwargs):
        import random
        return random.random()


# ノード登録用の辞書
NODE_CLASS_MAPPINGS = {
    "ProportionChangerKeypointDenoiser": ProportionChangerKeypointDenoiser,
    "ProportionChangerKeypointDenoiserAdvanced": ProportionChangerKeypointDenoiserAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProportionChangerKeypointDenoiser": "KeyPoint Denoiser",
    "ProportionChangerKeypointDenoiserAdvanced": "KeyPoint Denoiser (Advanced)",
}