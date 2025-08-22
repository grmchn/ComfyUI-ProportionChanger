"""
Core Algorithm Implementation
カルマンフィルタベース7段階パイプラインのメイン実装
"""

import numpy as np
import copy
import time
from typing import List, Dict, Tuple, Optional, Any

try:
    from .config import DenoiserConfig
    from .kalman_filter import KeypointKalmanFilter, initialize_kalman_filters, update_kalman_filters_batch
    from .body_analysis import analyze_body_statistics_enhanced, extract_pose_keypoints_from_frame, calculate_enhanced_orientation_score, get_frame_pose_data
except ImportError:
    # スタンドアロンテスト用
    from config import DenoiserConfig
    from kalman_filter import KeypointKalmanFilter, initialize_kalman_filters, update_kalman_filters_batch
    from body_analysis import analyze_body_statistics_enhanced, extract_pose_keypoints_from_frame, calculate_enhanced_orientation_score, get_frame_pose_data

def log_phase_start(phase_num: int, phase_name: str, verbose: bool = True):
    """段階開始ログ"""
    if verbose:
        print(f"\n🔥 [Phase {phase_num}] {phase_name} 開始...")

def log_phase_end(phase_num: int, phase_name: str, elapsed_ms: float, verbose: bool = True):
    """段階終了ログ"""
    if verbose:
        print(f"✅ [Phase {phase_num}] {phase_name} 完了 ({elapsed_ms:.1f}ms)")

def preprocess_keypoints(pose_keypoint_batch: List, config: DenoiserConfig) -> List:
    """
    Phase 1: 前処理（正規化・信頼度ゲーティング・軽い欠損埋め）
    """
    start_time = time.time()
    log_phase_start(1, "前処理・信頼度ゲーティング", config.verbose_logging)
    
    if config.verbose_logging:
        print(f"  📥 入力バッチサイズ: {len(pose_keypoint_batch)}フレーム")
    
    # 現時点では基本的なコピーのみ（将来的に拡張予定）
    preprocessed = copy.deepcopy(pose_keypoint_batch)
    
    # 信頼度統計
    if config.verbose_logging:
        total_keypoints = 0
        low_confidence_count = 0
        
        for frame_data in preprocessed:
            people_data = get_frame_pose_data(frame_data)
            if people_data:
                keypoints = extract_pose_keypoints_from_frame(people_data)
                for kp in keypoints:
                    total_keypoints += 1
                    if kp[2] < config.conf_min:
                        low_confidence_count += 1
        
        low_conf_rate = low_confidence_count / max(total_keypoints, 1)
        print(f"  📊 低信頼度キーポイント率: {low_conf_rate:.1%} (閾値<{config.conf_min})")
    
    elapsed_ms = (time.time() - start_time) * 1000
    log_phase_end(1, "前処理", elapsed_ms, config.verbose_logging)
    
    return preprocessed

def apply_kalman_filtering(pose_keypoint_batch: List, body_stats: Dict, config: DenoiserConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase 3-4: カルマンフィルタ + カイ二乗ゲーティング
    """
    start_time = time.time()
    log_phase_start(3, "カルマンフィルタ処理・外れ値ゲーティング", config.verbose_logging)
    
    T = len(pose_keypoint_batch)
    J = 25  # DWPose body keypoints
    
    # 結果保存用配列
    filtered_positions = np.zeros((T, J, 2), dtype=np.float32)
    filtered_confidences = np.zeros((T, J), dtype=np.float32)
    gate_results = np.zeros((T, J), dtype=bool)  # ゲート通過結果
    
    # カルマンフィルタ初期化
    if config.verbose_logging:
        print("  🔮 カルマンフィルタ初期化...")
    
    first_frame_people = get_frame_pose_data(pose_keypoint_batch[0])
    first_valid_frame = extract_pose_keypoints_from_frame(first_frame_people)
    kalman_filters = initialize_kalman_filters(first_valid_frame, body_stats['primary_scale'], config)
    
    if config.verbose_logging:
        print(f"  🔮 初期化完了: {len(kalman_filters)}個のフィルタ")
    
    # 進捗表示用
    progress_step = max(1, T // 10)
    total_observations = 0
    total_accepted = 0
    
    # フレームループ処理
    for frame_idx in range(T):
        current_frame_data = get_frame_pose_data(pose_keypoint_batch[frame_idx])
        current_keypoints = extract_pose_keypoints_from_frame(current_frame_data)
        
        # 現在フレームの正面度計算
        orientation_score = calculate_enhanced_orientation_score(current_keypoints)
        
        # ダンス動作パターン検出（Phase 4で本格実装予定）
        protection_factors = {}  # 現在は空辞書
        
        # カルマンフィルタバッチ更新
        positions, acceptances, mahalanobis_distances = update_kalman_filters_batch(
            kalman_filters, current_keypoints, orientation_score, protection_factors
        )
        
        # 結果を配列に保存
        for kp_idx in range(J):
            if kp_idx in positions:
                filtered_positions[frame_idx, kp_idx] = positions[kp_idx]
                filtered_confidences[frame_idx, kp_idx] = current_keypoints[kp_idx][2] if acceptances[kp_idx] else current_keypoints[kp_idx][2] * 0.7
                gate_results[frame_idx, kp_idx] = acceptances[kp_idx]
                
                total_observations += 1
                if acceptances[kp_idx]:
                    total_accepted += 1
            else:
                # フィルタ未初期化：低信頼度として記録
                filtered_positions[frame_idx, kp_idx] = current_keypoints[kp_idx][:2]
                filtered_confidences[frame_idx, kp_idx] = 0.1
                gate_results[frame_idx, kp_idx] = False
        
        # 進捗表示
        if config.verbose_logging and frame_idx % progress_step == 0:
            progress = (frame_idx + 1) / T
            print(f"  ⚡ フレーム処理進捗: {progress:.0%} ({frame_idx+1}/{T})")
    
    # 統計情報
    acceptance_rate = total_accepted / max(total_observations, 1)
    
    if config.verbose_logging:
        print(f"  📊 観測受け入れ率: {acceptance_rate:.1%}")
        print(f"  📊 総観測数: {total_observations}, 受け入れ数: {total_accepted}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    log_phase_end(3, "カルマンフィルタ処理", elapsed_ms, config.verbose_logging)
    
    return filtered_positions, filtered_confidences, gate_results

def apply_structural_constraints(filtered_positions: np.ndarray, 
                                filtered_confidences: np.ndarray,
                                body_stats: Dict, 
                                config: DenoiserConfig) -> np.ndarray:
    """
    Phase 5: 構造投影（骨長制約の軽量実装）
    """
    start_time = time.time()
    log_phase_start(5, "構造投影・骨長制約", config.verbose_logging)
    
    if not config.enable_bone_constraints:
        if config.verbose_logging:
            print("  ⏭️ 構造投影無効化: スキップ")
        log_phase_end(5, "構造投影（スキップ）", 0, config.verbose_logging)
        return filtered_positions.copy()
    
    T, J = filtered_positions.shape[:2]
    adjusted_positions = filtered_positions.copy()
    
    # 簡単な統計
    total_adjustments = 0
    frames_processed = 0
    
    if config.verbose_logging:
        print(f"  🦴 投影間隔: {config.projection_interval}フレームごと")
        print(f"  🦴 最大反復数: {config.max_iterations}")
    
    # 投影間隔に従って処理
    for frame_idx in range(0, T, config.projection_interval):
        # 簡単な骨長チェック（実装は簡略化）
        frame_keypoints = adjusted_positions[frame_idx]
        
        # 主要な骨長をチェック
        adjustments_made = 0
        
        # 上腕長のチェック（右腕）
        if (filtered_confidences[frame_idx, 2] > 0.3 and 
            filtered_confidences[frame_idx, 3] > 0.3):
            
            shoulder_pos = frame_keypoints[2]  # 右肩
            elbow_pos = frame_keypoints[3]    # 右肘
            current_length = np.linalg.norm(elbow_pos - shoulder_pos)
            
            # 統計的基準と比較（簡略化）
            if 'upper_arm_r' in body_stats['joint_references']:
                ref_length = body_stats['joint_references']['upper_arm_r']['median']
                
                # 許容範囲チェック
                min_length = ref_length * 0.6
                max_length = ref_length * 1.4
                
                if current_length < min_length or current_length > max_length:
                    # 軽い調整
                    target_length = np.clip(current_length, min_length, max_length)
                    direction = (elbow_pos - shoulder_pos) / (current_length + 1e-8)
                    
                    adjusted_elbow = shoulder_pos + direction * target_length
                    adjusted_positions[frame_idx, 3] = adjusted_elbow
                    
                    adjustments_made += 1
                    total_adjustments += 1
        
        frames_processed += 1
        
        # 進捗表示
        if config.verbose_logging and frame_idx % (config.projection_interval * 10) == 0:
            progress = frame_idx / T
            print(f"  🦴 投影進捗: {progress:.0%}")
    
    if config.verbose_logging:
        avg_adjustments = total_adjustments / max(frames_processed, 1)
        print(f"  📊 処理フレーム数: {frames_processed}")
        print(f"  📊 総調整回数: {total_adjustments}")
        print(f"  📊 平均調整数/フレーム: {avg_adjustments:.2f}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    log_phase_end(5, "構造投影", elapsed_ms, config.verbose_logging)
    
    return adjusted_positions

def apply_post_smoothing(filtered_positions: np.ndarray,
                        filtered_confidences: np.ndarray,
                        config: DenoiserConfig) -> np.ndarray:
    """
    Phase 6: 後処理スムージング
    """
    start_time = time.time()
    log_phase_start(6, "後処理スムージング", config.verbose_logging)
    
    T, J = filtered_positions.shape[:2]
    smoothed_positions = filtered_positions.copy()
    
    if config.use_rts_smoother:
        if config.verbose_logging:
            print("  🎢 RTSスムーザ適用（簡略実装）...")
        # RTSスムーザの簡略実装（将来的に拡張）
        # 現在は軽いスムージングで代用
        
    else:
        if config.verbose_logging:
            print("  🎢 軽量スムージング適用...")
    
    # 末端キーポイント（顔・手首・足首）に軽いスムージング
    try:
        from .config import FACE_KEYPOINT_INDICES, EXTREMITY_INDICES
    except ImportError:
        from config import FACE_KEYPOINT_INDICES, EXTREMITY_INDICES
    target_indices = FACE_KEYPOINT_INDICES + EXTREMITY_INDICES
    
    half_window = config.light_window_size // 2
    smoothing_applied = 0
    
    for kp_idx in target_indices:
        for t in range(half_window, T - half_window):
            if filtered_confidences[t, kp_idx] > 0.3:  # 信頼度チェック
                # 近傍ウィンドウの重み付き平均
                window_positions = filtered_positions[t-half_window:t+half_window+1, kp_idx]
                window_confs = filtered_confidences[t-half_window:t+half_window+1, kp_idx]
                
                # 信頼度重み付き平均
                valid_mask = window_confs > 0.3
                if np.sum(valid_mask) >= 3:  # 最小有効点数
                    weights = window_confs[valid_mask]
                    smoothed_positions[t, kp_idx] = np.average(window_positions[valid_mask], 
                                                             axis=0, weights=weights)
                    smoothing_applied += 1
    
    if config.verbose_logging:
        print(f"  📊 スムージング適用箇所: {smoothing_applied}")
        print(f"  📊 対象キーポイント: {len(target_indices)}個")
    
    elapsed_ms = (time.time() - start_time) * 1000
    log_phase_end(6, "後処理スムージング", elapsed_ms, config.verbose_logging)
    
    return smoothed_positions

def apply_gap_interpolation(smoothed_positions: np.ndarray,
                           filtered_confidences: np.ndarray,
                           config: DenoiserConfig) -> np.ndarray:
    """
    Phase 7: ギャップ補間
    """
    start_time = time.time()
    log_phase_start(7, "ギャップ補間・最終仕上げ", config.verbose_logging)
    
    T, J = smoothed_positions.shape[:2]
    final_positions = smoothed_positions.copy()
    
    gaps_found = 0
    gaps_interpolated = 0
    
    for kp_idx in range(J):
        # 低信頼度区間を特定
        missing_mask = filtered_confidences[:, kp_idx] < config.conf_min
        
        if not np.any(missing_mask):
            continue  # ギャップなし
        
        # 連続欠損区間を特定
        gaps = identify_gaps(missing_mask)
        gaps_found += len(gaps)
        
        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start + 1
            
            if gap_length <= config.short_gap_threshold:
                # 短ギャップ：線形補間
                interpolated = linear_interpolate_gap(
                    smoothed_positions, kp_idx, gap_start, gap_end
                )
                final_positions[gap_start:gap_end+1, kp_idx] = interpolated
                gaps_interpolated += 1
                
            elif gap_length <= config.medium_gap_threshold:
                # 中ギャップ：スプライン補間（簡略実装）
                interpolated = linear_interpolate_gap(  # 現在は線形で代用
                    smoothed_positions, kp_idx, gap_start, gap_end
                )
                final_positions[gap_start:gap_end+1, kp_idx] = interpolated
                gaps_interpolated += 1
                
            # 長ギャップは現在の値を保持（補間しない）
    
    if config.verbose_logging:
        print(f"  📊 検出ギャップ数: {gaps_found}")
        print(f"  📊 補間済みギャップ数: {gaps_interpolated}")
    
    elapsed_ms = (time.time() - start_time) * 1000
    log_phase_end(7, "ギャップ補間", elapsed_ms, config.verbose_logging)
    
    return final_positions

def identify_gaps(missing_mask: np.ndarray) -> List[Tuple[int, int]]:
    """連続する欠損区間を特定"""
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, is_missing in enumerate(missing_mask):
        if is_missing and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_missing and in_gap:
            gaps.append((gap_start, i-1))
            in_gap = False
    
    # 最後までギャップが続く場合
    if in_gap:
        gaps.append((gap_start, len(missing_mask)-1))
    
    return gaps

def linear_interpolate_gap(positions: np.ndarray, kp_idx: int, start: int, end: int) -> np.ndarray:
    """線形補間"""
    T = positions.shape[0]
    
    # 前後の有効な点を取得
    before_pos = positions[start-1, kp_idx] if start > 0 else positions[end+1, kp_idx]
    after_pos = positions[end+1, kp_idx] if end < T-1 else positions[start-1, kp_idx]
    
    # 線形補間
    gap_length = end - start + 1
    interpolated = np.zeros((gap_length, 2))
    
    for i in range(gap_length):
        t = (i + 1) / (gap_length + 1)  # 0 < t < 1
        interpolated[i] = (1 - t) * before_pos + t * after_pos
    
    return interpolated

def convert_to_pose_keypoint_batch(positions: np.ndarray, 
                                 confidences: np.ndarray, 
                                 original_batch: List) -> List:
    """NumPy配列からPOSE_KEYPOINT形式に変換"""
    result_batch = []
    
    for frame_idx in range(len(positions)):
        frame_data = copy.deepcopy(original_batch[frame_idx])
        people_data = get_frame_pose_data(frame_data)
        
        # pose_keypoints_2dを更新
        new_pose_keypoints = []
        for kp_idx in range(25):
            x, y = positions[frame_idx, kp_idx]
            conf = confidences[frame_idx, kp_idx]
            new_pose_keypoints.extend([float(x), float(y), float(conf)])
        
        people_data['pose_keypoints_2d'] = new_pose_keypoints
        result_batch.append(frame_data)
    
    return result_batch

def denoise_pose_keypoints_kalman(pose_keypoint_batch: List,
                                 config: Optional[DenoiserConfig] = None) -> List:
    """
    カルマンフィルタベースKeyPointデノイザー（7段階パイプライン完全版）
    
    Args:
        pose_keypoint_batch: POSE_KEYPOINT形式のバッチ
        config: デノイザー設定（Noneの場合はデフォルト設定）
    
    Returns:
        List: ノイズ除去済みPOSE_KEYPOINTバッチ
    """
    
    if config is None:
        config = DenoiserConfig()
    
    if len(pose_keypoint_batch) < 3:
        if config.verbose_logging:
            print("⚠️ バッチサイズ不足（<3フレーム）: 処理スキップ")
        return pose_keypoint_batch
    
    total_start_time = time.time()
    
    if config.verbose_logging:
        print(f"\n{'='*80}")
        print(f"🚀 KeyPoint Denoiser 実行開始")
        print(f"{'='*80}")
        print(f"📥 入力: {len(pose_keypoint_batch)}フレーム")
        print(f"⚙️ 設定: 詳細ログ{'ON' if config.verbose_logging else 'OFF'}, 骨長制約{'ON' if config.enable_bone_constraints else 'OFF'}")
    
    try:
        # Phase 1: 前処理
        preprocessed_batch = preprocess_keypoints(pose_keypoint_batch, config)
        
        # Phase 2: 体スケール・向き推定（analyze_body_statistics_enhanced内でPhase 1-2のログ出力）
        body_stats = analyze_body_statistics_enhanced(preprocessed_batch, config)
        
        # Phase 3-4: カルマンフィルタ + ゲーティング
        filtered_positions, filtered_confidences, gate_results = apply_kalman_filtering(
            preprocessed_batch, body_stats, config
        )
        
        # Phase 5: 構造投影
        adjusted_positions = apply_structural_constraints(
            filtered_positions, filtered_confidences, body_stats, config
        )
        
        # Phase 6: 後処理スムージング
        smoothed_positions = apply_post_smoothing(
            adjusted_positions, filtered_confidences, config
        )
        
        # Phase 7: ギャップ補間
        final_positions = apply_gap_interpolation(
            smoothed_positions, filtered_confidences, config
        )
        
        # 最終変換
        result_batch = convert_to_pose_keypoint_batch(
            final_positions, filtered_confidences, preprocessed_batch
        )
        
        # 統計サマリー
        total_elapsed_ms = (time.time() - total_start_time) * 1000
        total_keypoints = len(pose_keypoint_batch) * 25
        gated_keypoints = np.sum(~gate_results) if gate_results.size > 0 else 0
        rejection_rate = gated_keypoints / max(total_keypoints, 1)
        
        if config.verbose_logging:
            print(f"\n{'='*80}")
            print(f"✨ KeyPoint Denoiser 処理完了!")
            print(f"{'='*80}")
            print(f"⏱️  総処理時間: {total_elapsed_ms:.1f}ms")
            print(f"📊 ゲート棄却率: {rejection_rate:.1%}")
            print(f"📊 品質向上推定: {(1-rejection_rate)*body_stats['quality_score']:.1%}")
            print(f"🎯 体スケール: {body_stats['primary_scale']:.4f}")
            print(f"🎯 正面向き比率: {body_stats['orientation_stats']['front_facing_ratio']:.2%}")
            print(f"📈 データ品質: {body_stats['quality_score']:.3f}")
            print(f"{'='*80}")
        
        return result_batch
        
    except Exception as e:
        if config.verbose_logging:
            print(f"\n💥 処理エラー: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # エラー時は元データを返す
        return pose_keypoint_batch