"""
Body Analysis and Orientation Detection
体統計分析・正面度検出モジュール
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from .config import DenoiserConfig
except ImportError:
    # スタンドアロンテスト用
    from config import DenoiserConfig

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """2点間の距離を計算"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_pose_keypoints_from_frame(frame_people_data: Dict) -> List[List[float]]:
    """
    フレームデータからpose_keypointsを抽出
    [x1, y1, c1, x2, y2, c2, ...] -> [[x1, y1, c1], [x2, y2, c2], ...]
    """
    pose_keypoints = frame_people_data.get('pose_keypoints_2d', [])
    
    keypoints = []
    for i in range(0, len(pose_keypoints), 3):
        if i + 2 < len(pose_keypoints):
            keypoints.append([
                pose_keypoints[i],      # x
                pose_keypoints[i + 1],  # y  
                pose_keypoints[i + 2]   # confidence
            ])
    
    # 25個のキーポイントに満たない場合は0で埋める
    while len(keypoints) < 25:
        keypoints.append([0.0, 0.0, 0.0])
    
    return keypoints[:25]  # 25個に制限

def calculate_enhanced_orientation_score(pose_data: List[List[float]], verbose: bool = False) -> float:
    """
    改良された正面度計算：肩幅・腰幅正規化 + 顔補強
    0.0=側面, 1.0=正面
    """
    if len(pose_data) < 25:
        if verbose:
            print("      ⚠️ キーポイント数不足（<25）")
        return 0.5
    
    # 肩幅・腰幅取得
    shoulder_w = hip_w = 0
    confidence_threshold = 0.3
    
    if (pose_data[2][2] > confidence_threshold and pose_data[5][2] > confidence_threshold):  # 両肩見える
        shoulder_w = distance(pose_data[2][:2], pose_data[5][:2])
    if (pose_data[8][2] > confidence_threshold and pose_data[11][2] > confidence_threshold):  # 両腰見える
        hip_w = distance(pose_data[8][:2], pose_data[11][:2])
    
    # 仮の正面基準（統計的基準で後で置き換え予定）
    w_sh_est = 0.12  # 仮の正面時肩幅
    w_hp_est = 0.10  # 仮の正面時腰幅
    
    # 基本正面度（肩幅・腰幅の正規化平均）
    ori_components = []
    if shoulder_w > 0.02:  # 最小閾値
        ori_components.append(min(shoulder_w / w_sh_est, 1.0))
    if hip_w > 0.02:
        ori_components.append(min(hip_w / w_hp_est, 1.0))
    
    if ori_components:
        base_orientation = np.mean(ori_components)
    else:
        base_orientation = 0.5
    
    # 顔補強（目・口の横幅/縦幅 で顔yaw代理）
    face_bonus = 0
    if (pose_data[14][2] > confidence_threshold and pose_data[15][2] > confidence_threshold):  # 両目
        eye_distance = distance(pose_data[14][:2], pose_data[15][:2])
        if eye_distance > 0.015:  # 正規化座標で適度な目間距離
            face_bonus += 0.1
    
    # 最終正面度スコア
    final_score = np.clip(base_orientation + face_bonus, 0, 1)
    
    if verbose:
        print(f"      📐 肩幅={shoulder_w:.4f}, 腰幅={hip_w:.4f}")
        print(f"      📐 基本正面度={base_orientation:.3f}, 顔補強={face_bonus:.3f}")
        print(f"      📐 最終正面度スコア={final_score:.3f}")
    
    return final_score

def calculate_joint_distances_enhanced(pose_data: List[List[float]], verbose: bool = False) -> Dict[str, float]:
    """
    各関節間距離を計算（信頼度チェック付き）
    """
    distances = {}
    
    # 信頼度閾値
    confidence_threshold = 0.3
    
    # 肩幅（正面時のみ有効）
    if (pose_data[2][2] > confidence_threshold and pose_data[5][2] > confidence_threshold):
        shoulder_width = distance(pose_data[2][:2], pose_data[5][:2])
        if shoulder_width > 0.02:  # 最小閾値（側面時は除外）
            distances['shoulder_width'] = shoulder_width
    
    # 腰幅
    if (pose_data[8][2] > confidence_threshold and pose_data[11][2] > confidence_threshold):
        hip_width = distance(pose_data[8][:2], pose_data[11][:2])
        if hip_width > 0.02:
            distances['hip_width'] = hip_width
    
    # 胴体長（首-腰中点）
    if (pose_data[1][2] > confidence_threshold and 
        pose_data[8][2] > confidence_threshold and pose_data[11][2] > confidence_threshold):
        hip_center = ((pose_data[8][0] + pose_data[11][0]) / 2, 
                      (pose_data[8][1] + pose_data[11][1]) / 2)
        distances['torso_length'] = distance(pose_data[1][:2], hip_center)
    
    # 頭部サイズ（鼻-首）
    if (pose_data[0][2] > confidence_threshold and pose_data[1][2] > confidence_threshold):
        distances['head_size'] = distance(pose_data[0][:2], pose_data[1][:2])
    
    # 上腕・前腕・太腿・下腿（左右）
    joint_pairs = {
        'upper_arm_r': (2, 3),   # 右肩-右肘
        'upper_arm_l': (5, 6),   # 左肩-左肘
        'forearm_r': (3, 4),     # 右肘-右手首
        'forearm_l': (6, 7),     # 左肘-左手首
        'thigh_r': (11, 12),     # 右腰-右膝
        'thigh_l': (8, 9),       # 左腰-左膝
        'calf_r': (12, 13),      # 右膝-右足首
        'calf_l': (9, 10),       # 左膝-左足首
    }
    
    for joint_name, (idx1, idx2) in joint_pairs.items():
        if (pose_data[idx1][2] > confidence_threshold and 
            pose_data[idx2][2] > confidence_threshold):
            distances[joint_name] = distance(pose_data[idx1][:2], pose_data[idx2][:2])
    
    if verbose:
        print(f"      📏 検出関節数: {len(distances)}")
        for joint_name, dist in distances.items():
            print(f"      📏   {joint_name}: {dist:.4f}")
    
    return distances

def analyze_body_statistics_enhanced(pose_keypoint_batch: List, config: DenoiserConfig) -> Dict:
    """
    改良版：正面度フィルタリング＋頑健統計による体スケール・骨長基準計算
    重要改善点：「正面向きフレーム集合」で骨長基準を学習し、側面バイアスを回避
    """
    
    if config.verbose_logging:
        print("  🔍 [Phase 1-2] 体統計分析開始...")
        print(f"    📊 総フレーム数: {len(pose_keypoint_batch)}")
    
    # 各部位の距離統計を収集（正面フレーム優先）
    joint_distance_stats = {
        'shoulder_width': [],      # 肩幅（正面時のみ有効）
        'hip_width': [],           # 腰幅（正面度判定補強用）
        'torso_length': [],        # 胴体長（首-腰中点）
        'upper_arm_r': [], 'upper_arm_l': [],   # 上腕（肩-肘）
        'forearm_r': [], 'forearm_l': [],       # 前腕（肘-手首）
        'thigh_r': [], 'thigh_l': [],           # 太腿（腰-膝）
        'calf_r': [], 'calf_l': [],             # 下腿（膝-足首）
        'head_size': [],           # 頭部サイズ（鼻-首）
    }
    
    # 体の向き分析（正面度判定用）
    orientation_scores = []
    front_facing_frames = []  # 高正面度フレームのインデックス
    
    if config.verbose_logging:
        print("    📐 Phase 1-2a: 全フレーム正面度分析...")
    
    # Phase 1: 全フレームで正面度スコア計算
    for frame_idx, frame_data in enumerate(pose_keypoint_batch):
        if not frame_data or not frame_data[0].get('people'):
            continue
            
        pose_data = extract_pose_keypoints_from_frame(frame_data[0]['people'][0])
        
        # 改良された正面度計算（肩幅・腰幅・目間距離・顔yaw代理）
        orientation_score = calculate_enhanced_orientation_score(
            pose_data, verbose=config.verbose_logging and frame_idx < 3  # 最初の3フレームのみ詳細ログ
        )
        orientation_scores.append(orientation_score)
        
        # 高正面度フレーム選定（顔yaw≈0, 目間距離が高位パーセンタイル、conf高）
        if orientation_score > config.orientation_threshold:
            front_facing_frames.append(frame_idx)
    
    if config.verbose_logging:
        avg_orientation = np.mean(orientation_scores) if orientation_scores else 0.5
        print(f"    📐 平均正面度: {avg_orientation:.3f}")
        print(f"    📐 高正面度フレーム: {len(front_facing_frames)}個 (閾値>{config.orientation_threshold})")
    
    if config.verbose_logging:
        print("    📏 Phase 1-2b: 骨長統計データ収集...")
    
    # Phase 2: 正面フレーム群で骨長基準学習
    for frame_idx, frame_data in enumerate(pose_keypoint_batch):
        if not frame_data or not frame_data[0].get('people'):
            continue
            
        pose_data = extract_pose_keypoints_from_frame(frame_data[0]['people'][0])
        distances = calculate_joint_distances_enhanced(
            pose_data, verbose=config.verbose_logging and frame_idx == 0  # 最初のフレームのみ詳細ログ
        )
        
        # 正面フレームの場合：全関節データを収集
        if frame_idx in front_facing_frames:
            for joint_name, distance_value in distances.items():
                if distance_value > 0 and joint_name in joint_distance_stats:
                    joint_distance_stats[joint_name].append(distance_value)
        else:
            # 側面フレームの場合：向きに依存しない関節のみ収集
            orientation_independent = ['torso_length', 'head_size']
            for joint_name in orientation_independent:
                if joint_name in distances and distances[joint_name] > 0:
                    joint_distance_stats[joint_name].append(distances[joint_name])
    
    if config.verbose_logging:
        print("    🧮 Phase 1-2c: 頑健統計処理（Tukeyフェンス）...")
    
    # Phase 3: 頑健統計による基準値計算（Tukeyのフェンス or MAD使用）
    body_references = {}
    
    for joint_name, distances in joint_distance_stats.items():
        if len(distances) >= config.min_samples_for_stats:
            distances_array = np.array(distances)
            
            # Tukeyのフェンスによる外れ値除去（推奨手法）
            Q1, Q3 = np.percentile(distances_array, [25, 75])
            IQR = Q3 - Q1
            lower_fence = Q1 - config.outlier_fence_multiplier * IQR
            upper_fence = Q3 + config.outlier_fence_multiplier * IQR
            
            # フェンス内データのみ使用
            clean_distances = distances_array[
                (distances_array >= lower_fence) & (distances_array <= upper_fence)
            ]
            
            if len(clean_distances) > 0:
                body_references[joint_name] = {
                    'median': np.median(clean_distances),
                    'q25': np.percentile(clean_distances, 25),
                    'q75': np.percentile(clean_distances, 75),
                    'mad': np.median(np.abs(clean_distances - np.median(clean_distances))),  # MAD追加
                    'sample_count': len(clean_distances),
                    'outliers_removed': len(distances) - len(clean_distances)
                }
                
                if config.verbose_logging:
                    ref = body_references[joint_name]
                    print(f"      📊 {joint_name}: median={ref['median']:.4f}, samples={ref['sample_count']}, outliers={ref['outliers_removed']}")
    
    if config.verbose_logging:
        print("    💪 Phase 1-2d: 体スケール決定...")
    
    # Phase 4: 体スケール決定（複数部位からミックス推定）
    primary_scale_candidates = []
    
    # 胴体・肩幅・腰幅・上腕の中央値ミックスで推定
    scale_sources = {
        'torso_length': 1.0,        # そのまま使用
        'shoulder_width': 2.5,      # 体幅換算
        'hip_width': 3.0,           # 体幅換算（やや保守的）
        'upper_arm_r': 6.0,         # 上腕長から全身推定
        'upper_arm_l': 6.0,         # 上腕長から全身推定
    }
    
    for joint_name, scale_factor in scale_sources.items():
        if joint_name in body_references:
            estimated_scale = body_references[joint_name]['median'] * scale_factor
            primary_scale_candidates.append(estimated_scale)
            
            if config.verbose_logging:
                print(f"      💪 {joint_name} -> 体スケール推定: {estimated_scale:.4f}")
    
    # 頑健な体スケール決定（中央値使用）
    if primary_scale_candidates:
        primary_body_scale = np.median(primary_scale_candidates)
    else:
        primary_body_scale = config.fallback_body_scale
        if config.verbose_logging:
            print(f"      ⚠️ 体スケール推定失敗、フォールバック値使用: {primary_body_scale}")
    
    # 正面向き比率計算
    front_facing_ratio = len(front_facing_frames) / max(len(pose_keypoint_batch), 1)
    
    result = {
        'primary_scale': primary_body_scale,
        'joint_references': body_references,
        'orientation_stats': {
            'front_facing_ratio': front_facing_ratio,
            'front_frame_count': len(front_facing_frames),
            'avg_orientation': np.mean(orientation_scores) if orientation_scores else 0.5,
            'front_frame_indices': front_facing_frames
        },
        'scale_candidates': primary_scale_candidates,  # デバッグ用
        'quality_score': min(front_facing_ratio * 2.0, 1.0)  # データ品質指標
    }
    
    if config.verbose_logging:
        print(f"  ✅ [Phase 1-2] 体統計分析完了")
        print(f"    💪 最終体スケール: {primary_body_scale:.4f}")
        print(f"    🎯 正面向き比率: {front_facing_ratio:.2%}")
        print(f"    📊 データ品質スコア: {result['quality_score']:.3f}")
        print(f"    🔍 有効関節統計: {len(body_references)}種類")
    
    return result