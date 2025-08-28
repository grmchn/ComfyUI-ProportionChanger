# ComfyUI-ProportionChanger

> **注意**: このREADMEは[Claude Code](https://claude.ai/code) AI支援開発ツールを使用して自動生成されました。

**ComfyUI用の高度なDWPoseベース体型プロポーション調整・キーポイント処理システム**

このカスタムノードは、[kijai氏のComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)のWanVideo UniAnimate DWPose Detectorノードを分解してコピーしたものです。違いは、画像で入力していたところをDWPoseのKeyPointで入力できるようになったことで、これにより柔軟で詳細な体型操作ができるようになっています。

体型プロポーションの変換、ポーズキーポイントの操作、高度なプロポーションスケーリングアルゴリズムによる自然で一貫性のあるポーズシーケンスの作成を可能にします。

## 機能

### コアノード
- **ProportionChanger DWPose Detector**: キーポイント出力機能付き基本DWPose検出
- **ProportionChanger Reference**: リファレンスポーズからのプロポーション変換適用
- **ProportionChanger DWPose Render**: ポーズ可視化・レンダリング
- **ProportionChanger Params**: ポーズ調整用インタラクティブパラメータ制御
- **ProportionChanger Interpolator**: 流動的なモーション用フレーム補間
- **pose_keypoint input**: JSONからPOSE_KEYPOINT変換用ユーティリティ
- **pose_keypoint preview**: POSE_KEYPOINTからJSON変換用ユーティリティ

### 主要機能
- **プロポーション調整**: 体の部位（頭、胴体、四肢）を独立してスケーリング
- **リファレンスベーススケーリング**: リファレンスポーズのプロポーションをターゲットシーケンスに適用
- **形式変換**: DWPoseとPOSE_KEYPOINT形式間のシームレス変換
- **バッチ処理**: 複数フレームとポーズシーケンスの効率的処理
- **インタラクティブUI**: ポーズプレビュー付きリアルタイムパラメータ調整

## インストール

1. `custom_nodes`フォルダにこのリポジトリをクローン：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/[your-repo]/ComfyUI-ProportionChanger.git
```

2. 依存関係をインストール：
```bash
cd ComfyUI-ProportionChanger
pip install -r requirements.txt
```

3. ComfyUIを再起動

## 依存関係

### 必要なモデル
以下のモデルをダウンロードして正しいディレクトリに配置する必要があります：

**DWPoseモデル** (`unianimate/models/DWPose/`に配置):
- `dw-ll_ucoco_384_bs5.torchscript.pt` - メインポーズ検出モデル
- `yolox_l.torchscript.pt` - オブジェクト検出モデル

**ダウンロードリンク**:
- [Hugging Face モデルリポジトリ](https://huggingface.co/Kijai/WanVideo_comfy/tree/main)

### Python依存関係
完全なリストは`requirements.txt`を参照。主な依存関係：
- `torch`
- `torchvision`
- `opencv-python`
- `numpy`

## 使用方法

### 基本ワークフロー

1. **画像読み込み**: ソース画像を入力
2. **DWPose検出**: `ProportionChanger DWPose Detector`でポーズキーポイントを抽出
3. **プロポーション適用**: `ProportionChanger Reference`でリファレンスポーズを使用した変換
4. **結果レンダリング**: `ProportionChanger DWPose Render`で最終ポーズを可視化

### 高度な機能

#### プロポーションリファレンスシステム
- 理想的なプロポーションのリファレンスポーズを読み込み
- そのプロポーションをターゲット画像に自動適用
- 自動スケーリング機能で異なるキャンバスサイズに対応

#### インタラクティブパラメータ制御
- `ProportionChanger Params`でリアルタイム調整
- フルシーケンス適用前の変更プレビュー
- 個別体部位スケーリングの微調整

## ノードリファレンス

### ProportionChanger DWPose Detector
**用途**: キーポイント出力機能付き基本ポーズ検出
**入力**: `image`, `score_threshold`
**出力**: `pose_keypoint`

### ProportionChanger Reference
**用途**: リファレンスプロポーションのターゲットポーズへの適用
**入力**: `pose_keypoint`, `reference_pose_keypoint`
**出力**: `changed_pose_keypoint`

### ProportionChanger Params
**用途**: ポーズ操作用インタラクティブパラメータ調整
**入力**: `pose_keypoint`, 各種スケーリングパラメータ
**出力**: `adjusted_pose_keypoint`

## 技術実装

### コアアルゴリズム
- **DWPose検出**: WanVideo UniAnimate実装からの改良版
- **プロポーションスケーリング**: キーポイント座標の数学的変換
- **キャンバススケーリング**: 異なる画像サイズに対する自動調整

### データ形式
- **入力**: 標準画像形式（PNG、JPG等）
- **内部**: 正規化座標（0-1範囲）でのPOSE_KEYPOINT形式
- **メタデータ**: 正確な座標変換のためのキャンバス寸法

## 帰属とクレジット

このプロジェクトは、オープンソースコミュニティの素晴らしい成果に基づいて構築されています：

### 主要ソース
- **[kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)**: DWPose検出実装の主要基盤
- **[toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor)**: ProportionChanger Paramsノード機能のソース

### 開発手法
- **Claude Code**: AI支援プログラミングによる全開発
- **Vibe Coding**: AIガイダンス付き反復開発
- **Issue駆動開発**: 機能実装とバグ修正への体系的アプローチ

### 特別な感謝
- **kijai氏**: 堅牢なWanVideoWrapper基盤とDWPose実装
- **toyxyz氏**: 革新的なopenpose編集フレームワーク
- **Anthropic**: Claude Code開発ツール
- **オープンソースコミュニティ**: 継続的なインスピレーションとコラボレーション

## ライセンス

このプロジェクトは、異なるライセンスのソース素材を組み合わせているため、**GPL 3.0**でライセンスされています：

- **主要ソース**: [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache 2.0)
- **二次ソース**: [toyxyz/ComfyUI-ultimate-openpose-editor](https://github.com/toyxyz/ComfyUI-ultimate-openpose-editor) (GPL 3.0)

Apache 2.0とGPL 3.0ライセンスのプロジェクトからコードを組み合わせる場合、ライセンス互換性ルールに従って結果の派生作品はGPL 3.0で配布する必要があります。

**著作権表示:**
- オリジナルWanVideo UniAnimate DWPose Detector: Copyright by kijai
- ProportionChanger Params機能: Copyright by toyxyz  
- 修正・統合: このプロジェクトの貢献者

完全なGPL 3.0ライセンス条項については[LICENSE](LICENSE)ファイルを参照してください。

## 開発状況

このプロジェクトはClaude AIの支援により活発に開発されています。問題と機能要求はAI駆動の開発サイクルを通じて管理されています。

### 最近の更新
- リファレンスベースプロポーションスケーリングシステム
- バッチ処理機能
- インタラクティブパラメータ制御
- キャンバスサイズ互換性の改善

## コントリビューション

このプロジェクトは主にClaude Code手法で開発されていますが、コミュニティフィードバックと問題報告を歓迎します。開発はAI支援ワークフローに従うことにご注意ください。

## トラブルシューティング

### よくある問題
1. **モデル読み込みエラー**: DWPoseモデルが正しいディレクトリにあることを確認
2. **キャンバスサイズ不一致**: canvas_width/canvas_heightメタデータが存在することを確認
3. **座標範囲問題**: ポーズデータが正規化座標（0-1）を使用していることを確認

### デバッグ情報
詳細な処理情報については、`utils.py`でデバッグログを有効化してください。

## サンプルワークフロー

様々な機能と使用例を実証するサンプルComfyUIワークフローについては、`example_workflows/`ディレクトリを確認してください。

---

**注意**: これはポーズ操作とプロポーション調整のための特殊ツールです。一般的な動画生成には、オリジナルのWanVideoWrapperまたはネイティブComfyUI実装の使用を検討してください。

## 開発コンテキスト

### AI支援開発について
このプロジェクトは**Vibe Coding**手法により、Claude Codeを用いて全面的に開発されました：

- **コード品質**: AIによる一貫したコーディングスタイルと最適化
- **問題解決**: Claude AIによる体系的なissue管理とPR処理
- **技術革新**: 最新のAI開発手法による効率的な実装

### 分解・適応プロセス
- **WanVideoWrapper**: UniAnimate DWPose Detectorノードを分解・特殊化
- **Ultimate OpenPose Editor**: Paramsノードの機能を移植・統合
- **カスタム拡張**: KeyPoint Denoiser等の独自機能追加

この開発手法により、オリジナルの品質を維持しながら、特化した機能セットを効率的に実現しています。