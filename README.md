# YOLO11 Wild Boar Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![mAP](https://img.shields.io/badge/mAP@0.5-96.72%25-brightgreen.svg)]()

高精度野生猪検出のためのYOLO11転移学習プロジェクト

## 主な特徴

-  **超高精度**: mAP@0.5 96.72% 達成
-  **効率的学習**: わずか327枚の学習データ
-  **リアルタイム**: CPU/GPU対応の高速推論
-  **実用的**: 農業・野生動物管理に直接応用可能

## 性能指標

| 指標 | 値 |
|------|-----|
| mAP@0.5 | 96.72% |
| mAP@0.5:0.95 | 81.76% |
| Precision | 90.7% |
| Recall | 98.5% |
| 学習データ数 | 327枚 |
| クラス数 | 2 (wild_boar, domestic_pig) |

## クイックスタート

### インストール
'''bash
git clone https://github.com/s1f102302179/yolo11-boar-detection.git
cd yolo11-boar-detection
pip install -r requirements.txt
'''Python

## 推論実行
'''bash
python inference.py
'''Python

## フォルダ構成
- best.pt - 学習済みYOLO11モデル (mAP@0.5: 96.72%)
- inference.py - 動画推論スクリプト
- classes.txt - クラス定義 (wild_boar, domestic_pig)
- data.yaml - YOLO設定ファイル
- requirements.txt - 依存関係


