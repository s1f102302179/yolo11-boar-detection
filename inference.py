from ultralytics import YOLO
import cv2
import os

def simple_boar_detection():
    """シンプル猪検出"""
    
    # 正しいモデルパス
    model_path = r"consolidated_dataset\boar_detection\yolo11_boar_v1\weights\best.pt"
    
    print("=== 猪検出テスト ===")
    print(f"📂 モデルパス: {model_path}")
    
    # モデル存在確認
    if not os.path.exists(model_path):
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return
    
    # モデル読み込み
    print("🔄 モデル読み込み中...")
    model = YOLO(model_path)
    print("✅ モデル読み込み完了")
    
    # 動画パス
    video_path = r"C:\Users\amon\Videos\Captures\pig_test.mp4"
    
    # 動画存在確認
    if not os.path.exists(video_path):
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return
    
    print(f"🎬 動画: {video_path}")
    
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    
    # 動画情報
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📐 動画サイズ: {width} x {height}")
    print(f"⏱️ FPS: {fps}, 総フレーム: {total_frames}")
    print("🎯 信頼度70%以上のみ表示")
    print("📍 座標系: (0,0)-(1,1) 正規化座標")
    print("\n推論開始... (ESCで終了)")
    
    frame_count = 0
    detection_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 20フレームに1回処理（軽量化）
        if frame_count % 20 == 0:
            
            # YOLO推論
            results = model(frame, verbose=False)
            
            # 結果処理
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                
                frame_detections = []
                
                for box in boxes:
                    confidence = float(box.conf[0])
                    
                    # 信頼度70%以上のみ
                    if confidence >= 0.7:
                        
                        # 座標とクラス情報
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # 中央座標（正規化）
                        center_x = ((x1 + x2) / 2) / width
                        center_y = ((y1 + y2) / 2) / height
                        
                        frame_detections.append({
                            'confidence': confidence,
                            'class_name': class_name,
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })
                
                # 検出結果表示
                if frame_detections:
                    detection_count += 1
                    total_detections += len(frame_detections)
                    current_time = frame_count / fps
                    
                    print(f"\n📍 フレーム {frame_count} ({current_time:.1f}秒):")
                    for i, det in enumerate(frame_detections):
                        print(f"  猪 {i+1}: 座標=({det['center_x']:.3f}, {det['center_y']:.3f}), "
                              f"信頼度={det['confidence']:.3f}, クラス={det['class_name']}")
                    
                    # 画像に結果描画
                    annotated_frame = frame.copy()
                    for det in frame_detections:
                        x1, y1, x2, y2 = det['bbox']
                        
                        # バウンディングボックス
                        color = (0, 255, 0) if det['class_name'] == 'wild_boar' else (255, 0, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # 中央点
                        center_x_pixel = int(det['center_x'] * width)
                        center_y_pixel = int(det['center_y'] * height)
                        cv2.circle(annotated_frame, (center_x_pixel, center_y_pixel), 5, (0, 0, 255), -1)
                        
                        # ラベル
                        label = f"{det['class_name']}: {det['confidence']:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # 座標表示
                        coord_text = f"({det['center_x']:.3f}, {det['center_y']:.3f})"
                        cv2.putText(annotated_frame, coord_text, (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    cv2.imshow('Boar Detection', annotated_frame)
                else:
                    cv2.imshow('Boar Detection', frame)
            else:
                cv2.imshow('Boar Detection', frame)
            
            # 進捗表示
            if frame_count % 600 == 0:  # 30秒ごと
                progress = (frame_count / total_frames) * 100
                print(f"📊 進捗: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # キー入力チェック
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n⚠️ ユーザーによる中断")
            break
        elif key == ord(' '):  # スペース
            cv2.waitKey(0)  # 一時停止
    
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    
    # 最終統計
    print(f"\n✅ 処理完了")
    print(f"📊 最終結果:")
    print(f"  総フレーム数: {frame_count}")
    print(f"  処理フレーム数: {frame_count // 20}")
    print(f"  検出ありフレーム: {detection_count}")
    print(f"  総検出数: {total_detections}")
    
    if detection_count > 0:
        detection_rate = (detection_count / (frame_count // 20)) * 100
        avg_detections = total_detections / detection_count
        print(f"  検出率: {detection_rate:.1f}%")
        print(f"  平均検出数/フレーム: {avg_detections:.1f}")

if __name__ == "__main__":
    simple_boar_detection()