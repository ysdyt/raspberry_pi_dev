from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from datetime import datetime
import platform

# システム環境に応じてインポートを分岐
IS_RASPBERRY_PI = (
    platform.system() == "Linux"
    and os.path.exists("/sys/firmware/devicetree/base/model")
    and "Raspberry Pi" in open("/sys/firmware/devicetree/base/model").read()
)

if IS_RASPBERRY_PI:
    from picamera2 import Picamera2


def capture_image():
    """
    Raspberry Pi カメラで画像を撮影し、指定したパスに保存します

    Returns:
        撮影した画像のNumPy配列
    """
    if not IS_RASPBERRY_PI:
        # 開発環境では、テスト用の画像を読み込む
        print("開発環境では、テスト用の画像を使用します")
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if os.path.exists(test_image_path):
            return cv2.imread(test_image_path)
        else:
            # テスト画像がなければ黒い画像を生成
            print(f"テスト画像 {test_image_path} が見つかりません。黒い画像を生成します。")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    # Raspberry Pi環境での処理
    # Picamera2の初期化
    picam2 = Picamera2()
    # カメラの設定
    config = picam2.create_still_configuration()
    picam2.configure(config)
    # カメラを起動
    picam2.start()
    # カメラの安定化のために少し待機
    time.sleep(2)
    # 画像を撮影
    img_array = picam2.capture_array()
    # カメラを停止
    picam2.stop()
    # BGR形式に変換（YOLOとOpenCVはBGR形式を使用）
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_array_bgr


def process_image(img_array_bgr, confidence_threshold=0.3):
    """
    画像内の人物を検出し、バウンディングボックスを描画した画像と統計情報を返します

    Args:
        img_array_bgr: 処理する画像のNumPy配列
        confidence_threshold: 検出の信頼度しきい値

    Returns:
        processed_image: バウンディングボックス付きの画像
        stats: 検出統計情報の辞書
    """
    # YOLOv11をロード
    model_path = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
    model = YOLO(model_path)

    # 推論を実行
    results = model(img_array_bgr)

    # 人を検出（クラス0は通常「person」を表す）
    person_count = 0
    confidence_scores = []  # 信頼度スコアを保存するリスト
    img_with_boxes = img_array_bgr.copy()  # 元の画像をコピー

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # クラス0は「person」
            if cls == 0:
                # 信頼度を取得
                conf = float(box.conf[0])

                # 信頼度がしきい値より大きい場合のみカウントして表示
                if conf > confidence_threshold:
                    person_count += 1
                    confidence_scores.append(conf)

                    # バウンディングボックスの座標を取得
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # すべての検出結果を緑色で表示
                    box_color = (0, 255, 0)  # 緑色
                    text_color = (0, 255, 0)  # 緑色テキスト

                    # ボックスを描画
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), box_color, 2)

                    # 信頼度の数値のみを表示
                    conf_text = f"{conf:.2f}"
                    cv2.putText(img_with_boxes, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # 現在の時間を取得して画像に表示
    current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    # 曜日を日本語で取得
    weekday = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][datetime.now().weekday()]
    time_text = f"{current_time} {weekday}"

    # テキストサイズを計算して中央に配置
    text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = (img_with_boxes.shape[1] - text_size[0]) // 2
    text_y = 20  # 上部から20ピクセル下

    # 背景を少し暗くして読みやすくする
    cv2.putText(img_with_boxes, time_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 統計情報を辞書にまとめる
    stats = {
        "person_count": person_count,
        "confidence_scores": confidence_scores,
    }

    if confidence_scores:
        stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
    else:
        stats["avg_confidence"] = 0.0

    return img_with_boxes, stats


# スクリプトとして実行される場合のメイン処理
if __name__ == "__main__":
    confidence_threshold = 0.3

    try:
        # カメラで撮影
        captured_img = capture_image()
        # 画像処理
        processed_image, stats = process_image(captured_img, confidence_threshold)

        # ./processed_img以下に結果画像を保存
        output_dir = "./processed_img"
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_img_path = os.path.join(output_dir, f"processed_{timestamp}.jpg")

        cv2.imwrite(output_img_path, processed_image)

        # 結果出力
        print(f"信頼度{confidence_threshold * 100}%超の人数: {stats['person_count']}人")

        # 統計情報
        if stats["confidence_scores"]:
            avg_conf = stats["avg_confidence"]
            print(f"平均信頼度: {avg_conf:.2f} ({avg_conf * 100:.1f}%)")

        print(f"結果画像を保存しました: {output_img_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
