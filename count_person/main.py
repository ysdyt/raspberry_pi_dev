from ultralytics import YOLO
import cv2
import os
from datetime import datetime


def process_image(image_path, confidence_threshold=0.3):
    """
    画像内の人物を検出し、バウンディングボックスを描画した画像と統計情報を返します

    Args:
        image_path: 処理する画像のパス
        confidence_threshold: 検出の信頼度しきい値

    Returns:
        processed_image: バウンディングボックス付きの画像
        stats: 検出統計情報の辞書
    """
    # YOLOv11をロード
    model_path = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
    model = YOLO(model_path)

    # 画像を読み込む
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"エラー: 画像 {image_path} を読み込めませんでした")

    # 推論を実行
    results = model(img)

    # 人を検出（クラス0は通常「person」を表す）
    person_count = 0
    confidence_scores = []  # 信頼度スコアを保存するリスト
    img_with_boxes = img.copy()  # 元の画像をコピー

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


def process_default_image(confidence_threshold=0.3):
    """
    デフォルトの画像を処理します。固定パスの画像を使用。

    Args:
        confidence_threshold: 検出の信頼度しきい値

    Returns:
        img_with_boxes: バウンディングボックス付きの画像
        stats: 検出統計情報の辞書
    """
    # 画像パスを指定（~を展開）
    image_path = os.path.expanduser("~/Downloads/niho_now3.jpg")
    return process_image(image_path, confidence_threshold)


# スクリプトとして実行される場合のメイン処理
if __name__ == "__main__":
    # 画像パスを指定（~を展開）
    image_path = os.path.expanduser("~/Downloads/niho_now3.jpg")
    confidence_threshold = 0.3

    try:
        # 画像処理
        processed_image, stats = process_image(image_path, confidence_threshold)

        # 結果出力
        print(f"信頼度{confidence_threshold * 100}%超の人数: {stats['person_count']}人")

        # 統計情報
        if stats["confidence_scores"]:
            avg_conf = stats["avg_confidence"]
            print(f"平均信頼度: {avg_conf:.2f} ({avg_conf * 100:.1f}%)")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
