from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# YOLOv11をロード
model = YOLO("yolo11n.pt")

# 画像パスを指定（~を展開）
image_path = os.path.expanduser("~/Downloads/niho_now3.jpg")

# 画像を読み込む
img = cv2.imread(image_path)

if img is None:
    print(f"エラー: 画像 {image_path} を読み込めませんでした")
    exit()

# 推論を実行
results = model(img)

# 人を検出（クラス0は通常「person」を表す）
person_count = 0
confidence_scores = []  # 信頼度スコアを保存するリスト
confidence_threshold = 0.3  # 30%の信頼度しきい値
img_with_boxes = img.copy()  # 元の画像をコピー

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        # クラス0は「person」
        if cls == 0:
            # 信頼度を取得
            conf = float(box.conf[0])

            # 信頼度が30%より大きい場合のみカウントして表示
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

# 現在の時間を取得して画像名に使用
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join("processed_img", f"valid_person_detection_{timestamp}.jpg")

# 画像の中央上部に時間を表示
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

# 画像を保存
cv2.imwrite(output_path, img_with_boxes)

print(f"信頼度{confidence_threshold * 100}%超の人数: {person_count}人")
print(f"検出結果の画像を保存しました: {output_path}")

# 統計情報
if confidence_scores:
    avg_conf = sum(confidence_scores) / len(confidence_scores)
    print(f"平均信頼度: {avg_conf:.2f} ({avg_conf * 100:.1f}%)")
