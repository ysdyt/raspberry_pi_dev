from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from datetime import datetime
import platform
import glob  # 追加

# アプリケーションのルートディレクトリを取得
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# クラウド環境かどうかを判定する
IS_CLOUD_ENV = (
    os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT") == "cloud"
    or os.environ.get("STREAMLIT_CLOUD", "0") == "1"
    or not os.path.exists("/sys/firmware/devicetree/base")
)

# 実行端末がRaspberry Piかどうかを判定
IS_RASPBERRY_PI = False
if not IS_CLOUD_ENV:
    try:
        IS_RASPBERRY_PI = (
            platform.system() == "Linux"
            and os.path.exists("/sys/firmware/devicetree/base/model")
            and "Raspberry Pi" in open("/sys/firmware/devicetree/base/model").read()
        )
    except Exception as e:
        print(f"Raspberry Pi判定中にエラーが発生しました: {e}")
        pass

# Raspberry Pi環境でない場合はpicamera2のインポートをスキップ
PICAMERA_AVAILABLE = False
if IS_RASPBERRY_PI and not IS_CLOUD_ENV:
    try:
        from picamera2 import Picamera2

        PICAMERA_AVAILABLE = True
    except ImportError:
        print("picamera2モジュールをインポートできませんでした。カメラ機能は無効です。")


def capture_image():
    """
    Raspberry Pi カメラで画像を撮影し、指定したパスに保存します

    Returns:
        撮影した画像のNumPy配列と保存したファイルパス
    """
    # captured_imgディレクトリが存在しない場合は作成
    output_dir = os.path.join(os.path.dirname(__file__), "captured_img")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 撮影時間をファイル名として保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_img_path = os.path.join(output_dir, f"capture_{timestamp}.jpg")

    # 実行端末がRaspberry Piでない場合の処理
    if not IS_RASPBERRY_PI or not PICAMERA_AVAILABLE:
        # 開発環境では、テスト用の画像を読み込む
        print("開発環境では、テスト用の画像を使用します")
        test_image_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
        if os.path.exists(test_image_path):
            img_test = cv2.imread(test_image_path)
            # テスト画像をcaptured_imgに保存
            cv2.imwrite(output_img_path, img_test)
            return img_test, output_img_path
        else:
            # テスト画像がなければ黒い画像を生成
            print(f"テスト画像 {test_image_path} が見つかりません。黒い画像を生成します。")
            img_black = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(output_img_path, img_black)
            return img_black, output_img_path

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

    # 撮影した画像を保存
    cv2.imwrite(output_img_path, img_array_bgr)
    print(f"撮影画像を保存しました: {output_img_path}")

    # 撮影した画像を返すが、基本は使わない。保存された画像を使用する。
    # return img_array_bgr, output_img_path


def get_latest_captured_image():
    """
    captured_imgフォルダから最新の画像を取得します

    Returns:
        最新の画像のNumPy配列とそのファイルパス、画像がない場合はNone, None
    """
    # count_person/captured_img ディレクトリを参照
    output_dir = os.path.join(os.path.dirname(__file__), "captured_img")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリを作成しました: {output_dir}")

    # ファイル一覧を取得してソート
    files = glob.glob(os.path.join(output_dir, "capture_*.jpg"))
    if not files:
        print(f"{output_dir}フォルダに画像がありません。")
        return None, None

    # ファイル名でソート（日付形式なので、最新のものは最後）
    latest_file = max(files, key=os.path.getctime)

    # 画像を読み込み
    img = cv2.imread(latest_file)

    return img, latest_file


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
    # capture_image()を実行し、画像を保存するだけ
    capture_image()
