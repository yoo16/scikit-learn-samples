import os
import cv2
import pickle
import utils

# 動画ファイルのパスを標準入力から取得
video_name = input("Enter the name of the video file (without extension): ").strip()
video_path = utils.get_video_path(video_name + ".mp4")

# 動画ファイルが存在するかチェック
if not os.path.exists(video_path):
    print(f"Error: The file '{video_path}' does not exist.")
    exit(1)

# scikit-learnモデルの読み込み
model_name = "svm_crow_classifier.pkl"
model = utils.load_model(model_name)

cap = cv2.VideoCapture(video_path)

paused = False  # 再生・停止状態を管理

try:
    while cap.isOpened():  # 動画が正常に読み込まれている間
        if not paused:  # 再生中の場合のみフレームを取得
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read the frame. Exiting...")
                break

            # グレースケール変換とリサイズ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)

            # 予測の実行
            prediction = model.predict(resized)
            label = 'Positive' if prediction == 1 else 'Negative'

            # ポジティブの場合、枠を描画し一時停止
            if label == 'Positive':
                print("Positive detected! Pausing video...")
                # paused = True  # 自動的に一時停止

                # # 矩形の描画（仮にフレーム全体に枠を描く）
                # x, y, w, h = 10, 10, frame.shape[1] - 20, frame.shape[0] - 20
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 赤色の枠

            # 結果と枠付きフレームを表示
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Prediction', frame)

        # キー入力の確認
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESCキーで終了
            print("ESC pressed. Exiting...")
            break
        elif key == ord(' '):  # スペースキーで再生・停止を切り替え
            paused = not paused
            if paused:
                print("Video paused. Press SPACE to resume.")
            else:
                print("Video resumed.")

        # ウィンドウが閉じられた場合の確認
        if cv2.getWindowProperty('Video Prediction', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting...")
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Exiting gracefully...")

finally:
    # 動画とウィンドウの解放
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released successfully.")
