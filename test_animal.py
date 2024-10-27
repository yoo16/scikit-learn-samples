import os
import cv2
import numpy as np
import pickle
import utils

def load_model(model_name):
    """モデルをpickleファイルからロードする"""
    model_path = utils.get_model_path(model_name)
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' does not exist.")
        exit(1)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from '{model_path}'")
    return model

def predict_image(model, image_path):
    """画像を予測してクラス名と確率を返す"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: '{image_path}' could not be loaded.")
        return None, None

    # 画像のリサイズと1次元配列への変換
    resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    # 予測の実行
    probabilities = model.predict_proba(resized)[0]
    prediction = np.argmax(probabilities)
    class_names = ["dog", "cat", "bird"]

    return class_names[prediction], probabilities[prediction]

def visualize_prediction(image_path, prediction, probability):
    """画像に予測結果を描画し表示する"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load '{image_path}'")
        return

    # 予測結果と確率を描画
    label = f"{prediction}: {probability * 100:.2f}%"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 画像を表示してESCキーを待機
    cv2.imshow(f"{os.path.basename(image_path)}", img)
    key = cv2.waitKey(0) & 0xFF  # キー入力を取得

    # ESCキーが押されたら終了
    if key == 27:  # ESCキーのASCIIコードは27
        print("ESC pressed. Exiting...")
        cv2.destroyAllWindows()
        exit(0)  # プログラム全体を終了

    cv2.destroyAllWindows()

def test_images(model_name, folder):
    """フォルダ内のすべての画像をテストし、結果を表示"""
    model = load_model(model_name)

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # 画像ファイルのみ対象
            prediction, probability = predict_image(model, file_path)
            if prediction is not None:
                print(f"Image: {filename} --> Prediction: {prediction} ({probability * 100:.2f}%)")
                visualize_prediction(file_path, prediction, probability)  # 可視化
        else:
            print(f"Skipping non-image file: {filename}")

# --- メイン処理 ---
test_folder = utils.get_test_image_dir("animal")  # テストフォルダを取得
model_name = "animal_classifier.pkl"
test_images(model_name, test_folder)
