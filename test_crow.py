import os
import pickle
import cv2
import matplotlib.pyplot as plt
import utils
import math

def predict_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: '{image_path}' could not be loaded.")
        return None

    # 画像のリサイズと1次元配列への変換
    resized = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)

    # 予測の実行
    prediction = model.predict(resized)
    label = 'P' if prediction == 1 else 'N'
    return label

def show_predictions(model, folder_path, images_per_row=5):
    image_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in '{folder_path}'")
        return

    num_images = len(image_files)
    num_rows = math.ceil(num_images / images_per_row)  # 行数を計算

    # 画像のサイズ調整: 全体の幅と高さを調整
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(2 * images_per_row, 2 * num_rows))
    axes = axes.flatten()  # 1次元にしてループで扱いやすくする

    # 各画像に対して予測を実行し、結果を表示
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        label = predict_image(model, image_path)

        # 画像を表示
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"{image_file}\nPrediction: {label}", fontsize=8)  # 小さめのフォント
        axes[i].axis('off')

    # 使われなかった軸を非表示にする
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# モデルのパスを取得して読み込み
model_name = "crow_classifier.pkl"
model = utils.load_model(model_name)

# data/test/フォルダのパス
test_folder = utils.get_test_image_dir("crow")

# フォルダ内の画像を処理し、結果を表示
if os.path.exists(test_folder):
    show_predictions(model, test_folder, images_per_row=5)  # 1行に5枚表示
else:
    print(f"Error: The folder '{test_folder}' does not exist.")
