import os
import pickle
import cv2
import matplotlib.pyplot as plt
import utils


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


def show_predictions(model, folder_path):
    """フォルダ内の画像を順に処理し、予測結果を表示"""
    image_files = [f for f in os.listdir(
        folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No images found in '{folder_path}'")
        return

    num_images = len(image_files)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # 各画像に対して予測を実行し、結果を表示
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        label = predict_image(model, image_path)

        # 画像を表示
        ax = axes[i] if num_images > 1 else axes  # 画像が1枚の場合も対応
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f"{image_file}\nPrediction: {label}", fontsize=10)
        ax.axis('off')

    plt.show()


# モデルのパスを取得して読み込み
model_name = "crow_classifier.pkl"
model = utils.load_model(model_name)

# data/test/フォルダのパス
test_folder = utils.get_test_image_dir("crow")

# フォルダ内の画像を処理し、結果を表示
if os.path.exists(test_folder):
    show_predictions(model, test_folder)
else:
    print(f"Error: The folder '{test_folder}' does not exist.")
