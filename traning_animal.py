import os
import cv2
import numpy as np
import pickle
import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --- 画像読み込み関数 ---
def load_images_from_folder(base_folder):
    images = []
    labels = []
    label_dict = {"dogs": 0, "cats": 1, "birds": 2}

    for label_name, label in label_dict.items():
        class_folder = os.path.join(base_folder, label_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder '{class_folder}' does not exist.")
            continue

        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized = cv2.resize(img, (64, 64))  # サイズ変更
                images.append(resized.flatten())  # 1次元配列に変換
                labels.append(label)

    return np.array(images), np.array(labels)

# --- データの読み込み ---
base_folder = os.path.join(".", "data")  # ./data フォルダのパス
X, y = load_images_from_folder(base_folder)

# --- データの分割 (80% 訓練データ, 20% テストデータ) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SVMモデルの作成と訓練 ---
model = SVC(kernel='linear', probability=True)  # 線形SVM
model.fit(X_train, y_train)

# --- テストデータで予測 ---
y_pred = model.predict(X_test)

# --- モデルの評価 ---
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["dogs", "cats", "birds"]))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# --- 混同行列のプロット ---
def plot_confusion_matrix(conf_matrix, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap='Blues')

    # 軸ラベルの設定
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # 各セルに数値を表示
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    plt.show()

# 混同行列のプロット
# plot_confusion_matrix(conf_matrix, ["dogs", "cats", "birds"])
plot_confusion_matrix(conf_matrix, ["dogs", "cats", "birds"])

# モデルの保存
model_name = "animal_classifier.pkl"
model_path = utils.get_model_path(model_name)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_name}")
