import os
import cv2
import numpy as np
import pickle
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

# 画像読み込み関数 (画像データとファイル名を取得)
def load_images(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (64, 64))  # サイズ変更
            images.append(resized.flatten())  # 1次元配列に変換
            filenames.append(filename)  # ファイル名を保存
    return np.array(images), filenames

# ポジティブ画像とネガティブ画像のフォルダパス
positive_path = utils.get_traning_data_dir("positives")
negative_path = utils.get_traning_data_dir("negatives")

# ポジティブとネガティブの画像を読み込み
X_pos, filenames_pos = load_images(positive_path)
X_neg, filenames_neg = load_images(negative_path)

# ラベルの設定 (ポジティブ: 1, ネガティブ: 0)
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

# データとラベルの結合
X = np.vstack((X_pos, X_neg))
y = np.hstack((y_pos, y_neg))

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMモデルの作成と訓練
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # ポジティブクラスの確率

# --- モデルの精度を表示 ---
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# --- 混同行列とROC曲線を1つのグラフにまとめる ---
def plot_combined_graph(conf_matrix, y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 混同行列のプロット
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'], ax=axes[0])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Confusion Matrix')

    # ROC曲線のプロット
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # ランダム分類器の基準線
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")

    # レイアウト調整と表示
    plt.tight_layout()
    plt.show()

# モデルの保存
model_name = "crow_classifier.pkl"
model_path = utils.get_model_path(model_name)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# 混同行列とROC曲線のプロットを実行
plot_combined_graph(conf_matrix, y_test, y_prob)
