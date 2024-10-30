import os
import pickle

def get_traning_data_dir(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", name)
    return path

def get_test_image_dir(type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", "test", type)
    return path

def get_test_image_path(image_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", "test", f"{image_name}.jpg")
    return path

def get_video_path(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "videos", file_name)
    return path

def get_model_path(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")

    # modelsフォルダが存在しなければ作成
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    return os.path.join(models_dir, model_name)

def load_model(model_name):
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {model_path}")
    return model

def save_model(model, model_name):
    model_file = get_model_path(model_name)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_file}")
