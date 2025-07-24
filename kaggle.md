import os

# Base directory of your CASIA2 dataset in Kaggle input
base_dir = '/kaggle/input/casia-20-image-tampering-detection-dataset/CASIA2'

# Check subfolders
print("Subfolders:", os.listdir(base_dir))
print("Sample in 'Au':", os.listdir(os.path.join(base_dir, 'Au'))[:3])
print("Sample in 'Tp':", os.listdir(os.path.join(base_dir, 'Tp'))[:3])


extensions_supported = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

au_dir = os.path.join(base_dir, 'Au')
tp_dir = os.path.join(base_dir, 'Tp')

authentic_images = [os.path.join(au_dir, f) for f in os.listdir(au_dir) if f.lower().endswith(extensions_supported)]
tampered_images = [os.path.join(tp_dir, f) for f in os.listdir(tp_dir) if f.lower().endswith(extensions_supported)]

all_paths = authentic_images + tampered_images
all_labels = [0] * len(authentic_images) + [1] * len(tampered_images)

print(f"Authentic images: {len(authentic_images)}")
print(f"Tampered images: {len(tampered_images)}")



from sklearn.model_selection import train_test_split

paths_train, paths_val, labels_train, labels_val = train_test_split(
    all_paths,
    all_labels,
    stratify=all_labels,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(paths_train)}")
print(f"Validation samples: {len(paths_val)}")






import cv2
import numpy as np

def ela_image(image_path, quality=90):
    img = cv2.imread(image_path)
    temp_filename = '/kaggle/working/temp_ela.jpg'
    cv2.imwrite(temp_filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    ela_img = cv2.imread(temp_filename)
    diff = cv2.absdiff(img, ela_img)
    if diff.max() != 0:
        scale = 255.0 / diff.max()
        diff = (diff * scale).astype(np.uint8)
    return diff





IMG_SIZE = 256

def preprocess(image_path, label):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    ela = ela_image(image_path)
    ela = cv2.resize(ela, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

    combined = np.concatenate([img, ela[..., :1]], axis=-1)  # RGB + ELA channel (4 channels)
    return combined, label

def make_dataset(paths, labels):
    X, y = [], []
    for p, l in zip(paths, labels):
        img, lbl = preprocess(p, l)
        X.append(img)
        y.append(lbl)
    return np.array(X), np.array(y)

X_train, y_train = make_dataset(paths_train, labels_train)
X_val, y_val = make_dataset(paths_val, labels_val)

print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)





from tensorflow.keras import layers, models

input_shape = (IMG_SIZE, IMG_SIZE, 4)  # 4 input channels: RGB + ELA

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()





from sklearn.metrics import classification_report, confusion_matrix

val_preds = (model.predict(X_val) > 0.5).astype(int).flatten()

print(classification_report(y_val, val_preds, target_names=['Authentic', 'Tampered']))
print("Confusion matrix:\n", confusion_matrix(y_val, val_preds))




model.save('/kaggle/working/image_forgery_detector.h5')
print("Model saved to /kaggle/working/image_forgery_detector.h5")
