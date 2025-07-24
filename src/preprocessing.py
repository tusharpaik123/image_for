import cv2
import numpy as np

def ela_image(image_path, quality=90):
    img = cv2.imread(image_path)
    temp_filename = 'temp_ela.jpg'
    cv2.imwrite(temp_filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    ela_img = cv2.imread(temp_filename)
    diff = cv2.absdiff(img, ela_img)
    if diff.max() != 0:
        scale = 255.0 / diff.max()
        diff = (diff * scale).astype(np.uint8)
    return diff

def preprocess(image_path):
    IMG_SIZE = 256
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    ela = ela_image(image_path)
    ela = cv2.resize(ela, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    combined = np.concatenate([img, ela[..., :1]], axis=-1)
    return combined
