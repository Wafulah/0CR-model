import cv2

def read_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image, img_shape):
    return cv2.resize(image, img_shape[:2]) / 255.0
