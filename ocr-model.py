import sys
from tensorflow.keras.models import load_model
from utils.helpers import read_image, preprocess_image

MODEL_DIR = 'saved_models'  # Directory where models are saved

def load_pretrained_model(model_type, input_shape, num_classes):
    """Load a pre-trained model by type."""
    model_path = f"{MODEL_DIR}/{model_type}.h5"
    try:
        model = load_model(model_path)
        print(f"Loaded {model_type} model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")
        sys.exit(1)

def main(image_path, model_type):
    img_shape = (224, 224, 3)  # Input shape for all models
    num_classes = 26  # Number of output classes (A-Z)

    # Load the pre-trained model
    model = load_pretrained_model(model_type, img_shape, num_classes)

    # Read and preprocess the image
    image = read_image(image_path)
    processed_image = preprocess_image(image, img_shape)

    # Predict the class
    prediction = model.predict(processed_image[None, ...])
    predicted_character = chr(prediction.argmax() + ord('A'))
    print("Predicted Text:", predicted_character)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python ocr_model.py <path-to-image> <model-type>")
        print("Available model types: custom, efficientnetb7, mobilenetv2, vgg19, densenet121")
        sys.exit(1)

    image_path = sys.argv[1]
    model_type = sys.argv[2]
    main(image_path, model_type)
