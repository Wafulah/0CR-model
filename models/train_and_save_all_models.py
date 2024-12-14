import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.data_preprocessing import create_generators  # Assuming this exists for data generation
from tensorflow.keras.applications import VGG19, DenseNet121, EfficientNetB7, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPool2D, Input

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 26  # For example, A-Z mapping
EPOCHS = 10
BATCH_SIZE = 32
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def create_vgg19(input_shape, num_classes):
    base_model = VGG19(include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='selu'),
        Dense(512, activation='selu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def create_densenet121(input_shape, num_classes):
    base_model = DenseNet121(include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='selu'),
        Dense(512, activation='selu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def create_efficientnetb7(input_shape, num_classes):
    base_model = EfficientNetB7(include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='selu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def create_mobilenetv2(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(1024, activation='selu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def create_custom_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape, name="Input"),
        Conv2D(3, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((3, 3)),
        Conv2D(256, (3, 3), activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(1024, activation='selu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model

def train_and_save_model(model, model_name, train_gen, valid_gen):
    model_path = os.path.join(MODEL_DIR, f'{model_name}.h5')
    if os.path.exists(model_path):
        print(f"{model_name} model already exists at {model_path}")
        return

    print(f"Training {model_name}...")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=[ModelCheckpoint(model_path, save_best_only=True)]
    )
    print(f"{model_name} model saved to {model_path}")

if __name__ == "__main__":
    # Generate data
    train_gen, valid_gen, _ = create_generators(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # Create models
    models = {
        "vgg19": create_vgg19(IMG_SIZE, NUM_CLASSES),
        "densenet121": create_densenet121(IMG_SIZE, NUM_CLASSES),
        "efficientnetb7": create_efficientnetb7(IMG_SIZE, NUM_CLASSES),
        "mobilenetv2": create_mobilenetv2(IMG_SIZE, NUM_CLASSES),
        "custom_model": create_custom_model(IMG_SIZE, NUM_CLASSES),
    }

    # Train and save each model
    for model_name, model in models.items():
        train_and_save_model(model, model_name, train_gen, valid_gen)
