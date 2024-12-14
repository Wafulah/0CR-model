import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset
BASE_DIR = 'utils'
IMG_DIR = os.path.join(BASE_DIR, 'img')  # Path to the img folder
CSV_PATH = os.path.join(BASE_DIR, 'english.csv')  # Path to the CSV file

# Load data from CSV file containing image names and labels
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Modify directory_to_df to work with the CSV and image directory
def directory_to_df(img_path, csv_path):
    # Load the CSV file with image names and labels
    df = load_data(csv_path)
    
    # Generate the full image paths by combining the img_path and image names in the CSV
    df['image'] = df['image'].apply(lambda x: os.path.join(img_path, x))  # Construct the full path to the image
    
    # Return the DataFrame with image paths and labels
    return df

def create_generators(train_df, val_df, img_shape, batch_size):
    datagen = ImageDataGenerator()
    
    # Generate training and validation generators from DataFrame
    train_gen = datagen.flow_from_dataframe(train_df, x_col='image', y_col='label',
                                            target_size=img_shape[:2], batch_size=batch_size)
    val_gen = datagen.flow_from_dataframe(val_df, x_col='image', y_col='label',
                                          target_size=img_shape[:2], batch_size=batch_size)
    
    return train_gen, val_gen

# Example usage
if __name__ == "__main__":
    # Generate training and validation data from CSV and images
    train_df = directory_to_df(IMG_DIR, CSV_PATH)
    val_df = directory_to_df(IMG_DIR, CSV_PATH)  # You can split the data or use a separate validation CSV if available
    
    # Define image shape and batch size
    IMG_SIZE = (224, 224, 3)
    BATCH_SIZE = 32
    
    # Create data generators
    train_gen, val_gen = create_generators(train_df, val_df, img_shape=IMG_SIZE, batch_size=BATCH_SIZE)
