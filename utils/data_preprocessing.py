import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def directory_to_df(path):
    df = []
    chars = 'abcdefghijklmnopqrstuvwxyz'
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        cls_name = cls.split('_')[0]
        if cls_name not in chars:
            continue
        for img_path in os.listdir(cls_path):
            df.append([os.path.join(cls_path, img_path), cls_name])
    return pd.DataFrame(df, columns=['image', 'label'])

def create_generators(train_df, val_df, img_shape, batch_size):
    datagen = ImageDataGenerator()
    train_gen = datagen.flow_from_dataframe(train_df, x_col='image', y_col='label',
                                            target_size=img_shape[:2], batch_size=batch_size)
    val_gen = datagen.flow_from_dataframe(val_df, x_col='image', y_col='label',
                                          target_size=img_shape[:2], batch_size=batch_size)
    return train_gen, val_gen
