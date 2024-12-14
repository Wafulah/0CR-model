from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_densenet121(input_shape, num_classes, optimizer, loss_function):
    train_layers = DenseNet121(include_top=False, input_shape=input_shape)
    Den = Sequential()
    Den.add(train_layers)
    Den.add(Flatten())
    Den.add(Dense(1024, activation='selu'))
    Den.add(Dense(512, activation='selu'))
    Den.add(Dense(num_classes, activation='softmax'))
    
    Den.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return Den
