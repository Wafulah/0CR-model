from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

def create_vgg19(input_shape, num_classes, optimizer, loss_function):
    train_layers = VGG19(include_top=False, input_shape=input_shape)
    VG = Sequential()
    VG.add(train_layers)
    VG.add(Flatten())
    VG.add(Dense(1024, activation='selu'))
    VG.add(Dense(512, activation='selu'))
    VG.add(Dense(num_classes, activation='softmax'))
    
    VG.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    return VG
