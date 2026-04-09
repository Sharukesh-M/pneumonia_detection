import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
   Dense, Dropout, BatchNormalization)
 
def build_model(input_shape=(150,150,3)):
   model = Sequential([
       # Block 1 — 32 filters
       Conv2D(32,(3,3),padding='same',activation='relu',input_shape=input_shape),
       BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
       # Block 2 — 64 filters
       Conv2D(64,(3,3),padding='same',activation='relu'),
       BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
       # Block 3 — 128 filters
       Conv2D(128,(3,3),padding='same',activation='relu'),
       BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
       # Block 4 — 256 filters
       Conv2D(256,(3,3),padding='same',activation='relu'),
       BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
       # Classifier Head
       Flatten(),
       Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
       Dense(1,  activation='sigmoid')   # Binary output
   ])
   return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
