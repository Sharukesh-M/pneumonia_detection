import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from model import build_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
 
train_dir = 'chest_xray/train'
val_dir   = 'chest_xray/val'
test_dir  = 'chest_xray/test'
 
# Training augmentation
train_datagen = ImageDataGenerator(
   rescale=1./255,
   horizontal_flip=True,
   rotation_range=10,
   width_shift_range=0.1,
   height_shift_range=0.1,
   shear_range=0.2,
   zoom_range=0.2,
   fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)
 
train_gen = train_datagen.flow_from_directory(
   train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
val_gen = val_test_datagen.flow_from_directory(
   val_dir, target_size=(150,150), batch_size=32, class_mode='binary')
test_gen = val_test_datagen.flow_from_directory(
   test_dir, target_size=(150,150), batch_size=32,
   class_mode='binary', shuffle=False)

model = build_model()
model.compile(optimizer=Adam(0.001),
             loss='binary_crossentropy', metrics=['accuracy'])
 
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5,
                          restore_best_weights=True, verbose=1)
 
history = model.fit(
   train_gen,
   steps_per_epoch=train_gen.n // train_gen.batch_size,
   epochs=20,
   validation_data=val_gen,
   validation_steps=val_gen.n // val_gen.batch_size,
   callbacks=[reduce_lr, early_stop]
)
model.save('pneumonia_cnn_model.h5')
 
# Evaluation
test_gen.reset()
y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
y_true = test_gen.classes
print(classification_report(y_true, y_pred,
     target_names=['NORMAL','PNEUMONIA']))
