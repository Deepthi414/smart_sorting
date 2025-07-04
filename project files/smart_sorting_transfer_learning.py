
# Smart Sorting: Transfer Learning for Identifying Rotten Fruits and Vegetables

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os

# Set parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Define data directories
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=BATCH_SIZE, class_mode='binary')

# Load pre-trained MobileNetV2 model without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Save model
model.save('smart_sorting_model.h5')

# To use this model for predictions on new images:
# from tensorflow.keras.models import load_model
# model = load_model('smart_sorting_model.h5')
# prediction = model.predict(preprocessed_image)
