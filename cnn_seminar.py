import os
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_dataframe(folder_path):
    data = {'image': [], 'label': []}

    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                label = class_folder
                data['image'].append(image_path)
                data['label'].append(label)

    df = pd.DataFrame(data)
    return df

dataset_path = r'C:\Users\rahul\OneDrive\Desktop\Arjun_seminar\archive'
batch_size = 40
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
epochs = 20 # Increase epochs for better training

train_df = create_dataframe(os.path.join(dataset_path, 'train'))
test_df = create_dataframe(os.path.join(dataset_path, 'test'))
valid_df = create_dataframe(os.path.join(dataset_path, 'val'))

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(train_df,
                                        x_col='image',
                                        y_col='label',
                                        target_size=img_size,
                                        class_mode='categorical',
                                        color_mode='rgb',
                                        shuffle=True,
                                        batch_size=batch_size)

valid_gen = datagen.flow_from_dataframe(valid_df,
                                        x_col='image',
                                        y_col='label',
                                        target_size=img_size,
                                        class_mode='categorical',
                                        color_mode='rgb',
                                        shuffle=False,
                                        batch_size=batch_size)

test_gen = datagen.flow_from_dataframe(test_df,
                                       x_col='image',
                                       y_col='label',
                                       target_size=img_size,
                                       class_mode='categorical',
                                       color_mode='rgb',
                                       shuffle=False,
                                       batch_size=batch_size)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen,
                    epochs=epochs,
                    verbose=1,
                    validation_data=valid_gen)

# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save model weights in native Keras format
model.save('model_cnn.h5')
