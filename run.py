import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,

    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs = 50,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, early_stop]
)

model.load_weights('best_model.keras')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

def super_preprocess(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

while True:
    image_path = input("\nEnter image path (or 'exit'): ")
    if image_path.lower() == 'exit':
        break
    
    if not os.path.exists(image_path):
        print("Image not found!")
        continue
    
    try:
        processed_img = super_preprocess(image_path)
        prediction = model.predict(processed_img)
        confidence = np.max(prediction) * 100
        predicted_class = np.argmax(prediction)
        
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(processed_img[0,:,:,0], cmap='gray')
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.1f}%")
        
        plt.subplot(1,2,2)
        plt.bar(range(10), prediction[0])
        plt.xticks(range(10))
        plt.xlabel('Digits')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
