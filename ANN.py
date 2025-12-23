import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-GUI backend
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load datasets
# -----------------------------
img_height, img_width = 128, 128   # smaller images for speed
batch_size = 16

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\user\Desktop\AI project\dataset\train",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\user\Desktop\AI project\dataset\val",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\user\Desktop\AI project\dataset\test",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

class_names = train_ds.class_names
print("Class names:", class_names)

# -----------------------------
# 2. Normalize + Augment
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# -----------------------------
# 3. Define CNN model
# -----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 4. Early stopping callback
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# 5. Train model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# -----------------------------
# 6. Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2f}")

# -----------------------------
# 7. Save model (overwrite if exists)
# -----------------------------
save_path = r"C:\Users\user\Desktop\AI project\pneumonia_ann_augmented.h5"
model.save(save_path)
print(f"Model retrained and saved at: {save_path}")

# -----------------------------
# 8. Plot training curves
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Performance")
plt.savefig("training_curves.png")
print("Training curves saved as training_curves.png")
