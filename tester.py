import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. Load datasets (same preprocessing as training)
img_height, img_width = 128, 128
batch_size = 16

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\user\Desktop\AI project\AI-algorithm\dataset\test",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

class_names = test_ds.class_names

normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 2. Load saved model
model = tf.keras.models.load_model(r"C:\Users\user\Desktop\AI project\pneumonia_ann_augmented.h5")

# 3. Evaluate accuracy
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2f}")

# 4. Predictions
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()

# 5. Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 6. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], class_names)
plt.yticks([0,1], class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")

# 7. ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")
