import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Path to dataset
IMAGE_DIR = "C:/Users/trung/OneDrive/Desktop/B3/medicine/prac2/training_set"
TEST_DIR = "C:/Users/trung/OneDrive/Desktop/B3/medicine/prac2/test_set"

# Get file names
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if "_HC.png" in f])
mask_files = sorted([f for f in os.listdir(IMAGE_DIR) if "_Annotation.png" in f])
test_files = sorted([f for f in os.listdir(TEST_DIR) if "_HC.png" in f])

# Resize images
IMG_SIZE = (256, 256)

def load_image(image_path):
    image = cv2.imread(os.path.join(IMAGE_DIR, image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMG_SIZE) / 255.0  # Normalize
    return image

# Extract ellipse bounding box from annotation
def extract_ellipse(mask_path):
    mask = cv2.imread(os.path.join(IMAGE_DIR, mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if len(cnt) >= 5]
    if valid_contours:
        ellipse = cv2.fitEllipse(valid_contours[0])  # (x, y), (a, b), theta
        (x, y), (a, b), theta = ellipse
        return [x / 256, y / 256, a / 256, b / 256, theta / 360]  # Normalize values
    return [0, 0, 0, 0, 0]  # Default if no contour found

# Load images & ellipse parameters
images, ellipses = [], []
for img, mask in zip(image_files, mask_files):
    images.append(load_image(img))
    ellipses.append(extract_ellipse(mask))

# Convert to NumPy arrays
images = np.array(images).reshape(-1, 256, 256, 1)
ellipses = np.array(ellipses)

# Load test images
test_images = np.array([load_image(img) for img in test_files]).reshape(-1, 256, 256, 1)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(images, ellipses, test_size=0.2, random_state=42)

# ESPNet Regression Model
def espnet_regression_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)  # Predict x, y, a, b, theta
    
    model = keras.Model(inputs, outputs)
    return model

model = espnet_regression_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64
)

# Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training & Validation Loss')
plt.show()

def calculate_iou(true_ellipse, pred_ellipse):
    true_mask = np.zeros((256, 256), dtype=np.uint8)
    pred_mask = np.zeros((256, 256), dtype=np.uint8)

    # Chuyển đổi tọa độ từ [0,1] về kích thước ảnh gốc
    x1, y1, a1, b1, theta1 = (np.array(true_ellipse) * [256, 256, 256, 256, 360]).astype(int)
    x2, y2, a2, b2, theta2 = (np.array(pred_ellipse) * [256, 256, 256, 256, 360]).astype(int)

    # Kiểm tra nếu ellipse có kích thước hợp lệ
    if a1 > 0 and b1 > 0:
        cv2.ellipse(true_mask, (int(x1), int(y1)), (int(a1 // 2), int(b1 // 2)), float(theta1), 0, 360, 255, -1)

    if a2 > 0 and b2 > 0:
        cv2.ellipse(pred_mask, (int(x2), int(y2)), (int(a2 // 2), int(b2 // 2)), float(theta2), 0, 360, 255, -1)

    # Tính toán IoU
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    
    return intersection / union if union > 0 else 0

def calculate_circumference(a, b):
    """Calculate the circumference of an ellipse using Ramanujan's Approximation."""
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

# Evaluate IoU on validation set
y_pred = model.predict(X_val)
iou_scores = [calculate_iou(y_true, y_pred) for y_true, y_pred in zip(y_val, y_pred)]
mean_iou = np.mean(iou_scores)
print(f"Mean IoU on validation set: {mean_iou:.4f}")

def plot_predicted_ellipse(image, ellipse_params):
    x, y, a, b, theta = ellipse_params
    x, y, a, b, theta = x * 256, y * 256, a * 256, b * 256, theta * 360
    
    # Tính chu vi
    circumference = calculate_circumference(a, b)
    
    # Chuyển ảnh sang RGB để vẽ ellipse
    img = (image * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Vẽ ellipse lên ảnh
    if a > 0 and b > 0:
        cv2.ellipse(img, (int(x), int(y)), (int(a // 2), int(b // 2)), int(theta), 0, 360, (0, 255, 0), 2)
    
    # Hiển thị ảnh với chu vi
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted Ellipse - Circumference: {circumference:.2f} pixels")
    plt.axis("off")
    plt.show()

# Predict on test set
for img in test_images[:5]:
    img_input = np.expand_dims(img, axis=0)
    predicted_ellipse = model.predict(img_input)[0]
    plot_predicted_ellipse(img.squeeze(), predicted_ellipse)
