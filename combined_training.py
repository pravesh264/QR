import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to dataset
FIRST_PRINTS_DIR = "D:/Projects/QRGuardian/dataset/first_prints"
SECOND_PRINTS_DIR = "D:/Projects/QRGuardian/dataset/second_prints"

# --- Part 1: Data Exploration ---
def explore_data():
    first_prints_files = os.listdir(FIRST_PRINTS_DIR)
    second_prints_files = os.listdir(SECOND_PRINTS_DIR)
    
    print(f"Number of first prints: {len(first_prints_files)}")
    print(f"Number of second prints: {len(second_prints_files)}")
    
    first_img_path = os.path.join(FIRST_PRINTS_DIR, first_prints_files[0])
    second_img_path = os.path.join(SECOND_PRINTS_DIR, second_prints_files[0])
    first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    second_img = cv2.imread(second_img_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"First print image shape: {first_img.shape if first_img is not None else 'Invalid'}")
    print(f"Second print image shape: {second_img.shape if second_img is not None else 'Invalid'}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(first_img, cmap="gray") if first_img is not None else plt.text(0.5, 0.5, "Invalid Image", ha="center")
    plt.title("First Print")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(second_img, cmap="gray") if second_img is not None else plt.text(0.5, 0.5, "Invalid Image", ha="center")
    plt.title("Second Print")
    plt.axis("off")
    plt.savefig("dataset_comparison.png")
    plt.close()

# --- Part 2: Feature Engineering ---
def extract_features(img_path, add_noise=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load image at {img_path}")
        return None
    img = cv2.resize(img, (224, 224))
    if add_noise:
        noise = np.random.normal(0, 8, img.shape)  # Reduced to 8 for SVM test set
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    edges = cv2.Canny(img, 100, 200)
    edge_intensity = np.mean(edges)
    texture_variance = np.std(img)
    return np.array([edge_intensity, texture_variance])

def analyze_features():
    first_features = [extract_features(os.path.join(FIRST_PRINTS_DIR, fname)) for fname in os.listdir(FIRST_PRINTS_DIR)]
    second_features = [extract_features(os.path.join(SECOND_PRINTS_DIR, fname)) for fname in os.listdir(SECOND_PRINTS_DIR)]
    
    first_features = [f for f in first_features if f is not None]
    second_features = [f for f in second_features if f is not None]
    
    first_features = np.array(first_features)
    second_features = np.array(second_features)
    
    print("First Prints - Edge Intensity Mean:", first_features[:, 0].mean())
    print("First Prints - Texture Variance Mean:", first_features[:, 1].mean())
    print("Second Prints - Edge Intensity Mean:", second_features[:, 0].mean())
    print("Second Prints - Texture Variance Mean:", second_features[:, 1].mean())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(first_features[:, 0], first_features[:, 1], c="blue", label="First Prints", alpha=0.5)
    plt.scatter(second_features[:, 0], second_features[:, 1], c="red", label="Second Prints", alpha=0.5)
    plt.xlabel("Edge Intensity")
    plt.ylabel("Texture Variance")
    plt.legend()
    plt.savefig("feature_scatter.png")
    plt.close()

# --- Part 3: SVM Training ---
def train_svm():
    features = []
    labels = []
    file_paths = []
    
    for fname in os.listdir(FIRST_PRINTS_DIR):
        path = os.path.join(FIRST_PRINTS_DIR, fname)
        feats = extract_features(path)
        if feats is not None:
            features.append(feats)
            labels.append(0)
            file_paths.append(path)
    
    for fname in os.listdir(SECOND_PRINTS_DIR):
        path = os.path.join(SECOND_PRINTS_DIR, fname)
        feats = extract_features(path)
        if feats is not None:
            features.append(feats)
            labels.append(1)
            file_paths.append(path)
    
    X = np.array(features)
    y = np.array(labels)
    file_paths = np.array(file_paths)
    
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, file_paths, train_size=0.5, test_size=0.5, random_state=42
    )
    
    X_test_noisy = []
    for path in paths_test:
        feats = extract_features(path, add_noise=True)
        if feats is not None:
            X_test_noisy.append(feats)
        else:
            print(f"Skipping noisy feature extraction for {path}")
            X_test_noisy.append(X_test[len(X_test_noisy)])
    X_test = np.array(X_test_noisy)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {'C': [0.1, 0.5], 'kernel': ['linear']}  # Adjusted C range
    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_svm = grid_search.best_estimator_
    joblib.dump(best_svm, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print(f"SVM model saved with best params: {grid_search.best_params_}")
    
    return best_svm, X_test_scaled, y_test, scaler

# --- Part 4: CNN Training ---
def add_noise(image):
    noise = np.random.normal(0, 25, image.shape)  # Increased to 25 for CNN test set
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def load_images_from_folder(folder, add_noise_to_test=False):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            if add_noise_to_test:
                img = add_noise(img)
            images.append(img)
        else:
            print(f"Warning: Failed to load image at {os.path.join(folder, filename)}")
    return images

def train_cnn():
    first_prints = load_images_from_folder(FIRST_PRINTS_DIR)
    second_prints = load_images_from_folder(SECOND_PRINTS_DIR)
    
    X = np.array(first_prints + second_prints)
    y = np.array([0] * len(first_prints) + [1] * len(second_prints))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=42)
    
    X_test = np.array([add_noise(img) for img in X_test])  # Ensure noise is applied
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=16)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=16)
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reverted to 0.0001
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(X_train) // 16),
        epochs=50,
        validation_data=test_generator,
        validation_steps=max(1, len(X_test) // 16)
    )
    
    model.save("cnn_model.keras")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig("cnn_training_history.png")
    plt.close()
    
    return model, X_test, y_test

# --- Part 5: Model Evaluation ---
def evaluate_models(svm, cnn, X_test_svm, y_test_svm, X_test_cnn, y_test_cnn):
    # SVM Evaluation
    svm_pred = svm.predict(X_test_svm)
    svm_accuracy = accuracy_score(y_test_svm, svm_pred)
    svm_precision = precision_score(y_test_svm, svm_pred)
    svm_recall = recall_score(y_test_svm, svm_pred)
    svm_f1 = f1_score(y_test_svm, svm_pred)
    svm_cm = confusion_matrix(y_test_svm, svm_pred)
    
    # CNN Evaluation
    cnn_pred = (cnn.predict(X_test_cnn / 255.0) > 0.5).astype(int)
    cnn_accuracy = accuracy_score(y_test_cnn, cnn_pred)
    cnn_precision = precision_score(y_test_cnn, cnn_pred)
    cnn_recall = recall_score(y_test_cnn, cnn_pred)
    cnn_f1 = f1_score(y_test_cnn, cnn_pred)
    cnn_cm = confusion_matrix(y_test_cnn, cnn_pred)
    
    # Print detailed comparison
    print("Model Comparison:")
    print("\nSVM Metrics:")
    print(f"Accuracy: {svm_accuracy:.2f}")
    print(f"Precision: {svm_precision:.2f}")
    print(f"Recall: {svm_recall:.2f}")
    print(f"F1-Score: {svm_f1:.2f}")
    print("\nCNN Metrics:")
    print(f"Accuracy: {cnn_accuracy:.2f}")
    print(f"Precision: {cnn_precision:.2f}")
    print(f"Recall: {cnn_recall:.2f}")
    print(f"F1-Score: {cnn_f1:.2f}")
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("SVM Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    sns.heatmap(cnn_cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("CNN Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.savefig("confusion_matrices.png")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    print("=== Step 1: Data Exploration ===")
    explore_data()
    
    print("\n=== Step 2: Feature Engineering ===")
    analyze_features()
    
    print("\n=== Step 3: Training SVM ===")
    svm, X_test_svm, y_test_svm, scaler = train_svm()
    
    print("\n=== Step 4: Training CNN ===")
    cnn, X_test_cnn, y_test_cnn = train_cnn()
    
    print("\n=== Step 5: Model Evaluation ===")
    evaluate_models(svm, cnn, X_test_svm, y_test_svm, X_test_cnn, y_test_cnn)
    
    print("\nTraining and analysis complete.")