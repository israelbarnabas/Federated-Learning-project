# fl_project/task.py
"""
FL task module for WISDM dataset with TensorFlow.
Includes:
- Non-IID partitioning using Dirichlet distribution
- Per-client train/test split
- Sliding window creation
- Model loading
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, Input
import tensorflow as tf

# =========================
# Config
# =========================
DATA_PATH = "cleaned_wisdm_data.csv"
WINDOW_SIZE = 100   # ~2 seconds at 50Hz
STEP = 50           # 50% overlap
TEST_SIZE = 0.2     # Train/test split ratio
NUM_CLASSES = None  # Will be set dynamically
cached_data = None  # Cache loaded dataset


# =========================
# Sliding window creation
# =========================
def create_windows(df):
    """
    Convert accelerometer data into sliding windows.
    Labels: last sample of window.
    """
    global NUM_CLASSES

    # Encode activity labels
    le = LabelEncoder()
    df["activity"] = le.fit_transform(df["activity"])
    NUM_CLASSES = len(le.classes_)

    X_windows = []
    y_windows = []

    # Ensure temporal order
    df = df.sort_values("timestamp")
    x_vals = df[["x", "y", "z"]].values
    y_vals = df["activity"].values

    for start in range(0, len(df) - WINDOW_SIZE, STEP):
        end = start + WINDOW_SIZE
        window = x_vals[start:end]
        label = y_vals[end - 1]
        X_windows.append(window)
        y_windows.append(label)

    X = np.array(X_windows)
    y = np.array(y_windows)
    return X, y


# =========================
# Dataset loading
# =========================
def load_dataset():
    """
    Load WISDM CSV, create sliding windows, cache results.
    """
    global cached_data
    if cached_data is None:
        print("Loading WISDM dataset...")
        df = pd.read_csv(DATA_PATH)
        X, y = create_windows(df)
        cached_data = (X, y)
    return cached_data


# =========================
# Non-IID partitioning
# =========================
def dirichlet_split(X, y, num_clients, alpha=0.5, min_samples_per_client=10):
    """
    Non-IID partition using Dirichlet distribution.
    Returns: list of tuples [(X_client, y_client), ...]
    """
    data_per_client = [[] for _ in range(num_clients)]
    labels = np.unique(y)

    for label in labels:
        idx_label = np.where(y == label)[0]
        np.random.shuffle(idx_label)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))

        # Convert to counts
        counts = np.maximum((proportions * len(idx_label)).astype(int), 1)
        
        # Ensure we don't exceed available samples
        counts = np.minimum(counts, len(idx_label) // num_clients + 1)

        start = 0
        for i, cnt in enumerate(counts):
            end = min(start + cnt, len(idx_label))
            if start < end:
                data_per_client[i].extend(idx_label[start:end])
            start = end
            if start >= len(idx_label):
                break

    # Build client datasets
    client_data = []
    for indices in data_per_client:
        if len(indices) > 0:
            X_c = X[indices]
            y_c = y[indices]
            client_data.append((X_c, y_c))
        else:
            # Empty client gets small random sample
            random_indices = np.random.choice(len(X), size=min_samples_per_client, replace=False)
            client_data.append((X[random_indices], y[random_indices]))

    return client_data


# =========================
# Per-client data loading
# =========================
def get_client_data(client_id, num_clients, non_iid=True, alpha=0.5):
    """
    Return (x_train, y_train, x_test, y_test) for a single client.
    """
    X, y = load_dataset()

    if non_iid:
        # Set seed for reproducibility across calls
        np.random.seed(42)
        clients = dirichlet_split(X, y, num_clients, alpha=alpha)
    else:
        # IID split
        total_samples = len(X)
        per_client = total_samples // num_clients
        clients = []
        for i in range(num_clients):
            start = i * per_client
            end = start + per_client if i < num_clients - 1 else total_samples
            clients.append((X[start:end], y[start:end]))

    # Handle case where client_id exceeds available clients
    if client_id >= len(clients):
        client_id = client_id % len(clients)

    X_client, y_client = clients[client_id]
    
    # Ensure we have enough data
    if len(X_client) < 10:
        # Fallback: give random sample
        np.random.seed(client_id)
        indices = np.random.choice(len(X), size=100, replace=False)
        X_client, y_client = X[indices], y[indices]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X_client, y_client, 
        test_size=TEST_SIZE, 
        shuffle=True,
        random_state=client_id
    )
    
    return x_train, y_train, x_test, y_test


# =========================
# Model definition
# =========================
def load_model(input_shape=(100, 3), num_classes=None):
    """
    Create CNN model for HAR.
    """
    if num_classes is None:
        num_classes = 6  # Default for WISDM

    model = Sequential([
        Input(shape=input_shape),
        
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model