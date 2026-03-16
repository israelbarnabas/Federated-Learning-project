"""
Enhanced FL task module for WISDM dataset with improved preprocessing,
smaller model for DP compatibility, and better data distribution.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, Input, Model
from tensorflow.keras import regularizers
import tensorflow as tf

# =========================
# Config
# =========================
DATA_PATH = "cleaned_wisdm_data.csv"
WINDOW_SIZE = 100   # ~2 seconds at 50Hz
STEP = 50           # 50% overlap
TEST_SIZE = 0.2     # Train/test split
VAL_SIZE = 0.1      # Validation split (new)
MIN_SAMPLES_PER_CLIENT = 50  # Minimum to ensure trainable clients (was 10)
NUM_CLASSES = None
cached_data = None
cached_scaler = None  # For consistent normalization


# =========================
# Enhanced Sliding Window
# =========================
def create_windows(df, window_size=WINDOW_SIZE, step=STEP):
    """
    Convert accelerometer data into sliding windows with optional overlap.
    Improved: Better handling of edge cases, consistent window generation.
    """
    global NUM_CLASSES

    # Encode activity labels
    le = LabelEncoder()
    df["activity"] = le.fit_transform(df["activity"])
    NUM_CLASSES = len(le.classes_)
    
    print(f"[Data] Loaded {len(df)} samples, {NUM_CLASSES} classes: {list(le.classes_)}")

    # Ensure temporal order
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Extract values
    x_vals = df[["x", "y", "z"]].values
    y_vals = df["activity"].values

    # Create windows with validation
    X_windows = []
    y_windows = []
    
    total_windows = (len(df) - window_size) // step + 1
    
    for i, start in enumerate(range(0, len(df) - window_size + 1, step)):
        end = start + window_size
        if end > len(df):
            break
            
        window = x_vals[start:end]
        label = y_vals[end - 1]  # Label from last timestep
        
        # Validate window
        if window.shape == (window_size, 3) and not np.any(np.isnan(window)):
            X_windows.append(window)
            y_windows.append(label)
        
        # Progress indicator for large datasets
        if i % 10000 == 0 and i > 0:
            print(f"[Data] Processed {i}/{total_windows} windows...")

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_windows, dtype=np.int32)
    
    print(f"[Data] Created {len(X)} valid windows from {len(df)} raw samples")
    
    return X, y, le  # Return encoder for reference


# =========================
# Data Normalization (NEW)
# =========================
def normalize_data(X_train, X_test, fit_scaler=True):
    """
    Standardize accelerometer data per-axis.
    Critical for DP-SGD: prevents gradient explosion from varying scales.
    """
    global cached_scaler
    
    # Reshape for sklearn: (samples*timesteps, features)
    original_shape = X_train.shape
    X_train_reshaped = X_train.reshape(-1, 3)
    X_test_reshaped = X_test.reshape(-1, 3)
    
    if fit_scaler or cached_scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        cached_scaler = scaler
        print(f"[Data] Fitted scaler: mean={scaler.mean_}, scale={scaler.scale_}")
    else:
        scaler = cached_scaler
        X_train_scaled = scaler.transform(X_train_reshaped)
    
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back
    X_train = X_train_scaled.reshape(original_shape)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    return X_train, X_test


# =========================
# Improved Non-IID Partitioning
# =========================
def dirichlet_split(X, y, num_clients, alpha=0.5, min_samples=MIN_SAMPLES_PER_CLIENT, seed=None):
    """
    Improved Dirichlet partition with:
    - Configurable seed for reproducibility
    - Minimum sample guarantee per client
    - Better handling of class imbalance
    """
    np.random.seed(seed)  # Only set if provided, else use current state
    
    data_per_client = [[] for _ in range(num_clients)]
    labels = np.unique(y)
    
    print(f"[Partition] Dirichlet α={alpha}, {num_clients} clients, {len(labels)} classes")

    for label in labels:
        idx_label = np.where(y == label)[0]
        np.random.shuffle(idx_label)
        
        # Dirichlet proportions
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        
        # Convert to counts with minimum guarantee
        total_label_samples = len(idx_label)
        min_total = min_samples * num_clients
        
        if total_label_samples < min_total:
            # Not enough samples: distribute evenly with minimum
            counts = np.full(num_clients, max(1, total_label_samples // num_clients))
        else:
            # Dirichlet allocation with floor
            counts = (proportions * total_label_samples).astype(int)
            # Ensure minimum
            counts = np.maximum(counts, min_samples // len(labels))
            # Adjust to not exceed total
            while counts.sum() > total_label_samples:
                counts[np.argmax(counts)] -= 1
        
        # Assign indices
        start = 0
        for i, cnt in enumerate(counts):
            end = min(start + cnt, len(idx_label))
            if start < end:
                data_per_client[i].extend(idx_label[start:end])
            start = end
            if start >= len(idx_label):
                break

    # Build client datasets with validation
    client_data = []
    empty_clients = 0
    
    for i, indices in enumerate(data_per_client):
        if len(indices) >= min_samples:
            X_c = X[indices]
            y_c = y[indices]
            client_data.append((X_c, y_c))
        else:
            # Fill with random samples from other clients
            empty_clients += 1
            all_indices = np.concatenate([data_per_client[j] for j in range(num_clients) if j != i])
            if len(all_indices) >= min_samples:
                selected = np.random.choice(all_indices, size=min_samples, replace=False)
                client_data.append((X[selected], y[selected]))
            else:
                # Fallback to global random
                selected = np.random.choice(len(X), size=min_samples, replace=False)
                client_data.append((X[selected], y[selected]))
    
    if empty_clients > 0:
        print(f"[Partition] Filled {empty_clients} clients with minimum samples")
    
    # Print distribution stats
    sample_counts = [len(d[0]) for d in client_data]
    print(f"[Partition] Samples per client: min={min(sample_counts)}, "
          f"max={max(sample_counts)}, mean={np.mean(sample_counts):.1f}")
    
    return client_data


# =========================
# Per-Client Data Loading (Enhanced)
# =========================
def get_client_data(client_id, num_clients, non_iid=True, alpha=0.5, seed=42, 
                    return_val=False):
    """
    Enhanced client data loading with:
    - Consistent seeding for reproducibility
    - Data normalization
    - Optional validation split
    - Minimum sample guarantees
    
    Args:
        return_val: If True, also return validation set
    """
    global cached_data, cached_scaler
    
    # Load dataset
    if cached_data is None:
        print("Loading WISDM dataset...")
        df = pd.read_csv(DATA_PATH)
        X, y, label_encoder = create_windows(df)
        cached_data = (X, y, label_encoder)
    else:
        X, y, _ = cached_data
    
    # Partition
    if non_iid:
        # Use client_id to create varied but reproducible splits
        partition_seed = seed + client_id if seed else None
        clients = dirichlet_split(X, y, num_clients, alpha=alpha, seed=partition_seed)
    else:
        # IID with deterministic seed
        np.random.seed(seed)
        total_samples = len(X)
        per_client = total_samples // num_clients
        clients = []
        for i in range(num_clients):
            start = i * per_client
            end = start + per_client if i < num_clients - 1 else total_samples
            clients.append((X[start:end], y[start:end]))
    
    # Handle client_id overflow
    if client_id >= len(clients):
        client_id = client_id % len(clients)
    
    X_client, y_client = clients[client_id]
    
    # Ensure minimum samples
    if len(X_client) < MIN_SAMPLES_PER_CLIENT:
        print(f"[Client {client_id}] Warning: only {len(X_client)} samples, augmenting...")
        # Duplicate with small noise
        needed = MIN_SAMPLES_PER_CLIENT - len(X_client)
        indices = np.random.choice(len(X_client), size=needed, replace=True)
        X_aug = X_client[indices] + np.random.normal(0, 0.01, size=(needed, WINDOW_SIZE, 3))
        y_aug = y_client[indices]
        X_client = np.concatenate([X_client, X_aug])
        y_client = np.concatenate([y_client, y_aug])
    
    # Split: train/val/test
    if return_val:
        # Three-way split
        x_temp, x_test, y_temp, y_test = train_test_split(
            X_client, y_client, test_size=TEST_SIZE, shuffle=True, 
            random_state=client_id, stratify=y_client if len(np.unique(y_client)) > 1 else None
        )
        val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=val_size_adjusted, shuffle=True,
            random_state=client_id, stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        # Normalize
        x_train, x_test = normalize_data(x_train, x_test, fit_scaler=True)
        _, x_val = normalize_data(x_val, x_val, fit_scaler=False)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        # Two-way split (original behavior)
        x_train, x_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=TEST_SIZE, shuffle=True,
            random_state=client_id, stratify=y_client if len(np.unique(y_client)) > 1 else None
        )
        
        # Normalize
        x_train, x_test = normalize_data(x_train, x_test, fit_scaler=True)
        
        return x_train, y_train, x_test, y_test


# =========================
# DP-Compatible Model (Reduced Size)
# =========================
def load_model(input_shape=(100, 3), num_classes=None, for_dp=False):
    """
    Create CNN model for HAR.
    Args:
        for_dp: If True, use smaller model suitable for DP training
               (fewer parameters = less noise needed)
    """
    if num_classes is None:
        num_classes = NUM_CLASSES or 6

    if for_dp:
        # SMALL model for DP: ~50K parameters vs ~200K in large model
        # Less noise needed per parameter update
        model = Sequential([
            Input(shape=input_shape),
            
            # Lightweight feature extraction
            layers.Conv1D(16, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.1),
            
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Lower learning rate for stability with DP noise
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        print(f"[Model] DP-optimized: ~{model.count_params()/1000:.1f}K parameters")
        
    else:
        # STANDARD model (your current version, slightly improved)
        model = Sequential([
            Input(shape=input_shape),
            
            layers.Conv1D(32, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),

            layers.Conv1D(64, 3, activation='relu'),
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
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
        print(f"[Model] Standard: ~{model.count_params()/1000:.1f}K parameters")

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =========================
# Utility: Model Size Check
# =========================
def get_model_size(model):
    """Calculate model size in bytes for network simulation."""
    total_params = model.count_params()
    # Assume float32 (4 bytes per parameter)
    return total_params * 4


# =========================
# Quick Test
# =========================
if __name__ == "__main__":
    print("Testing task.py...")
    
    # Test data loading
    x_train, y_train, x_test, y_test = get_client_data(0, 30, non_iid=True, alpha=0.5)
    print(f"\nClient 0: {len(x_train)} train, {len(x_test)} test")
    print(f"Data range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    
    # Test model creation
    model_small = load_model(for_dp=True)
    model_large = load_model(for_dp=False)
    
    print(f"\nSmall model size: {get_model_size(model_small)/1e6:.2f} MB")
    print(f"Large model size: {get_model_size(model_large)/1e6:.2f} MB")
    
    # Quick training test
    print("\nQuick training test (3 epochs)...")
    history = model_small.fit(x_train[:100], y_train[:100], epochs=3, verbose=1)
    print(f"Final accuracy: {history.history['accuracy'][-1]:.3f}")