import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Premature Ventricular (V)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)"
}

CLASS_SHORT_NAMES = ["N", "S", "V", "F", "Q"]

def generate_gaussian_ecg_beat(label: int, length: int = 187) -> np.ndarray:
    """
    Generates a physiologically plausible 1D ECG heartbeat of length 187
    using a sum of Gaussian waves (P, Q, R, S, T waves) with class-specific morphologies.
    """
    t = np.linspace(0, 1.0, length)
    signal = np.zeros(length)
    
    # Random variation / physiological jitter
    jitter = np.random.normal(0, 0.015)
    amp_jitter = np.random.uniform(0.9, 1.1)

    if label == 0:
        # Class 0: Normal Beat (N)
        # P-wave
        signal += 0.15 * np.exp(-((t - (0.20 + jitter)) ** 2) / (2 * (0.03 ** 2)))
        # Q-wave (small negative)
        signal -= 0.12 * np.exp(-((t - (0.35 + jitter)) ** 2) / (2 * (0.012 ** 2)))
        # R-wave (sharp tall positive)
        signal += 1.00 * np.exp(-((t - (0.38 + jitter)) ** 2) / (2 * (0.016 ** 2)))
        # S-wave (negative dip)
        signal -= 0.28 * np.exp(-((t - (0.42 + jitter)) ** 2) / (2 * (0.014 ** 2)))
        # T-wave (smooth positive)
        signal += 0.35 * np.exp(-((t - (0.60 + jitter)) ** 2) / (2 * (0.055 ** 2)))
        
    elif label == 1:
        # Class 1: Supraventricular Premature Beat (S)
        # Premature timing, abnormal P wave
        signal += 0.08 * np.exp(-((t - (0.16 + jitter)) ** 2) / (2 * (0.025 ** 2)))
        signal -= 0.10 * np.exp(-((t - (0.28 + jitter)) ** 2) / (2 * (0.012 ** 2)))
        signal += 0.85 * np.exp(-((t - (0.31 + jitter)) ** 2) / (2 * (0.015 ** 2)))
        signal -= 0.22 * np.exp(-((t - (0.35 + jitter)) ** 2) / (2 * (0.014 ** 2)))
        signal += 0.30 * np.exp(-((t - (0.50 + jitter)) ** 2) / (2 * (0.050 ** 2)))
        
    elif label == 2:
        # Class 2: Premature Ventricular Contraction (V)
        # No P wave, wide bizarre QRS complex, inverted discordant T wave
        signal -= 0.15 * np.exp(-((t - (0.25 + jitter)) ** 2) / (2 * (0.03 ** 2)))
        signal += 0.90 * np.exp(-((t - (0.34 + jitter)) ** 2) / (2 * (0.045 ** 2)))
        signal -= 0.65 * np.exp(-((t - (0.42 + jitter)) ** 2) / (2 * (0.035 ** 2)))
        signal -= 0.40 * np.exp(-((t - (0.62 + jitter)) ** 2) / (2 * (0.060 ** 2)))
        
    elif label == 3:
        # Class 3: Fusion Beat (F)
        # Hybrid between Normal and PVC
        signal += 0.08 * np.exp(-((t - (0.18 + jitter)) ** 2) / (2 * (0.03 ** 2)))
        signal -= 0.12 * np.exp(-((t - (0.32 + jitter)) ** 2) / (2 * (0.02 ** 2)))
        signal += 0.75 * np.exp(-((t - (0.36 + jitter)) ** 2) / (2 * (0.030 ** 2)))
        signal -= 0.40 * np.exp(-((t - (0.43 + jitter)) ** 2) / (2 * (0.025 ** 2)))
        signal += 0.15 * np.exp(-((t - (0.58 + jitter)) ** 2) / (2 * (0.050 ** 2)))
        
    elif label == 4:
        # Class 4: Unclassifiable / Paced Beat (Q)
        # Sharp artificial pacing spike followed by wide complex
        signal += 0.95 * np.exp(-((t - (0.22 + jitter)) ** 2) / (2 * (0.004 ** 2))) # spike
        signal -= 0.30 * np.exp(-((t - (0.27 + jitter)) ** 2) / (2 * (0.015 ** 2)))
        signal += 0.70 * np.exp(-((t - (0.35 + jitter)) ** 2) / (2 * (0.035 ** 2)))
        signal -= 0.50 * np.exp(-((t - (0.44 + jitter)) ** 2) / (2 * (0.030 ** 2)))
        signal += 0.25 * np.exp(-((t - (0.65 + jitter)) ** 2) / (2 * (0.045 ** 2)))

    # Apply amplitude jitter and physiological baseline noise
    signal = signal * amp_jitter
    noise = np.random.normal(0, 0.012, length)
    signal = signal + noise
    
    # Zero-pad tail like standard MIT-BIH dataset (segments after the T-wave fade to 0)
    tail_fade = np.clip((t - 0.75) / 0.15, 0, 1)
    signal = signal * (1.0 - tail_fade)
    
    # Normalize to [0, 1] range as in MIT-BIH
    s_min = signal.min()
    s_max = signal.max()
    if s_max - s_min > 1e-6:
        signal = (signal - s_min) / (s_max - s_min)
    else:
        signal = np.zeros(length)
        
    return signal.astype(np.float32)

def generate_synthetic_dataset(n_samples: int = 1000, imbalanced: bool = True):
    """
    Generates a full synthetic dataset with MIT-BIH class imbalance:
    N: ~82.7%, S: ~2.5%, V: ~6.6%, F: ~0.7%, Q: ~7.5%
    """
    if imbalanced:
        # Proportions matching MIT-BIH real distribution
        ratios = [0.827, 0.025, 0.066, 0.007, 0.075]
    else:
        ratios = [0.2, 0.2, 0.2, 0.2, 0.2]
        
    counts = [max(10, int(r * n_samples)) for r in ratios]
    
    X_list = []
    y_list = []
    for cls_idx, count in enumerate(counts):
        for _ in range(count):
            beat = generate_gaussian_ecg_beat(cls_idx)
            X_list.append(beat)
            y_list.append(cls_idx)
            
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (N, 1, L)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_ecg_data(data_dir: str = "data", fast: bool = False):
    """
    Loads ECG dataset from data_dir/mitbih_train.csv and data_dir/mitbih_test.csv.
    If files are not present or fast=True, uses synthetic Gaussian ECG fallback.
    Returns: (train_dataset, test_dataset, is_synthetic)
    """
    train_path = os.path.join(data_dir, "mitbih_train.csv")
    test_path = os.path.join(data_dir, "mitbih_test.csv")
    
    use_synthetic = fast or not (os.path.exists(train_path) and os.path.exists(test_path))
    
    if not use_synthetic:
        try:
            print(f"[Data] Loading real MIT-BIH dataset from {train_path} and {test_path}...")
            train_df = pd.read_csv(train_path, header=None)
            test_df = pd.read_csv(test_path, header=None)
            
            X_train = train_df.iloc[:, :-1].values.astype(np.float32)
            y_train = train_df.iloc[:, -1].values.astype(np.int64)
            
            X_test = test_df.iloc[:, :-1].values.astype(np.float32)
            y_test = test_df.iloc[:, -1].values.astype(np.int64)
            
            print(f"[Data] Loaded real data: Train {X_train.shape}, Test {X_test.shape}")
            return ECGDataset(X_train, y_train), ECGDataset(X_test, y_test), False
        except Exception as e:
            print(f"[Data] Warning: Error reading CSVs ({e}). Falling back to synthetic ECG data.")
            use_synthetic = True
            
    # Synthetic Gaussian ECG Fallback
    train_size = 1200 if fast else 5000
    test_size = 300 if fast else 1000
    print(f"[Data] Generating synthetic Gaussian ECG fallback dataset (Train: {train_size}, Test: {test_size})...")
    X_train, y_train = generate_synthetic_dataset(train_size, imbalanced=True)
    X_test, y_test = generate_synthetic_dataset(test_size, imbalanced=True)
    
    return ECGDataset(X_train, y_train), ECGDataset(X_test, y_test), True
