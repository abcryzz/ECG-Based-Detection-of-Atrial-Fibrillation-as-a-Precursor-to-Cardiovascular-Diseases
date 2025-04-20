
import numpy as np
import pandas as pd

def preprocess_ecg_data(source):
    """
    Load & preprocess ECG CSV (5000×12) exactly as in training:
      • Normalize per-channel: mean/std over (batch, timesteps)
      • Return np.ndarray shape (1, 5000, 12), dtype float32
    """
    # 1. Load CSV (skip header, take first 12 cols)
    if isinstance(source, (str, bytes)) or hasattr(source, 'read'):
        df = pd.read_csv(source, header=0, usecols=range(12))
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError(f"Expected file path or DataFrame, got {type(source)}")

    # 2. Validate shape
    if df.shape != (5000, 12):
        raise ValueError(f"Data must be 5000×12, got {df.shape}")

    # 3. To NumPy float32
    arr = df.values.astype(np.float32)

    # 4. Transpose & add batch → (1, 12, 5000)
    arr = arr.T                # (12, 5000)
    arr = np.expand_dims(arr,  axis=0)    # (1,12,5000) :contentReference[oaicite:0]{index=0}

    # 5. Per‑channel mean/std across batch & time dims
    mean = arr.mean(axis=(0, 2), keepdims=True)  # shape (1,12,1) :contentReference[oaicite:1]{index=1}
    std  = arr.std (axis=(0, 2), keepdims=True)  # shape (1,12,1) :contentReference[oaicite:2]{index=2}
    if np.any(std == 0):
        raise ValueError("Zero std on at least one channel; cannot normalize.")
    arr = (arr - mean) / std

    # 6. Permute to (batch, seq_len, channels) → (1, 5000, 12)
    arr = arr.transpose(0, 2, 1)

    return arr
