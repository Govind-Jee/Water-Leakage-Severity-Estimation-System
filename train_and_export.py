"""
Model Training and Export Pipeline
==================================
Trains the RBF-SVM classifier on Wavelet Scattering Transform (WST)
features extracted from accelerometer vibration signals, and serializes
the pre-trained model and StandardScaler into the 'models/' directory.

Author: Govind Jee (ICFCR 2026, Paper ID 105)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import joblib
from kymatio.numpy import Scattering1D
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

CONFIG = {'Accelerometer': {'fs': 51200, 'J': 9, 'Q': 1}}
SEVERITY_CLASSES = {'No Demand (ND)': 0, 'Low Flow (0.18 L/s)': 1, 'High Flow (0.47 L/s)': 2}

def load_and_preprocess_data(root_dir, modality='Accelerometer', topology='Branched'):
    fs = CONFIG[modality]['fs']
    all_segments, all_labels = [], []
    search_path = os.path.join(root_dir, modality, topology)
    
    if not os.path.exists(search_path):
        print(f"Directory '{search_path}' not found. Please provide path to benchmark dataset.")
        return np.array([]), np.array([])
        
    for root, dirs, files in os.walk(search_path):
        for filename in files:
            if not filename.endswith(".csv"): 
                continue
            if "_NL_" in filename and "ND" not in filename:
                continue
                
            label = -1
            if "0.18" in filename: 
                label = SEVERITY_CLASSES['Low Flow (0.18 L/s)']
            elif "0.47" in filename: 
                label = SEVERITY_CLASSES['High Flow (0.47 L/s)']
            elif "ND" in filename: 
                label = SEVERITY_CLASSES['No Demand (ND)']
            
            if label != -1:
                try:
                    df = pd.read_csv(os.path.join(root, filename))
                    
                    target_col = 0
                    if df.shape[1] > 1:
                        for idx in range(df.shape[1]):
                            if not (np.all(np.diff(df.iloc[:, idx]) >= 0) or np.all(np.diff(df.iloc[:, idx]) <= 0)):
                                target_col = idx
                                break
                        if target_col == 0 and np.all(np.diff(df.iloc[:, 0]) >= 0): 
                            target_col = -1
                            
                    signal = df.iloc[:, target_col].values
                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

                    seg_len = int(fs * 1.0)
                    n_segs = min(len(signal) // seg_len, 30)
                    
                    for i in range(n_segs):
                        seg = signal[i*seg_len : (i+1)*seg_len]
                        seg = seg - np.mean(seg)
                        
                        # Wavelet Denoising (Symlet-3 soft thresholding)
                        coeffs = pywt.wavedec(seg, 'sym3', level=4)
                        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                        if sigma == 0: 
                            sigma = 1e-6
                        thresh = sigma * np.sqrt(2 * np.log(len(seg)))
                        new_coeffs = [pywt.threshold(c, thresh, 'soft') for c in coeffs]
                        clean_seg = pywt.waverec(new_coeffs, 'sym3')
                        
                        if len(clean_seg) > len(seg): 
                            clean_seg = clean_seg[:len(seg)]
                        all_segments.append(clean_seg)
                        all_labels.append(label)
                except Exception as e:
                    pass
                    
    return np.array(all_segments), np.array(all_labels)

def extract_wst_features(segments):
    J = CONFIG['Accelerometer']['J']
    Q = CONFIG['Accelerometer']['Q']
    scattering = Scattering1D(J=J, shape=(segments.shape[-1],), Q=Q)
    features = []
    for i in range(0, len(segments), 50):
        batch = segments[i:i+50]
        Sx = scattering(batch)
        Sx = Sx[:, 1:, :] 
        Sx_log = np.log(np.abs(Sx) + 1e-6)
        Sx_inv = np.mean(Sx_log, axis=-1)
        features.append(Sx_inv)
    return np.nan_to_num(np.vstack(features), nan=0.0, posinf=0.0, neginf=0.0)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Accelerometer Data...")
    X_raw, y = load_and_preprocess_data("./Data", 'Accelerometer', 'Branched')
    
    if len(X_raw) == 0:
        print("Dataset not detected at './Data'. Exiting without overwriting models.")
        exit(0)
        
    print(f"Extracting WST Features ({len(X_raw)} segments)...")
    X_feat = extract_wst_features(X_raw)
    
    print("Fitting Scaler and Training SVM...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    svm = SVC(kernel='rbf', C=100, gamma=1.0)
    svm.fit(X_scaled, y)
    
    # Save artifacts directly into models/ directory
    model_dest = MODELS_DIR / "svm_accelerometer_model.joblib"
    scaler_dest = MODELS_DIR / "scaler.joblib"
    
    joblib.dump(svm, model_dest)
    joblib.dump(scaler, scaler_dest)
    print(f"✅ Model saved to:  {model_dest}")
    print(f"✅ Scaler saved to: {scaler_dest}")
