"""
Non-Invasive Water Leakage Severity Classification Pipeline
============================================================
Comparative sensor study implementation for ICFCR 2026 Paper 105:
"Non-Invasive Leakage Severity Estimation in Water Distribution Networks
using Wavelet Scattering Transforms: A Comparative Sensor Study"

Authors: Shiju Kumar P S, Govind Jee, Raj Kishor
Department of Computing Technologies, SRM Institute of Science and Technology

This script executes:
1. Signal windowing (1-second non-overlapping windows).
2. DWT wavelet denoising (Symlet-3 soft thresholding).
3. Wavelet Scattering Transform (WST) 2nd-order feature extraction.
4. Data-leakage-free train/test partitioning (scaler fitted strictly on training partition).
5. 5-Fold Stratified Cross-Validation with fold-isolated scaling pipeline.
6. Diagnostic visualization (Confusion Matrix, PCA, and Supervised LDA).
"""

import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from kymatio.numpy import Scattering1D
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- SENSOR CONFIGURATIONS ---
CONFIG = {
    'Accelerometer': {'fs': 51200, 'J': 9, 'Q': 1},      
    'Dynamic Pressure Sensor': {'fs': 51200, 'J': 9, 'Q': 1}, 
    'Hydrophone': {'fs': 8000, 'J': 6, 'Q': 1}           
}

# --- SEVERITY CLASSES ---
SEVERITY_CLASSES = {
    'No Demand (ND)': 0,
    'Low Flow (0.18 L/s)': 1,
    'High Flow (0.47 L/s)': 2
}

def load_and_preprocess_data(root_dir, modality, topology):
    """
    Scans directory for raw sensor recordings, segments them into 1-second windows,
    and applies Symlet-3 soft threshold wavelet denoising to each segment.
    """
    if modality not in CONFIG:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of {list(CONFIG.keys())}")
        
    fs = CONFIG[modality]['fs']
    all_segments = []
    all_labels = []
    
    search_path = os.path.join(root_dir, modality, topology)
    print(f"Scanning directory: {search_path}")

    if not os.path.exists(search_path):
        print(f"Warning: Directory '{search_path}' not found. Please provide path to benchmark dataset.")
        return np.array([]), np.array([])

    for root, dirs, files in os.walk(search_path):
        for filename in files:
            # Filtering logic per modality
            if modality == 'Hydrophone' and ("_NN_" in filename or not filename.endswith(".raw")): 
                continue
            if modality != 'Hydrophone' and not filename.endswith(".csv"): 
                continue

            # Severity label resolution
            label = -1
            if "0.18" in filename: 
                label = SEVERITY_CLASSES['Low Flow (0.18 L/s)']
            elif "0.47" in filename: 
                label = SEVERITY_CLASSES['High Flow (0.47 L/s)']
            elif "ND" in filename: 
                label = SEVERITY_CLASSES['No Demand (ND)']
            
            if label != -1:
                try:
                    file_path = os.path.join(root, filename)
                    if modality == 'Hydrophone':
                        signal = np.fromfile(file_path, dtype=np.float32)
                    else:
                        df = pd.read_csv(file_path)
                        target_col = 0
                        if df.shape[1] > 1:
                            for i in range(df.shape[1]):
                                if not (np.all(np.diff(df.iloc[:, i]) >= 0) or np.all(np.diff(df.iloc[:, i]) <= 0)):
                                    target_col = i
                                    break
                            if target_col == 0 and np.all(np.diff(df.iloc[:, 0]) >= 0): 
                                target_col = -1
                        signal = df.iloc[:, target_col].values

                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

                    # 1-second windowing
                    seg_len = int(fs * 1.0)
                    n_segs = min(len(signal) // seg_len, 30)
                    
                    for i in range(n_segs):
                        seg = signal[i*seg_len : (i+1)*seg_len]
                        seg = seg - np.mean(seg)
                        
                        # Discrete Wavelet Transform denoising (Symlet-3 soft threshold)
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

def extract_wst_features(segments, modality):
    """
    Extracts time-averaged log Wavelet Scattering Transform coefficients
    for each 1-second conditioned segment.
    """
    print(f"Extracting WST features for {modality} (segments: {len(segments)})...")
    J = CONFIG[modality]['J']
    Q = CONFIG[modality]['Q']
    scattering = Scattering1D(J=J, shape=(segments.shape[-1],), Q=Q)
    
    features = []
    batch_size = 50
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        Sx = scattering(batch)
        Sx = Sx[:, 1:, :]  # Exclude zeroth-order DC
        Sx_log = np.log(np.abs(Sx) + 1e-6)
        Sx_inv = np.mean(Sx_log, axis=-1)
        features.append(Sx_inv)
        
    return np.nan_to_num(np.vstack(features), nan=0.0, posinf=0.0, neginf=0.0)

def plot_paper_figures(X_train_scaled, y_train, X_test_scaled, y_test, y_pred, classes_dict, modality):
    """
    Generates publication figures:
    1. Confusion Matrix on held-out test split.
    2. Unsupervised PCA visualization (variance projection without labels).
    3. Supervised LDA visualization (class separability in feature space).
    """
    sorted_class_names = [k for k, v in sorted(classes_dict.items(), key=lambda item: item[1])]
    prefix = modality.replace(" ", "_")
    
    # --- 1. CONFUSION MATRIX ---
    print("Generating Figure 1: Confusion Matrix on held-out test data...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted_class_names, yticklabels=sorted_class_names,
                cbar=False, annot_kws={"size": 14})
    plt.title(f'Confusion Matrix ({modality}): Leak Severity', fontsize=13)
    plt.ylabel('True Physical State')
    plt.xlabel('Predicted Severity Class')
    plt.tight_layout()
    plt.savefig(f'Figure_{prefix}_CM.png', dpi=300)
    plt.close()
    
    # --- 2. PCA (Unsupervised) ---
    print("Generating Figure 2: Unsupervised PCA Projection...")
    pca = PCA(n_components=2)
    # Fit PCA on training data, project test data
    pca.fit(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="True Severity")
    plt.title(f'PCA Projection (Unsupervised) - {modality}\n(Captures primary variance axes without class labels)', fontsize=12)
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'Figure_{prefix}_PCA.png', dpi=300)
    plt.close()

    # --- 3. LDA (Supervised) ---
    print("Generating Figure 3: Supervised Linear Discriminant Projection (LDA)...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    # Fit LDA on training data, project test data to evaluate separability
    lda.fit(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.85)
    plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="True Severity")
    plt.title(f'Supervised Linear Discriminant Projection (LDA) - {modality}\n(Illustrates class separability in feature space)', fontsize=12)
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'Figure_{prefix}_LDA.png', dpi=300)
    plt.close()

    print(f"✅ Figures saved as Figure_{prefix}_CM.png, Figure_{prefix}_PCA.png, Figure_{prefix}_LDA.png.")

def main():
    ROOT_DIR = "./Data"    
    MODALITY = 'Accelerometer'
    TOPOLOGY = 'Branched'       

    print(f"=== Water Leakage Severity Classification ===")
    print(f"Modality: {MODALITY} | Topology: {TOPOLOGY}\n")
    
    # 1. Load & window data
    X_raw, y = load_and_preprocess_data(ROOT_DIR, MODALITY, TOPOLOGY)
    if len(X_raw) == 0:
        print("Note: Benchmark dataset not found locally. To evaluate with live data, place")
        print("the Mendeley benchmark in './Data' (see README for download instructions).")
        return

    print(f"Total windowed segments loaded: {len(X_raw)}")

    # 2. Extract WST features
    X_feat = extract_wst_features(X_raw, MODALITY)
    print(f"Extracted feature matrix shape: {X_feat.shape} (N_samples x N_features)")

    # 3. Train/Test Partitioning FIRST (Strict Data-Leakage Prevention)
    # Scaler is fitted exclusively on the training partition.
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.20, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Fit Classifier on Training Data Only
    print("\nTraining RBF Support Vector Classifier (C=100, gamma=1.0)...")
    svm = SVC(kernel='rbf', C=100, gamma=1.0)
    svm.fit(X_train_scaled, y_train)

    # 5. Evaluate on Held-out Test Data
    y_pred = svm.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-Out Test Split Accuracy (80/20): {test_acc*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=SEVERITY_CLASSES.keys()))

    # 6. Fold-Isolated 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation (fold-isolated scaling)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=100, gamma=1.0))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_feat, y, cv=cv, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # 7. Generate Diagnostic Visualizations
    plot_paper_figures(X_train_scaled, y_train, X_test_scaled, y_test, y_pred, SEVERITY_CLASSES, MODALITY)

if __name__ == "__main__":
    main()
