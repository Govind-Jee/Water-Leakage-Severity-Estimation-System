import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from kymatio.numpy import Scattering1D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
CONFIG = {
    'Accelerometer': {'fs': 51200, 'J': 9, 'Q': 1},      
    'Dynamic Pressure Sensor': {'fs': 51200, 'J': 9, 'Q': 1}, 
    'Hydrophone': {'fs': 8000, 'J': 6, 'Q': 1}           
}

SEVERITY_CLASSES = {
    'No Demand (ND)': 0,
    'Low Flow (0.18 L/s)': 1,
    'High Flow (0.47 L/s)': 2
}

# def plot_paper_figures(X_test, y_test, y_pred, classes_dict):
#     """Generates 3 key figures including LDA proof."""
#     sorted_class_names = [k for k, v in sorted(classes_dict.items(), key=lambda item: item[1])]
    
#     # 1. Matrix
#     print("Generating Figure 1 (Confusion Matrix)...")
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(7, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=sorted_class_names, yticklabels=sorted_class_names,
#                 cbar=False, annot_kws={"size": 14})
#     plt.title('Confusion Matrix: Leak Severity', fontsize=14)
#     plt.ylabel('True Physical State')
#     plt.xlabel('Predicted Severity')
#     plt.tight_layout()
#     plt.savefig('Figure_Hydrophone_CM.png', dpi=300)
    
#     # 2. PCA
#     print("Generating Figure 2 (PCA)...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_test)
#     plt.figure(figsize=(9, 6))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
#     plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="Severity")
#     plt.title('PCA Visualization (Unsupervised)', fontsize=14)
#     plt.savefig('Figure2_Hydrophone_PCA_Clusters.png', dpi=300)

#     # 3. LDA (Proof)
#     print("Generating Figure 3 (LDA)... THIS IS YOUR PROOF.")
#     try:
#         lda = LinearDiscriminantAnalysis(n_components=2)
#         X_lda = lda.fit_transform(X_test, y_test)
#         plt.figure(figsize=(9, 6))
#         scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.9)
#         plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="Severity")
#         plt.title('LDA Visualization (Supervised Proof)', fontsize=14)
#         plt.savefig('Figure_Hydrophone_LDA.png', dpi=300)
#     except Exception as e:
#         print(f"Skipping LDA Plot (Data issue): {e}")

#     print("✅ All figures saved. Displaying plots...")
#     plt.show()
import os

def plot_paper_figures(X_test, y_test, y_pred, classes_dict):
    """Generates 3 key figures and saves them to the script's directory."""
    
    # 1. GET THE SCRIPT'S DIRECTORY
    # This ensures files save next to hydrophone.py, not wherever the terminal is pointing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    sorted_class_names = [k for k, v in sorted(classes_dict.items(), key=lambda item: item[1])]
    
    # --- FIGURE 1: CONFUSION MATRIX ---
    print("Generating Figure 1 (Confusion Matrix)...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted_class_names, yticklabels=sorted_class_names,
                cbar=False, annot_kws={"size": 14})
    plt.title('Confusion Matrix: Leak Severity (Hydrophone)', fontsize=14)
    plt.ylabel('True Physical State')
    plt.xlabel('Predicted Severity')
    plt.tight_layout()
    
    # Save to absolute path
    save_path = os.path.join(script_dir, 'Figure_Hydrophone_CM.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    
    # --- FIGURE 2: PCA ---
    print("Generating Figure 2 (PCA)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="Severity")
    plt.title('PCA Visualization (Hydrophone)', fontsize=14)
    
    save_path = os.path.join(script_dir, 'Figure2_Hydrophone_PCA_Clusters.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

    # --- FIGURE 3: LDA (Proof) ---
    print("Generating Figure 3 (LDA)...")
    try:
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_lda = lda.fit_transform(X_test, y_test)
        plt.figure(figsize=(9, 6))
        scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_test, cmap='viridis', s=60, edgecolors='k', alpha=0.9)
        plt.legend(handles=scatter.legend_elements()[0], labels=sorted_class_names, title="Severity")
        plt.title('LDA Visualization (Hydrophone Proof)', fontsize=14)
        
        save_path = os.path.join(script_dir, 'Figure_Hydrophone_LDA.png')
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Skipping LDA Plot (Data issue): {e}")

    print("✅ All figures saved. Displaying plots...")
    plt.show()

def load_and_preprocess_data(root_dir, modality, topology):
    fs = CONFIG[modality]['fs']
    all_segments = []
    all_labels = []
    
    search_path = os.path.join(root_dir, modality, topology)
    print(f"Scanning: {search_path}")

    for root, dirs, files in os.walk(search_path):
        for filename in files:
            if modality == 'Hydrophone' and ("_NN_" in filename or not filename.endswith(".raw")): continue
            if modality != 'Hydrophone' and not filename.endswith(".csv"): continue

            label = -1
            if "0.18" in filename: label = SEVERITY_CLASSES['Low Flow (0.18 L/s)']
            elif "0.47" in filename: label = SEVERITY_CLASSES['High Flow (0.47 L/s)']
            elif "ND" in filename: label = SEVERITY_CLASSES['No Demand (ND)']
            
            if label != -1:
                try:
                    file_path = os.path.join(root, filename)
                    # Load Data
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
                            if target_col == 0 and np.all(np.diff(df.iloc[:, 0]) >= 0): target_col = -1
                        signal = df.iloc[:, target_col].values

                    # --- CRITICAL FIX: CLEAN RAW DATA ---
                    # Replace NaN/Inf with 0 to prevent crashes
                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

                    # Segment
                    seg_len = int(fs * 1.0)
                    if len(signal) < seg_len: continue # Skip if too short
                    
                    n_segs = min(len(signal) // seg_len, 30)
                    
                    for i in range(n_segs):
                        seg = signal[i*seg_len : (i+1)*seg_len]
                        
                        # Double check segment cleanliness
                        if np.any(np.isnan(seg)) or np.any(np.isinf(seg)): continue
                        
                        seg = seg - np.mean(seg)
                        
                        coeffs = pywt.wavedec(seg, 'sym3', level=4)
                        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                        if sigma == 0: sigma = 1e-6 # Prevent div by zero
                        
                        thresh = sigma * np.sqrt(2 * np.log(len(seg)))
                        new_coeffs = [pywt.threshold(c, thresh, 'soft') for c in coeffs]
                        clean_seg = pywt.waverec(new_coeffs, 'sym3')
                        
                        if len(clean_seg) > len(seg): clean_seg = clean_seg[:len(seg)]
                        all_segments.append(clean_seg)
                        all_labels.append(label)
                except Exception as e:
                    # print(f"Error processing {filename}: {e}") 
                    pass

    return np.array(all_segments), np.array(all_labels)

def extract_wst_features(segments, modality):
    print(f"Extracting WST features...")
    J = CONFIG[modality]['J']
    scattering = Scattering1D(J=J, shape=(segments.shape[-1],), Q=1)
    
    features = []
    batch_size = 50
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        try:
            Sx = scattering(batch)
            Sx = Sx[:, 1:, :] 
            Sx_log = np.log(np.abs(Sx) + 1e-6)
            Sx_inv = np.mean(Sx_log, axis=-1)
            features.append(Sx_inv)
        except:
            pass
            
    # --- CRITICAL FIX: CLEAN FEATURES ---
    # Final check before SVM
    result = np.vstack(features)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def main():
    ROOT_DIR = "./Data"    
    
    # CHANGE THIS TO TEST DIFFERENT SENSORS
    MODALITY = 'Hydrophone'  # Options: 'Accelerometer', 'Dynamic Pressure Sensor', 'Hydrophone'
    TOPOLOGY = 'Branched'       

    print(f"--- STARTING SEVERITY CLASSIFICATION ({MODALITY}) ---")
    
    # 1. Load
    X_raw, y = load_and_preprocess_data(ROOT_DIR, MODALITY, TOPOLOGY)
    if len(X_raw) == 0:
        print("Error: No data loaded.")
        return
    print(f"Data Loaded: {len(X_raw)} segments.")

    # 2. Features
    X_feat = extract_wst_features(X_raw, MODALITY)
    
    # 3. Scale & Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    
    # --- CRITICAL FIX: Handle Infinity AND NaN created by Scaling ---
    # Hydrophones often create 'Divide by Zero' errors during scaling.
    # We force all bad values (NaN, Infinity) to 0.0
    print("Sanitizing data (Removing NaNs and Infinities)...")
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Train
    print("Training SVM...")
    # Check if data is clean one last time
    if not np.isfinite(X_train).all():
        print("Error: Training data still contains infinite values!")
        return
        
    svm = SVC(kernel='rbf', C=100, gamma=1)
    svm.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=SEVERITY_CLASSES.keys()))

    # 6. Plot Figures
    plot_paper_figures(X_test, y_test, y_pred, SEVERITY_CLASSES)

if __name__ == "__main__":
    main()