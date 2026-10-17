"""
Virtual Flow Meter: Non-Invasive Leak Severity Estimator
========================================================
Interactive diagnostic dashboard for non-invasive water leakage severity
estimation in water distribution networks using Wavelet Scattering Transform (WST)
and Support Vector Machines (SVM).

Author: Govind Jee (ICFCR 2026, Paper ID 105)
Supported Sensor Modality: Surface-mounted Accelerometer (Vibration, fs = 51,200 Hz)
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import pywt
import joblib
from kymatio.numpy import Scattering1D

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Virtual Flow Meter | Leak Severity Estimator",
    page_icon="💧",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# --- SYSTEM LOADER ---
@st.cache_resource
def load_system():
    """
    Locates and loads the pre-trained SVM model and StandardScaler.
    Resolves paths relative to the project directory with fallback support.
    """
    # Primary path: models/ directory
    model_path = MODELS_DIR / "svm_accelerometer_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    
    # Fallback path: application root
    if not model_path.exists():
        model_path = BASE_DIR / "svm_accelerometer_model.joblib"
    if not scaler_path.exists():
        scaler_path = BASE_DIR / "scaler.joblib"
        
    if not model_path.exists() or not scaler_path.exists():
        error_msg = (
            "Required model artifacts could not be found.\n\n"
            f"- Looked for model at: `{model_path}`\n"
            f"- Looked for scaler at: `{scaler_path}`\n\n"
            "Please ensure 'svm_accelerometer_model.joblib' and 'scaler.joblib' "
            "are present in the `models/` folder, or generate them using "
            "`python3 train_and_export.py`."
        )
        return None, None, None, 51200, error_msg

    try:
        svm_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        fs = 51200
        # WST configuration matching accelerometer training: J=9, Q=1
        scattering = Scattering1D(J=9, shape=(fs,), Q=1)
        return svm_model, scaler, scattering, fs, None
    except Exception as e:
        return None, None, None, 51200, f"Error loading model artifacts: {str(e)}"

# --- HEADER & BRANDING ---
st.title("💧 Virtual Flow Meter: Leak Severity Estimator")
st.caption("Non-Invasive Pipe-Wall Vibration Telemetry · ICFCR 2026 Paper 105")

# --- MODALITY DISCLOSURE ---
st.info(
    "**Sensor Modality Scope:** This interactive dashboard is calibrated exclusively for "
    "**non-invasive pipe surface accelerometer signals** ($f_s = 51,200\\text{ Hz}$). "
    "Intrusive dynamic pressure and submerged hydrophone acoustic modalities from our "
    "comparative study are evaluated offline in `water_leak_classifier.py` and are not "
    "served by this vibration-based proxy."
)

svm, scaler, wst, fs, load_error = load_system()

if load_error is not None:
    st.error(load_error)
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader(
    "Upload Accelerometer Vibration Data (CSV format)",
    type=['csv'],
    help="Provide a CSV file containing vibration acceleration readings sampled at 51,200 Hz."
)

if uploaded_file is not None:
    signal = None

    try:
        df = pd.read_csv(uploaded_file)
        
        # Smart Column Detection for time-series acceleration
        target_col = 0
        if df.shape[1] > 1:
            for i in range(df.shape[1]):
                # If column is strictly monotonic, it is likely a timestamp/index
                if not (np.all(np.diff(df.iloc[:, i]) >= 0) or np.all(np.diff(df.iloc[:, i]) <= 0)):
                    target_col = i
                    break
            if target_col == 0 and np.all(np.diff(df.iloc[:, 0]) >= 0): 
                target_col = -1
                
        col_name = df.columns[target_col]
        st.write(f"**Selected Telemetry Column:** `{col_name}`")
        signal = df.iloc[:, target_col].values
        
        # Preview first 2000 points of raw vibration
        st.subheader("Raw Signal Preview (First ~39 ms)")
        st.line_chart(df.iloc[:2000, target_col])

    except Exception as e:
        st.error(f"Failed to parse CSV file: {str(e)}")
        st.stop()

    # Sanitize signal
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # --- DIAGNOSTICS EXECUTION ---
    if st.button("Run Severity Diagnostics", type="primary"):
        with st.spinner("Conditioning vibration signal and computing Wavelet Scattering Transform..."):
            
            # Check length: at least 1 second of data required at 51.2 kHz
            if len(signal) < fs:
                st.error(
                    f"Insufficient signal duration: Provided {len(signal)} samples, but at least "
                    f"1 second of data ({fs} samples at 51.2 kHz) is required."
                )
                st.stop()
                
            # Extract 1-second analysis segment (default to 5th-6th second or first available second)
            start_idx = fs * 5 
            end_idx = fs * 6
            if len(signal) < end_idx:
                seg = signal[:fs]
            else:
                seg = signal[start_idx:end_idx]
                
            # Demean
            seg = seg - np.mean(seg)
            
            # DWT Wavelet Denoising (Symlet-3 soft thresholding)
            coeffs = pywt.wavedec(seg, 'sym3', level=4)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            if sigma == 0: 
                sigma = 1e-6
            thresh = sigma * np.sqrt(2 * np.log(len(seg)))
            new_coeffs = [pywt.threshold(c, thresh, 'soft') for c in coeffs]
            clean_seg = pywt.waverec(new_coeffs, 'sym3')
            
            if len(clean_seg) > len(seg): 
                clean_seg = clean_seg[:len(seg)]
            
            # WST Feature Extraction
            Sx = wst(clean_seg)
            Sx = Sx[1:, :]  # Exclude zeroth-order DC component
            Sx_log = np.log(np.abs(Sx) + 1e-6)
            features = np.mean(Sx_log, axis=-1).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Dimension compatibility check
            if features.shape[1] != scaler.n_features_in_:
                st.error(
                    f"Feature dimension mismatch: Extracted {features.shape[1]} WST coefficients, "
                    f"but pre-trained scaler expects {scaler.n_features_in_} features."
                )
                st.stop()
            
            # Standardize and Predict
            features_scaled = scaler.transform(features)
            prediction = svm.predict(features_scaled)[0]
            decision_vals = svm.decision_function(features_scaled)[0]
            
            # --- RESULTS DISPLAY ---
            st.divider()
            st.subheader("Diagnostic Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if prediction == 0:
                    st.success("### 🟢 Class 0: No Demand (Normal)")
                    st.markdown("**Equivalent Discharge:** `0.00 L/s` (Intact pipe)")
                    st.markdown("**Action:** Baseline hydraulic state. No maintenance required.")
                elif prediction == 1:
                    st.warning("### 🟡 Class 1: Low Severity Leak")
                    st.markdown("**Equivalent Discharge:** `~0.18 L/s` (Minor orifice leakage)")
                    st.markdown("**Action:** Incipient breach detected. Schedule non-emergency field inspection.")
                elif prediction == 2:
                    st.error("### 🔴 Class 2: High Severity Leak / Burst")
                    st.markdown("**Equivalent Discharge:** `~0.47 L/s` (Major pipe rupture / orifice)")
                    st.markdown("**Action:** High-priority leak alert. Immediate maintenance crew dispatch required.")

            with col2:
                st.metric("Model Architecture", "WST + RBF-SVM")
                st.metric("Sampling Rate", f"{fs:,} Hz")
                st.metric("Extracted Features", f"{features.shape[1]} coefficients")
                
            with st.expander("Diagnostic Decision Scores (One-vs-Rest Margins)"):
                st.write({
                    "No Demand Margin": float(decision_vals[0]),
                    "Low Severity Margin": float(decision_vals[1]),
                    "High Severity Margin": float(decision_vals[2]),
                })
