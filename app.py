import streamlit as st
import numpy as np
import pandas as pd
import pywt
import joblib
from kymatio.numpy import Scattering1D

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Virtual Flow Meter", layout="centered")

# --- CUSTOM CSS FOR CTECH BRANDING ---
st.title("💧 Virtual Flow Meter: Leak Severity Estimator")
st.markdown("Upload raw non-invasive Accelerometer data (CSV) or Hydrophone data (RAW) to predict leakage severity.")

# --- LOAD MODELS ---
@st.cache_resource
def load_system():
    # Note: If you process hydrophone data, ensure you are loading the hydrophone SVM 
    # and setting fs=8000 and J=6 to match your training script later!
    svm_model = joblib.load('svm_accelerometer_model.joblib')
    scaler = joblib.load('scaler.joblib')
    fs = 51200
    scattering = Scattering1D(J=9, shape=(fs,), Q=1)
    return svm_model, scaler, scattering, fs

try:
    svm, scaler, wst, fs = load_system()
except FileNotFoundError:
    st.error("Missing model files. Run train_and_export.py first.")
    st.stop()

# --- UPLOADER ---
uploaded_file = st.file_uploader("Upload Sensor Data", type=['csv', 'raw'])

if uploaded_file is not None:
    signal = None

    # --- 1. PARSE FILE BY TYPE ---
    if uploaded_file.name.endswith('.csv'):
        # Handle Accelerometer / Dynamic Pressure Sensor data
        df = pd.read_csv(uploaded_file)
        
        # Smart Column Detection
        target_col = 0
        if df.shape[1] > 1:
            for i in range(df.shape[1]):
                if not (np.all(np.diff(df.iloc[:, i]) >= 0) or np.all(np.diff(df.iloc[:, i]) <= 0)):
                    target_col = i
                    break
            if target_col == 0 and np.all(np.diff(df.iloc[:, 0]) >= 0): 
                target_col = -1
                
        st.write(f"**Analyzing Signal Column:** `{df.columns[target_col]}`")
        signal = df.iloc[:, target_col].values
        
        # Plot the actual vibration signal
        st.line_chart(df.iloc[:2000, target_col])

    elif uploaded_file.name.endswith('.raw'):
        # Handle Hydrophone binary data
        bytes_data = uploaded_file.read()
        signal = np.frombuffer(bytes_data, dtype=np.float32)
        
        st.write("**Analyzing Hydrophone Binary Signal**")
        # Plot the first 2000 data points of the numpy array
        st.line_chart(signal[:2000])

    else:
        st.error("Unsupported file format.")
        st.stop()

    # --- 2. SANITIZE SIGNAL ---
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 3. DIAGNOSTICS EXECUTION ---
    if st.button("Run Diagnostics", type="primary"):
        with st.spinner("Demodulating structural vibrations via WST..."):
            
            # Check if there is enough data
            if len(signal) < fs:
                st.error("Data too short. Need at least 1 second of data.")
                st.stop()
                
            # Extract a 1-second segment
            start_idx = fs * 5 
            end_idx = fs * 6
            if len(signal) < end_idx:
                # Fallback to the first second if the file is very short
                seg = signal[:fs]
            else:
                seg = signal[start_idx:end_idx]
                
            seg = seg - np.mean(seg)
            
            # Wavelet Denoising
            coeffs = pywt.wavedec(seg, 'sym3', level=4)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            if sigma == 0: sigma = 1e-6
            thresh = sigma * np.sqrt(2 * np.log(len(seg)))
            new_coeffs = [pywt.threshold(c, thresh, 'soft') for c in coeffs]
            clean_seg = pywt.waverec(new_coeffs, 'sym3')
            
            if len(clean_seg) > len(seg): 
                clean_seg = clean_seg[:len(seg)]
            
            # WST Feature Extraction
            Sx = wst(clean_seg)
            Sx = Sx[1:, :] 
            Sx_log = np.log(np.abs(Sx) + 1e-6)
            features = np.mean(Sx_log, axis=-1).reshape(1, -1)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale and Predict
            features_scaled = scaler.transform(features)
            prediction = svm.predict(features_scaled)[0]
            
            # --- OUTPUT ---
            st.divider()
            if prediction == 0:
                st.success("🟢 System Status: No Demand (ND) - Normal Operation")
            elif prediction == 1:
                st.warning("🟡 Leak Detected: Low Severity - Schedule Maintenance")
            elif prediction == 2:
                st.error("🔴 CRITICAL LEAK: High Severity - Immediate Dispatch Required")