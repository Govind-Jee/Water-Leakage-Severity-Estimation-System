# Non-Invasive Water Leakage Severity Estimation System

[![Conference](https://img.shields.io/badge/Conference-ICFCR%202026-blue.svg)](presentation/ICFCR2026_Presentation.pdf)
[![Paper](https://img.shields.io/badge/Paper-PID%20105-success.svg)](paper/ICFCR2026_Paper105.pdf)
[![Presentation](https://img.shields.io/badge/Presentation-Slides%20PDF-orange.svg)](presentation/ICFCR2026_Presentation.pdf)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](app.py)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official open-source research implementation and diagnostic platform for ICFCR 2026 Paper 105:
> **"Non-Invasive Leakage Severity Estimation in Water Distribution Networks using Wavelet Scattering Transforms: A Comparative Sensor Study"**  
> *Shiju Kumar P S, Govind Jee, Raj Kishor*  
> *Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, India*  
> *International Conference on Futuristic Technologies in Control Systems & Renewable Energy (ICFCR 2026) · Paper ID 105*  
> *September 23–24, 2026 · MESCE Kuttippuram, Kerala, India*

---

## 📌 Executive Summary

Non-Revenue Water (NRW) due to pipe bursts and undetected micro-leaks results in catastrophic global resource and revenue losses. While conventional acoustic monitoring can detect the binary existence of a leak, **quantifying leak severity non-invasively** remains an elusive challenge due to non-stationary transient noise, environmental interference, and pipe boundary reflections.

This repository implements a machine learning system that transforms high-frequency surface vibration signals into accurate, discrete leak severity classifications:
1. **Non-Invasive Sensing**: Operates via pipe surface-mounted accelerometers without drilling into pressurized mains or risking water contamination.
2. **Physics-Informed Feature Demodulation**: Employs a 2nd-order **Wavelet Scattering Transform (WST)** using Morlet wavelets to yield translation-invariant, deformation-stable multiscale vibrational representations.
3. **Rigorous Signal Conditioning**: Denoises transient turbulence using Symlet-3 (`sym3`) discrete wavelet transform with Donoho universal soft thresholding.
4. **Virtual Flow Meter UI**: Includes an interactive **Streamlit** diagnostic web dashboard with pre-trained SVM model weights for accelerometer telemetry.

---

## 🔬 Benchmark & Scientific Findings

The methodology was evaluated on the benchmark published by Aghashahi, Sela, and Banks (2022) across three distinct sensor modalities operating under identical hydraulic scenarios on a branched testbed:
* **Accelerometer (Non-Invasive, Surface-Mounted)**: $f_s = 51,200\text{ Hz}$ (510 windowed segments)
* **Dynamic Pressure Transducer (Intrusive, Wetted)**: $f_s = 51,200\text{ Hz}$ (541 windowed segments)
* **Hydrophone (Invasive, Acoustic Submerged)**: $f_s = 8,000\text{ Hz}$ (900 windowed segments)

### Severity Classes Evaluated
* **Class 0 — No Demand (ND)**: Baseline intact pipe, $0.00\text{ L/s}$
* **Class 1 — Low Flow**: Minor orifice leak, $0.18\text{ L/s}$
* **Class 2 — High Flow**: Major burst orifice, $0.47\text{ L/s}$

---

## 📈 Validation Protocols & Performance Distinction

To ensure scientific honesty and methodological transparency, we explicitly distinguish between the different validation protocols evaluated:

### 1. 5-Fold Stratified Cross-Validation (Segment-Level, Paper & Presentation)
Evaluates classification performance across randomly partitioned 1-second signal segments within the experimental runs:

| Sensor Modality | Invasiveness | Sampling Rate ($f_s$) | CV Accuracy (Mean ± SD) | Macro F1-Score | Key Distinction |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Accelerometer** | **Non-Invasive** | **51,200 Hz** | **99.0% ± 1.2%** | **0.990** | **Optimal**: Captures shell pipe-wall shear and flexural vibration modes |
| **Dynamic Pressure** | Intrusive | 51,200 Hz | **93.7% ± 1.9%** | **0.954** | High accuracy; small number of misclassifications between adjacent flow rates |
| **Hydrophone** | Invasive | 8,000 Hz | **42.0% ± 1.7%** | **0.437** | Ineffective for severity estimation; acoustic reflections mask discrete flow rates |

### 2. Recording-Level Grouped Cross-Validation (Cross-Run Generalization)
Evaluated in the research audit by holding out entire 30-second parent recording sessions/runs (`GroupKFold` across independent recording files):
* **Accelerometer**: **64.00% ± 20.95%**
* **Dynamic Pressure**: **56.01% ± 23.53%**
* **Hydrophone**: **27.00% ± 5.95%**

> **Scientific Disclosure on Generalization:**  
> Grouped cross-validation measures consistency across distinct experimental acquisition runs within the benchmark testbed. **It does not represent evaluation on unseen physical pipe-network topologies**, as all evaluated runs were collected on the benchmark's branched pipe layout. The higher variance across recording runs reflects acoustic and vibrational variations between distinct acquisition sessions, emphasizing the importance of multi-point calibration in practical utility deployments.

### 3. Exploratory Grid Search & Held-Out Test Split
* **Exploratory Grid Search**: Peak exploratory accuracy on accelerometer WST features reached **99.2%** with $C=100.0, \gamma=1.0$.
* **Held-Out Test Evaluation (80/20)**: Implemented in `water_leak_classifier.py` with strict data-leakage prevention (`StandardScaler` fitted strictly on the training partition and evaluated on the held-out test partition).

---

## 📊 Visualizations & Research Figures

All 16 high-resolution figures from the research paper are available in [`assets/figures/`](assets/figures/):

| Accelerometer Confusion Matrix | Supervised Class Separation (LDA) | Sensor Modality Benchmark |
| :---: | :---: | :---: |
| ![CM Accelerometer](assets/figures/Fig_CM_Accelerometer.png) | ![LDA Accelerometer](assets/figures/Fig_LDA_Accelerometer.png) | ![Accuracy Comparison](assets/figures/Fig_Accuracy_Comparison.png) |
| *Held-out test evaluation (99.0% CV)* | *Supervised projection showing feature separability* | *Surface Accelerometer outperforms intrusive sensors* |

| Multiclass ROC Curve (Accelerometer) | Dynamic Pressure Confusion Matrix | Raw Accelerometer Waveforms |
| :---: | :---: | :---: |
| ![ROC Accelerometer](assets/figures/Fig_ROC_Accelerometer.png) | ![Pressure CM](assets/figures/Fig_CM_Dynamic_Pressure_Sensor.png) | ![Waveforms](assets/figures/Fig_Waveforms_Accelerometer.png) |
| *Macro-average AUC = 1.00* | *Pressure Sensor CM (93.7% CV)* | *Vibration response per severity level* |

*Additional PCA projections, LDA plots, and waveforms for Dynamic Pressure and Hydrophone sensors are cataloged in [`assets/figures/`](assets/figures/).*

---

## 💻 Virtual Flow Meter Diagnostic Dashboard

The repository includes a web-based diagnostic application built with Streamlit ([`app.py`](app.py)), enabling utility operators to upload accelerometer telemetry and obtain instant severity predictions.

> **Sensor Scope Notice:** The interactive web dashboard serves the **non-invasive Accelerometer vibration modality** ($f_s = 51,200\text{ Hz}$). Intrusive dynamic pressure and hydrophone models are offline research baselines evaluated in `water_leak_classifier.py` and are not loaded in the real-time UI.

| Dashboard Upload | Normal Operation (No Demand) |
| :---: | :---: |
| ![UI Upload](assets/screenshots/ui_dashboard_landing.png) | ![UI ND](assets/screenshots/ui_no_demand_normal.png) |
| **Low Leak Detected (0.18 L/s)** | **Critical Leak Detected (0.47 L/s)** |
| ![UI Low](assets/screenshots/ui_low_severity_leak.png) | ![UI High](assets/screenshots/ui_high_severity_critical.png) |

---

## 🛠️ Repository Architecture

The verified remote repository structure contains **38 items** under the `main` branch:

```text
Water-Leakage-Severity-Estimation-System/
├── LICENSE                                # MIT License with research assets notice
├── README.md                              # Comprehensive project documentation
├── app.py                                 # Streamlit Virtual Flow Meter application
├── requirements.txt                       # Pinned Python dependencies
├── train_and_export.py                    # Autonomous model retraining & export pipeline
├── water_leak_classifier.py               # Scientific pipeline (DWT + WST + SVM)
├── models/                                # Pre-trained production artifacts
│   ├── scaler.joblib                      # Fitted feature standardizer (52 features)
│   └── svm_accelerometer_model.joblib     # Pre-trained RBF-SVM classifier
├── paper/                                 # Research paper manuscript
│   └── ICFCR2026_Paper105.pdf             # ICFCR 2026 Paper 105 camera-ready PDF
├── presentation/                          # Conference slide deck
│   └── ICFCR2026_Presentation.pdf         # Official 15-slide conference presentation PDF
├── docs/                                  # Presentation delivery documentation
│   └── conference_presentation_speech.md  # 15-slide verbatim live delivery speech
└── assets/
    ├── screenshots/                       # Streamlit UI walkthrough captures
    │   ├── ui_dashboard_landing.png
    │   ├── ui_no_demand_normal.png
    │   ├── ui_low_severity_leak.png
    │   └── ui_high_severity_critical.png
    └── figures/                           # 16 High-resolution publication plots
        ├── Fig_Accuracy_Comparison.png
        ├── Fig_CM_Accelerometer.png
        ├── Fig_CM_Dynamic_Pressure_Sensor.png
        ├── Fig_CM_Hydrophone.png
        ├── Fig_LDA_Accelerometer.png
        ├── Fig_LDA_Dynamic_Pressure_Sensor.png
        ├── Fig_LDA_Hydrophone.png
        ├── Fig_PCA_Accelerometer.png
        ├── Fig_PCA_Dynamic_Pressure_Sensor.png
        ├── Fig_PCA_Hydrophone.png
        ├── Fig_ROC_Accelerometer.png
        ├── Fig_ROC_Dynamic_Pressure_Sensor.png
        ├── Fig_ROC_Hydrophone.png
        ├── Fig_Waveforms_Accelerometer.png
        ├── Fig_Waveforms_Dynamic_Pressure_Sensor.png
        └── Fig_Waveforms_Hydrophone.png
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/Govind-Jee/Water-Leakage-Severity-Estimation-System.git
cd Water-Leakage-Severity-Estimation-System

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Virtual Flow Meter (Web UI)
Launch the interactive diagnostic interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, upload an accelerometer vibration CSV recording ($f_s = 51,200\text{ Hz}$), and click **Run Severity Diagnostics**.

### 3. Run the Scientific Classification Pipeline
To execute the complete preprocessing, multiscale feature extraction, and evaluation pipeline:
```bash
python3 water_leak_classifier.py
```

### 4. Retrain and Export Model Weights
To retrain the model and export updated serialized `.joblib` weights into `models/`:
```bash
python3 train_and_export.py
```

---

## ⚙️ Algorithmic Formulation

### 1. DWT Denoising (Symlet-3)
Signals are decomposed into 4 levels of detail coefficients using the `sym3` wavelet:
$$\hat{\sigma} = \frac{\text{median}(|d_1|)}{0.6745}, \quad \lambda = \hat{\sigma}\sqrt{2\ln N}$$
Soft thresholding is applied to attenuate high-frequency measurement noise while preserving transient leak events.

### 2. Wavelet Scattering Transform (WST)
Features are computed using a cascade of Morlet wavelets $\psi_{\lambda}$ and modulus operators:
$$S_0 x(t) = x \ast \phi_J(t)$$
$$S_1 x(t, \lambda_1) = |x \ast \psi_{\lambda_1}| \ast \phi_J(t)$$
$$S_2 x(t, \lambda_1, \lambda_2) = ||x \ast \psi_{\lambda_1}| \ast \psi_{\lambda_2}| \ast \phi_J(t)$$
Invariance to spatial/temporal micro-shifts is guaranteed by the low-pass filter $\phi_J$. Second-order coefficients capture the multiscale modulation structure of leak-induced turbulence.

### 3. Support Vector Machine Classification
Features are standardized and classified using an RBF kernel:
$$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
Hyperparameters ($C=100.0, \gamma=1.0$) optimize the soft-margin decision boundaries in the demodulated scattering representation space.

---

## 📑 Citation

If you use this research, codebase, or pre-trained models, please cite our conference paper:

```bibtex
@inproceedings{kumar2026noninvasive,
  author    = {Shiju Kumar P S and Govind Jee and Raj Kishor},
  title     = {Non-Invasive Leakage Severity Estimation in Water Distribution Networks using Wavelet Scattering Transforms: A Comparative Sensor Study},
  booktitle = {International Conference on Futuristic Technologies in Control Systems \& Renewable Energy (ICFCR 2026)},
  year      = {2026},
  address   = {MESCE Kuttippuram, India},
  note      = {Paper ID 105}
}
```

---

## 📜 License

* **Codebase & Utilities:** The source code and diagnostic application are licensed under the [MIT License](LICENSE).
* **Research Publications & Materials:** The research manuscript (`paper/`), presentation slides (`presentation/`), and conference documentation (`docs/`) are copyright © 2026 by the authors (Shiju Kumar P S, Govind Jee, Raj Kishor) and ICFCR 2026. All rights reserved.
* **Benchmark Dataset:** Experimental measurements are from Aghashahi, Sela, and Banks (2022), distributed via Mendeley Data under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.
