# 🫀 CardioDiff: 1D Denoising Diffusion for ECG Synthesis & Imbalanced Arrhythmia Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cardiodiff.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/varshil-nvm/cardiodiff)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Live Interactive Application:** [https://cardiodiff.streamlit.app/](https://cardiodiff.streamlit.app/)  
> **Full Technical Report:** [REPORT.md](REPORT.md)

---

## 📌 Overview

Cardiovascular disease diagnosis relies heavily on Electrocardiogram (ECG) heartbeat classification. However, clinical datasets such as the benchmark **MIT-BIH Arrhythmia Database** suffer from severe class imbalance:
- **Class 0 (Normal Sinus Rhythm, N):** ~82.8% of recordings
- **Class 1 (Supraventricular Ectopic Beat, S):** ~2.5%
- **Class 2 (Premature Ventricular Contraction, V):** ~6.6%
- **Class 3 (Fusion Beat, F):** ~0.7% (< 1,000 beats total)
- **Class 4 (Paced / Unknown Beat, Q):** ~7.4%

Standard classifiers trained on raw data severely underperform on life-threatening minority arrhythmias. Traditional oversampling techniques like Random Oversampling (ROS) induce severe overfitting.

**CardioDiff** addresses this challenge by introducing a **Conditional 1D Denoising Diffusion Probabilistic Model (DDPM)** engineered for 125 Hz ECG waveforms (187 points). CardioDiff learns the continuous electrophysiological manifold of minority classes and synthesizes realistic, diverse heartbeats to augment clinical classifiers.

---

## 📊 Key Results (Evaluated on 21,892 Real MIT-BIH Test Beats)

| Metric | Baseline (Imbalanced) | Random Oversampling (ROS) | **CardioDiff DDPM (Ours)** | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 94.46% | 93.04% | **96.90%** | **+2.44%** |
| **Macro-F1 Score** | 0.6554 | 0.7300 | **0.8210** | **+16.56%** |
| **Minority Recall (Classes 1–4)** | 0.4898 | 0.6387 | **0.7876** | **+29.78%** |
| **1D-FID Score** *(Fidelity)* | N/A | N/A | **40.0401** | *High morphological realism* |
| **Diversity Ratio** *(Ideal ~1.0)* | N/A | N/A | **1.0253** | *No mode collapse* |

### Per-Class Minority Recall Recovery:
- **Class 1 (Supraventricular, S):** 14.03% $\to$ **47.66%** ($+33.63\%$)
- **Class 2 (Ventricular, V):** 63.95% $\to$ **93.51%** ($+29.56\%$)
- **Class 3 (Fusion Beat, F):** 21.60% $\to$ **77.16%** ($+55.56\%$)
- **Class 4 (Paced / Unknown, Q):** 96.33% $\to$ **96.70%**

---

## 🖥️ Interactive Web Application (4 Tabs)

The Streamlit dashboard allows real-time interactive exploration:
1. **Tab 1: Generate** — Synthesize individual cardiac waveforms conditioned on any chosen arrhythmia class; view amplitude statistics, energy, and P-QRS-T wave intervals.
2. **Tab 2: Diffusion Lab** — Interactive step scrubber visualizing the forward noising process ($q(x_t|x_0)$) and the reverse denoising trajectory ($p_\theta(x_{t-1}|x_t)$) across timesteps.
3. **Tab 3: Results** — Side-by-side confusion matrices, classification metric comparisons, ablation study tables, and 1D-FID / Diversity metrics.
4. **Tab 4: Latent Space** — Real-time 2D PCA projections of penultimate 64-dim 1D-CNN feature embeddings comparing real and synthetic manifolds.

---

## 📁 Repository Structure

```text
cardiodiff/
├── app.py                      # Main 4-Tab Streamlit interactive dashboard
├── train_all.py                # End-to-end training orchestrator (Baseline + DDPM + Augmentation)
├── evaluate.py                 # Benchmarking, 1D-FID, Diversity Ratio, and plot generator
├── colab_train.md              # Google Colab T4 GPU execution guide
├── REPORT.md                   # Full academic technical report with LaTeX formulas
├── requirements.txt            # Python dependencies
├── models/
│   ├── ecg_data.py             # MIT-BIH dataset loader & Gaussian synthetic fallback
│   ├── classifier.py           # 1D-CNN classifier & penultimate feature extractor
│   ├── diffusion_1d.py         # Conditional 1D DDPM Denoiser & U-Net
│   └── metrics.py              # 1D-FID, Diversity Ratio, Macro-F1, and Minority Recall
├── weights/                    # Pretrained model checkpoints
│   ├── classifier_baseline.pt  # Raw imbalanced baseline 1D-CNN (253 KB)
│   ├── diffusion_ecg.pt        # Trained 1D DDPM denoiser (2.9 MB)
│   └── classifier_augmented.pt # Augmented 1D-CNN classifier (253 KB)
└── results/                    # Evaluation artifacts & figures
    ├── metrics.csv             # Primary quantitative comparison table
    ├── metrics.json            # Complete classification reports & matrices
    ├── confusion_matrix.png    # Side-by-side confusion matrix heatmaps
    ├── ecg_samples_grid.png    # Real vs. DDPM-synthesized morphological grid
    ├── fid_diversity_comparison.png # Generative quality bar charts
    └── latent_space_tsne.png   # PCA latent manifold projections
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/varshil-nvm/cardiodiff.git
cd cardiodiff
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fast Smoke Test (Built-in Gaussian Fallback, CPU friendly, ~1 min)
```bash
python train_all.py --fast
python evaluate.py --fast
```

### 3. Launch Local Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Full Training on MIT-BIH Dataset
Place `mitbih_train.csv` and `mitbih_test.csv` inside `data/`, then run:
```bash
python train_all.py
python evaluate.py
```

---

## 📖 Citation & License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
Dataset citation: PhysioNet / MIT-BIH Arrhythmia Database (Moody GB, Mark RG. IEEE EMB 2001).
