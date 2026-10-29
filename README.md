# 🧠 GenLab: A Unified Generative AI Playground

GenLab is an interactive, unified playground built with Streamlit and PyTorch for exploring fundamental concepts in Generative AI. It allows real-time execution, model comparison, self-attention tracing, and prompt analysis.

## 📁 Project Structure
```text
genlab/
├── app.py                # Main Streamlit Multi-Tab UI dashboard
├── toy_data.py           # Synthetic 2D data generator
├── prompt_engine.py      # Rule-based mock LLM and prompt templater
├── requirements.txt      # Python dependencies
├── DEPLOY.md             # Detailed deployment steps
└── models/
    ├── vae.py            # Variational Autoencoder implementation
    ├── gan.py            # Generative Adversarial Network implementation
    ├── diffusion.py      # Denoising Diffusion Probabilistic Model (DDPM)
    └── attention.py      # Multi-Head Self-Attention layers
```

## 🚀 Quick Start (Local Run)
1. Ensure you have **Python 3.8+** installed.
2. Clone or extract this directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the application:
   ```bash
   streamlit run app.py
   ```
