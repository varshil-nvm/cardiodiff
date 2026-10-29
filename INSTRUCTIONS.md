# GenLab Instructions

## Overview
GenLab is an interactive, unified playground built with Streamlit and PyTorch for exploring fundamental concepts in Generative AI. It provides real-time execution, model comparison, self-attention tracing, and prompt analysis.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```
Or using Makefile:
```bash
make run
```

## Features

### 1. Data Pipeline
- Visualize 2D toy datasets (moons, circles, blobs)
- Normalized to [-1, 1] range for neural network training

### 2. Variational Autoencoder (VAE)
- Continuous VAE for learning latent representations
- Adjustable β parameter for KL-divergence scaling
- Visualize loss trajectory and generated samples

### 3. Generative Adversarial Networks (GAN)
- Generator vs Discriminator adversarial training
- Track both losses over time
- Visualize generated samples

### 4. Diffusion Process (DDPM)
- Denoising Diffusion Probabilistic Model
- Linear noise scheduler
- Forward noise injection and reverse denoising
- Time-conditioned noise prediction

### 5. Self-Attention Visualization
- Scaled dot-product multi-head self-attention
- Interactive heatmap visualization
- Adjustable number of attention heads

### 6. Prompt Engineering
- Test different prompt templates (direct, few-shot, roleplay)
- Mock LLM inference with temperature control
- Observe how prompt structure affects responses

## Architecture

```
genlab/
├── app.py                # Main Streamlit UI
├── toy_data.py           # Synthetic data generation
├── prompt_engine.py      # Prompt templating & mock inference
├── requirements.txt      # Dependencies
├── Makefile             # Build/run commands
└── models/
    ├── vae.py           # VAE implementation
    ├── gan.py           # GAN implementation
    ├── diffusion.py     # DDPM implementation
    └── attention.py     # Self-attention implementation
```

## Development

### Environment Setup
Use Python 3.8+. Install dependencies from requirements.txt.

### Available Commands (Makefile)
- `make install` - Install dependencies
- `make run` - Run the Streamlit app
- `make dev` - Run with auto-reload on save
- `make clean` - Remove pycache files
- `make test` - Run tests (if any)

## Deployment
See DEPLOY.md for Streamlit Cloud deployment instructions.

## Notes
- All models run on CPU by default
- Training is interactive through the UI
- Visualizations powered by Plotly
- Uses PyTorch for model implementations
