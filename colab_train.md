# Google Colab T4 GPU Training Guide: CardioDiff & MIT-BIH

This notebook guide allows you to train the **CardioDiff 1D Conditional Diffusion Model** and **ECG Arrhythmia Classifiers** on a free Google Colab T4 GPU in ~15-20 minutes.

---

### Step 1: Select GPU Runtime
In Google Colab:
1. Navigate to **Runtime** > **Change runtime type**.
2. Select **T4 GPU** as the Hardware accelerator.
3. Click **Save**.

Run this cell to verify GPU allocation:
```python
!nvidia-smi
```

---

### Step 2: Set Up Project Directory & Clone / Upload Code
Run the following cell to prepare the directory structure:
```bash
# Clone or upload your repository
!git clone <YOUR_GITHUB_REPO_URL> project || mkdir -p project
%cd project
!mkdir -p data weights results models
```

---

### Step 3: Install Dependencies
```bash
!pip install -q torch torchvision torchaudio
!pip install -q pandas numpy scipy scikit-learn matplotlib seaborn plotly streamlit tqdm
```

---

### Step 4: Download the Kaggle MIT-BIH Heartbeat Dataset
You can either upload your `kaggle.json` API token or directly download the dataset:

```python
import os

# Option A: Using Kaggle API (Recommended)
# Place kaggle.json in /root/.kaggle/
# from google.colab import files
# files.upload() # Upload kaggle.json
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d shayanfazeli/heartbeat
# !unzip -q heartbeat.zip -d data/

# Option B: Verify CSV files exist
assert os.path.exists("data/mitbih_train.csv"), "Place mitbih_train.csv in project/data/"
assert os.path.exists("data/mitbih_test.csv"), "Place mitbih_test.csv in project/data/"
print("MIT-BIH Dataset ready!")
```

---

### Step 5: Run Full Training
Run the complete training pipeline using CUDA acceleration:
```bash
!python train_all.py --device cuda --epochs_cls 15 --epochs_diff 25 --batch_size 64
```
*Expected training time: ~12-15 minutes on T4 GPU.*

---

### Step 6: Run Comprehensive Evaluation & Metric Computation
Compute Macro-F1, Minority Recall, 1D-FID, Diversity Ratio, and generate all result plots:
```bash
!python evaluate.py --device cuda
```

---

### Step 7: Package & Download Trained Artifacts
Package the trained checkpoints (`weights/*.pt`) and evaluation outputs (`results/*`) into `trained_artifacts.zip`:
```python
import zipfile
import os
from google.colab import files

with zipfile.ZipFile("trained_artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for folder in ["weights", "results"]:
        for root, dirs, files_list in os.walk(folder):
            for file in files_list:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file_path)

print("trained_artifacts.zip generated successfully!")
files.download("trained_artifacts.zip")
```

Once downloaded, extract `trained_artifacts.zip` into your local `project/` directory:
- Local `weights/` will be populated with `.pt` checkpoints.
- Local `results/` will be populated with `metrics.csv`, `metrics.json`, and `.png` plots.
