import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from models.ecg_data import load_ecg_data, CLASS_NAMES
from models.classifier import ECG1DCNN
from models.diffusion_1d import ECG1DDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D-CNN Classifier and Conditional DDPM for ECG Synthesis")
    parser.add_argument("--fast", action="store_true", help="Fast smoke test mode (CPU friendly, ~1-2 minutes)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset CSVs")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--epochs_cls", type=int, default=None, help="Epochs for classifier")
    parser.add_argument("--epochs_diff", type=int, default=None, help="Epochs for diffusion model")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    return parser.parse_args()

def train_classifier(model, train_loader, epochs, lr, device, desc="Classifier"):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Training {desc} ({epochs} Epochs) ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 3) == 0:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")
            
    print(f"Completed {desc} training in {time.time() - start_time:.1f}s")
    return model

def train_diffusion_model(diffusion, train_loader, epochs, lr, device):
    diffusion.to(device)
    diffusion.train()
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n--- Training Conditional 1D DDPM ({epochs} Epochs, {diffusion.timesteps} Timesteps) ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            
        avg_loss = total_loss / max(1, total_samples)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 3) == 0:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Diffusion MSE Loss: {avg_loss:.5f}")
            
    print(f"Completed DDPM training in {time.time() - start_time:.1f}s")
    return diffusion

def generate_minority_augmentations(diffusion, device, samples_per_class: int = 50):
    """
    Generates synthetic samples for minority arrhythmia classes (1, 2, 3, 4).
    """
    print(f"\n[Augment] Generating {samples_per_class} synthetic beats per minority class (1..4) with DDPM...")
    aug_x = []
    aug_y = []
    
    for cls in range(1, 5):
        samples = diffusion.sample(class_label=cls, n_samples=samples_per_class, device=device)
        aug_x.append(samples)
        aug_y.append(np.full((samples_per_class,), cls, dtype=np.int64))
        
    aug_x = np.concatenate(aug_x, axis=0) # (N_aug, 1, 187)
    aug_y = np.concatenate(aug_y, axis=0)
    
    return TensorDataset(torch.tensor(aug_x, dtype=torch.float32), torch.tensor(aug_y, dtype=torch.long))

def main():
    args = parse_args()
    
    # Configure hyperparams based on --fast flag
    if args.fast:
        print("========================================================")
        print("  FAST SMOKE TEST MODE ACTIVATED (Target: ~1-2 min CPU)  ")
        print("========================================================")
        epochs_cls = args.epochs_cls or 2
        epochs_diff = args.epochs_diff or 2
        batch_size = args.batch_size or 32
        diff_timesteps = 30
        samples_per_minority = 25
    else:
        print("========================================================")
        print("  FULL TRAINING MODE ACTIVATED                          ")
        print("========================================================")
        epochs_cls = args.epochs_cls or 10
        epochs_diff = args.epochs_diff or 12
        batch_size = args.batch_size or 64
        diff_timesteps = 100
        samples_per_minority = 150

    # Device selection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Compute] Target execution device: {device}")
    
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Load data
    train_dataset, test_dataset, is_synthetic = load_ecg_data(args.data_dir, fast=args.fast)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Train Baseline Classifier
    baseline_cls = ECG1DCNN(num_classes=5)
    baseline_cls = train_classifier(
        baseline_cls, train_loader, epochs=epochs_cls, lr=2e-3, device=device, desc="Baseline Classifier"
    )
    baseline_path = os.path.join(args.weights_dir, "classifier_baseline.pt")
    torch.save(baseline_cls.state_dict(), baseline_path)
    print(f"[Checkpoint] Saved Baseline Classifier to {baseline_path}")
    
    # 3. Train Conditional 1D DDPM
    diffusion_model = ECG1DDiffusion(timesteps=diff_timesteps, num_classes=5)
    diffusion_model = train_diffusion_model(
        diffusion_model, train_loader, epochs=epochs_diff, lr=1e-3, device=device
    )
    diff_path = os.path.join(args.weights_dir, "diffusion_ecg.pt")
    torch.save(diffusion_model.state_dict(), diff_path)
    print(f"[Checkpoint] Saved Conditional DDPM to {diff_path}")
    
    # 4. Generate synthetic minority samples and create augmented dataset
    aug_dataset = generate_minority_augmentations(diffusion_model, device, samples_per_class=samples_per_minority)
    augmented_train_dataset = ConcatDataset([train_dataset, aug_dataset])
    aug_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    print(f"[Augment] Combined Dataset size: {len(train_dataset)} -> {len(augmented_train_dataset)} samples")
    
    # 5. Train Augmented Classifier
    augmented_cls = ECG1DCNN(num_classes=5)
    augmented_cls = train_classifier(
        augmented_cls, aug_loader, epochs=epochs_cls, lr=2e-3, device=device, desc="Augmented Classifier"
    )
    aug_path = os.path.join(args.weights_dir, "classifier_augmented.pt")
    torch.save(augmented_cls.state_dict(), aug_path)
    print(f"[Checkpoint] Saved Augmented Classifier to {aug_path}")
    
    print("\n========================================================")
    print("  ALL TRAINING STAGES COMPLETE! Checkpoints generated:  ")
    print(f"  - {baseline_path}")
    print(f"  - {diff_path}")
    print(f"  - {aug_path}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
