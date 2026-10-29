import numpy as np
import torch
from scipy import linalg
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Fréchet Distance.
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    diff_norm_sq = diff.dot(diff)

    # Product of covariances
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    except Exception:
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if isinstance(covmean, tuple):
        covmean = covmean[0]

    # Numerical imaginary component handling
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff_norm_sq + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.real(fid))

def compute_1d_fid(model_fe, real_signals: torch.Tensor, gen_signals: torch.Tensor, device: str = 'cpu') -> float:
    """
    Computes 1D-FID between real ECG signals and generated ECG signals
    using features extracted from the 1D-CNN classifier.
    """
    model_fe.eval()
    model_fe.to(device)
    
    with torch.no_grad():
        r_feats = []
        for i in range(0, len(real_signals), 64):
            batch = real_signals[i:i+64].to(device)
            feats = model_fe.extract_features(batch)
            r_feats.append(feats.cpu().numpy())
        r_feats = np.concatenate(r_feats, axis=0)
        
        g_feats = []
        for i in range(0, len(gen_signals), 64):
            batch = gen_signals[i:i+64].to(device)
            feats = model_fe.extract_features(batch)
            g_feats.append(feats.cpu().numpy())
        g_feats = np.concatenate(g_feats, axis=0)

    mu_r = np.mean(r_feats, axis=0)
    sigma_r = np.cov(r_feats, rowvar=False)

    mu_g = np.mean(g_feats, axis=0)
    sigma_g = np.cov(g_feats, rowvar=False)

    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

def compute_diversity_ratio(real_signals: np.ndarray, gen_signals: np.ndarray, max_pairs: int = 500) -> float:
    """
    Computes the Diversity Ratio: ratio of average pairwise distance of generated
    samples to average pairwise distance of real samples.
    """
    real_flat = real_signals.reshape(len(real_signals), -1)
    gen_flat = gen_signals.reshape(len(gen_signals), -1)
    
    def mean_pairwise_distance(arr, max_n):
        n = min(len(arr), max_n)
        sub = arr[np.random.choice(len(arr), n, replace=False)] if len(arr) > n else arr
        diffs = sub[:, None, :] - sub[None, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
        # Mask out diagonal (distance to self)
        mask = ~np.eye(len(sub), dtype=bool)
        return float(np.mean(dists[mask])) if len(sub) > 1 else 1.0

    d_real = mean_pairwise_distance(real_flat, max_pairs)
    d_gen = mean_pairwise_distance(gen_flat, max_pairs)
    
    if d_real < 1e-6:
        return 1.0
    return float(d_gen / d_real)

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes Macro-F1, Minority Recall, Overall Accuracy, and per-class metrics.
    Minority classes in MIT-BIH are classes 1, 2, 3, 4.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # Calculate per-class recall
    recalls = []
    for cls in range(5):
        denom = np.sum(y_true == cls)
        if denom > 0:
            rec = np.sum((y_true == cls) & (y_pred == cls)) / denom
        else:
            rec = 0.0
        recalls.append(rec)
        
    # Minority recall is average recall of classes 1, 2, 3, 4
    minority_recall = float(np.mean(recalls[1:]))
    
    report_dict = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    
    return {
        "macro_f1": float(macro_f1),
        "minority_recall": float(minority_recall),
        "accuracy": float(acc),
        "class_recalls": [float(r) for r in recalls],
        "confusion_matrix": cm.tolist(),
        "report": report_dict
    }
