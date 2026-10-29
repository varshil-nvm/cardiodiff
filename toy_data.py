import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
import plotly.graph_objects as go

def get_toy_dataset(name='moons', n=2000, seed=42):
    """Generates a scaled [-1, 1] 2D dataset deterministically."""
    np.random.seed(seed)
    if name == 'moons':
        X, _ = make_moons(n_samples=n, noise=0.08, random_state=seed)
    elif name == 'circles':
        X, _ = make_circles(n_samples=n, noise=0.06, factor=0.5, random_state=seed)
    elif name == 'blobs':
        X, _ = make_blobs(n_samples=n, centers=3, cluster_std=1.0, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Normalize to [-1, 1] range to aid neural network convergence
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X = 2.0 * (X - x_min) / (x_max - x_min + 1e-8) - 1.0
    return X.astype(np.float32)

def plot_dataset_2d(data, title='Dataset'):
    """Creates an interactive Plotly Scatter plot of the 2D dataset."""
    fig = go.Figure(data=go.Scatter(
        x=data[:, 0], y=data[:, 1], 
        mode='markers',
        marker=dict(size=5, color='rgba(99, 110, 250, 0.6)', line=dict(width=0.5, color='white'))
    ))
    fig.update_layout(
        title=title, xaxis_title='X1', yaxis_title='X2',
        template='plotly_dark', width=600, height=400
    )
    return fig

def plot_samples_2d(samples, title='Generated Samples'):
    """Creates an interactive Plotly Scatter plot for synthetic samples."""
    return plot_dataset_2d(samples, title=title)
