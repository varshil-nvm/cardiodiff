import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPNoiseEstimator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, time_dim=16):
        super(MLPNoiseEstimator, self).__init__()
        # Simple sinusoidal-like embeddings for time steps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x_t, t):
        # x_t: [B, input_dim], t: [B, 1]
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x_t, t_emb], dim=-1)
        return self.net(x_input)

class DDPM(nn.Module):
    def __init__(self, eps_model, beta_start=1e-4, beta_end=0.02, n_steps=100, device='cpu'):
        super(DDPM, self).__init__()
        self.eps_model = eps_model.to(device)
        self.n_steps = n_steps
        self.device = device
        
        # Schedulers
        self.beta = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        
    def sample_forward(self, x_0, t, noise):
        # q(x_t | x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t]).view(-1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
    def sample_backward(self, n_samples=500):
        self.eps_model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, 2, device=self.device)
            for t_idx in reversed(range(self.n_steps)):
                t = torch.full((n_samples, 1), t_idx, dtype=torch.float32, device=self.device)
                predicted_noise = self.eps_model(x, t / self.n_steps)
                
                # Reconstruct x_{t-1} via reverse diffusion formula
                beta_t = self.beta[t_idx]
                alpha_t = self.alpha[t_idx]
                alpha_bar_t = self.alpha_bar[t_idx]
                
                mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise)
                
                if t_idx > 0:
                    noise = torch.randn_like(x)
                    sigma_t = torch.sqrt(beta_t)
                    x = mean + sigma_t * noise
                else:
                    x = mean
        return x.cpu().numpy()

def train_diffusion(data, epochs=150, lr=1e-3, n_steps=100, device='cpu'):
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    n_samples = data.shape[0]
    
    eps_model = MLPNoiseEstimator(input_dim=2, hidden_dim=64, time_dim=16)
    ddpm = DDPM(eps_model, n_steps=n_steps, device=device)
    optimizer = optim.Adam(ddpm.parameters(), lr=lr)
    
    ddpm.eps_model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample t deterministically distributed across samples
        t = torch.randint(0, n_steps, (n_samples,), device=device)
        noise = torch.randn_like(tensor_data)
        
        # Forward diffusion process
        x_t = ddpm.sample_forward(tensor_data, t, noise)
        
        # Estimate injected noise
        predicted_noise = ddpm.eps_model(x_t, t.float().view(-1, 1) / n_steps)
        
        # Calculate reconstruction/denoising loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return ddpm, losses
