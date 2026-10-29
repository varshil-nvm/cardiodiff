import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # outputting raw continuous 2D coordinates
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Mean Squared Error as reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(data, epochs=150, lr=1e-3, latent_dim=2, beta=0.1, device='cpu'):
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    model = VAE(input_dim=2, hidden_dim=64, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(tensor_data)
        loss, r_loss, k_loss = loss_function(recon, tensor_data, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return model, losses
