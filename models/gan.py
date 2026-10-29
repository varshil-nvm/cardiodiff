import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=64, output_dim=2):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) # Unscaled logits representing fake vs real
        )
        
    def forward(self, x):
        return self.net(x)

def train_gan(data, epochs=150, lr=1e-3, latent_dim=2, device='cpu'):
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    n_samples = data.shape[0]
    
    generator = Generator(latent_dim=latent_dim, hidden_dim=64, output_dim=2).to(device)
    discriminator = Discriminator(input_dim=2, hidden_dim=64).to(device)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    g_losses = []
    d_losses = []
    
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        # ---------------------
        # Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()
        
        # Real samples
        real_labels = torch.ones(n_samples, 1, device=device)
        d_real_preds = discriminator(tensor_data)
        d_real_loss = criterion(d_real_preds, real_labels)
        
        # Fake samples
        z = torch.randn(n_samples, latent_dim, device=device)
        fake_data = generator(z)
        fake_labels = torch.zeros(n_samples, 1, device=device)
        d_fake_preds = discriminator(fake_data.detach())
        d_fake_loss = criterion(d_fake_preds, fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        d_losses.append(d_loss.item())
        
        # ---------------------
        # Train Generator
        # ---------------------
        g_optimizer.zero_grad()
        
        # Generator wants Discriminator to classify fake samples as Real
        d_fake_preds_for_g = discriminator(fake_data)
        g_loss = criterion(d_fake_preds_for_g, real_labels)
        g_loss.backward()
        g_optimizer.step()
        g_losses.append(g_loss.item())
        
    return generator, discriminator, g_losses, d_losses
