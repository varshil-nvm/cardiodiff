import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act1 = nn.SiLU()
        
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act2 = nn.SiLU()
        
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.gn1(self.conv1(x)))
        # Inject condition
        cond_bias = self.cond_proj(cond).unsqueeze(-1)
        h = h + cond_bias
        h = self.act2(self.gn2(self.conv2(h)))
        return h + self.residual(x)

class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for ECG waveform diffusion.
    Accepts 1D signal of shape (B, 1, 187), conditioned on class label and timestep.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_classes: int = 5, base_dim: int = 64):
        super().__init__()
        cond_dim = base_dim * 2
        
        # Timestep and Class conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        
        # Initial projection
        self.init_conv = nn.Conv1d(in_channels, base_dim, kernel_size=7, padding=3)
        
        # Down blocks
        self.down1 = ConvBlock1D(base_dim, base_dim, cond_dim)
        self.down2 = ConvBlock1D(base_dim, base_dim * 2, cond_dim)
        self.down3 = ConvBlock1D(base_dim * 2, base_dim * 4, cond_dim)
        
        # Bottleneck
        self.mid1 = ConvBlock1D(base_dim * 4, base_dim * 4, cond_dim)
        self.mid2 = ConvBlock1D(base_dim * 4, base_dim * 4, cond_dim)
        
        # Up blocks
        self.up3 = ConvBlock1D(base_dim * 4 + base_dim * 4, base_dim * 2, cond_dim)
        self.up2 = ConvBlock1D(base_dim * 2 + base_dim * 2, base_dim, cond_dim)
        self.up1 = ConvBlock1D(base_dim + base_dim, base_dim, cond_dim)
        
        # Final output conv
        self.out_conv = nn.Conv1d(base_dim, out_channels, kernel_size=7, padding=3)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute combined conditioning vector
        t_emb = self.time_mlp(t)
        c_emb = self.class_embed(y)
        cond = t_emb + c_emb
        
        h0 = self.init_conv(x)
        h1 = self.down1(h0, cond)
        h2 = self.down2(h1, cond)
        h3 = self.down3(h2, cond)
        
        m = self.mid1(h3, cond)
        m = self.mid2(m, cond)
        
        u3 = self.up3(torch.cat([m, h3], dim=1), cond)
        u2 = self.up2(torch.cat([u3, h2], dim=1), cond)
        u1 = self.up1(torch.cat([u2, h1], dim=1), cond)
        
        out = self.out_conv(u1)
        return out

class ECG1DDiffusion(nn.Module):
    """
    Complete DDPM (Denoising Diffusion Probabilistic Model) for 1D ECG synthesis.
    """
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, num_classes: int = 5):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.model = ConditionalUNet1D(in_channels=1, out_channels=1, num_classes=num_classes, base_dim=32)
        
        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Variance for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion process: sample q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates MSE loss between true noise and network prediction.
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, y)
        
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor, t: int, y: torch.Tensor) -> torch.Tensor:
        """
        Single step of reverse diffusion p(x_{t-1} | x_t, y)
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        pred_noise = self.model(x_t, t_tensor, y)
        
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Mean
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)
        
        if t == 0:
            return model_mean
        else:
            var = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, class_label: int, n_samples: int = 1, length: int = 187, device: str = 'cpu', return_intermediates: bool = False):
        """
        Samples synthetic 1D ECG heartbeats conditioned on class_label.
        Optionally returns intermediates at intervals for the Diffusion Lab visualization.
        """
        self.eval()
        self.to(device)
        
        x = torch.randn((n_samples, 1, length), device=device)
        y = torch.full((n_samples,), class_label, device=device, dtype=torch.long)
        
        intermediates = []
        # Save initial pure noise
        if return_intermediates:
            intermediates.append((self.timesteps, x.cpu().numpy()))
            
        # Step intervals to capture for visualization
        capture_steps = set(range(0, self.timesteps, max(1, self.timesteps // 8)))
        capture_steps.add(0)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample_step(x, t, y)
            if return_intermediates and t in capture_steps:
                intermediates.append((t, x.cpu().numpy()))
                
        # Normalize output samples to [0, 1] range
        x_out = x.cpu().numpy()
        for i in range(len(x_out)):
            s_min = x_out[i, 0].min()
            s_max = x_out[i, 0].max()
            if s_max - s_min > 1e-6:
                x_out[i, 0] = (x_out[i, 0] - s_min) / (s_max - s_min)
                
        if return_intermediates:
            return x_out, intermediates
        return x_out
