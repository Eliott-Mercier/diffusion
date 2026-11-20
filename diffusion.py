import math

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T, schedule="linear", beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":  # Nichol & Dhariwal 2021
            s = torch.arange(T + 1) / T
            f = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        else:
            raise ValueError(schedule)
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        eps = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        xt, eps = self.add_noise(x0, t)
        return F.mse_loss(model(xt, t), eps)

    @torch.no_grad()
    def denoise_step(self, model, x, t):
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = model(x, tt)
        x = (x - self.betas[t] / (1 - self.alpha_bar[t]).sqrt() * eps) / self.alphas[t].sqrt()
        if t > 0:
            x = x + self.betas[t].sqrt() * torch.randn_like(x)  # sigma_t^2 = beta_t
        return x

    @torch.no_grad()
    def predict_x0(self, model, xt, t):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ((xt - (1 - ab).sqrt() * model(xt, t)) / ab.sqrt()).clamp(-1, 1)

    @torch.no_grad()
    def sample(self, model, n, image_size, device):
        model.eval()
        x = torch.randn(n, 1, image_size, image_size, device=device)
        for t in reversed(range(self.T)):
            x = self.denoise_step(model, x, t)
        model.train()
        return x.clamp(-1, 1)
