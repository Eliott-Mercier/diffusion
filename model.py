import math

import torch
import torch.nn as nn


def time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


def conv(c_in, c_out, stride=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, base_ch=32, emb_dim=128, skip=True, use_t=True):
        super().__init__()
        c = base_ch
        self.emb_dim = emb_dim
        self.skip, self.use_t = skip, use_t
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        self.inc = conv(1, c)
        self.down1 = conv(c, 2 * c, stride=2)
        self.down2 = conv(2 * c, 4 * c, stride=2)
        self.mid = conv(4 * c, 4 * c)
        self.up1 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec1 = conv(4 * c, 2 * c)
        self.up2 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.dec2 = conv(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

        self.t_inc, self.t_down1 = nn.Linear(emb_dim, c), nn.Linear(emb_dim, 2 * c)
        self.t_down2, self.t_dec1 = nn.Linear(emb_dim, 4 * c), nn.Linear(emb_dim, 2 * c)

    def forward(self, x, t):
        if not self.use_t:
            t = torch.zeros_like(t)
        e = self.time_mlp(time_embedding(t, self.emb_dim))

        h0 = self.inc(x) + self.t_inc(e)[:, :, None, None]
        h1 = self.down1(h0) + self.t_down1(e)[:, :, None, None]
        h2 = self.down2(h1) + self.t_down2(e)[:, :, None, None]
        # sans skip : des zéros à la place des cartes de l'encodeur, même nombre de paramètres
        s0, s1 = (h0, h1) if self.skip else (torch.zeros_like(h0), torch.zeros_like(h1))
        h = self.mid(h2)
        h = self.dec1(torch.cat([self.up1(h), s1], dim=1)) + self.t_dec1(e)[:, :, None, None]
        h = self.dec2(torch.cat([self.up2(h), s0], dim=1))
        return self.out(h)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.block1 = nn.Sequential(nn.GroupNorm(8, c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, padding=1))
        self.t_proj = nn.Linear(emb_dim, c_out)
        self.block2 = nn.Sequential(nn.GroupNorm(8, c_out), nn.SiLU(), nn.Conv2d(c_out, c_out, 3, padding=1))
        self.shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.block1(x) + self.t_proj(e)[:, :, None, None]
        return self.block2(h) + self.shortcut(x)


class ResUNet(nn.Module):
    def __init__(self, base_ch=32, emb_dim=128):
        super().__init__()
        c = base_ch
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inc = nn.Conv2d(1, c, 3, padding=1)
        self.res0 = ResBlock(c, c, emb_dim)
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)
        self.res1 = ResBlock(2 * c, 2 * c, emb_dim)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)
        self.mid = ResBlock(4 * c, 4 * c, emb_dim)
        self.up1 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1)
        self.dec1 = ResBlock(4 * c, 2 * c, emb_dim)
        self.up2 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1)
        self.dec2 = ResBlock(2 * c, c, emb_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        e = self.time_mlp(time_embedding(t, self.emb_dim))
        h0 = self.res0(self.inc(x), e)
        h1 = self.res1(self.down1(h0), e)
        h = self.mid(self.down2(h1), e)
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), e)
        h = self.dec2(torch.cat([self.up2(h), h0], dim=1), e)
        return self.out(h)


def build_model(cfg):
    if cfg.model == "unet":
        return UNet(cfg.base_ch, cfg.emb_dim)
    if cfg.model == "unet_noskip":
        return UNet(cfg.base_ch, cfg.emb_dim, skip=False)
    if cfg.model == "unet_not":
        return UNet(cfg.base_ch, cfg.emb_dim, use_t=False)
    if cfg.model == "resunet":
        return ResUNet(cfg.base_ch, cfg.emb_dim)
    raise ValueError(f"Modèle inconnu : {cfg.model}")
