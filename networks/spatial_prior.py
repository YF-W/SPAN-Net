import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCoordinateEncoding2D(nn.Module):
    def __init__(self, channels, temperature=10000):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        self.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)

        dim_t = self.channels // 2
        div_term = torch.exp(
            torch.arange(0, dim_t, 2).float() * (-math.log(self.temperature) / dim_t)
        ).to(x.device)

        pe_y = torch.zeros(H, dim_t, device=x.device)
        pe_y[:, 0::2] = torch.sin(y_embed.unsqueeze(1) * div_term)
        pe_y[:, 1::2] = torch.cos(y_embed.unsqueeze(1) * div_term)

        pe_x = torch.zeros(W, dim_t, device=x.device)
        pe_x[:, 0::2] = torch.sin(x_embed.unsqueeze(1) * div_term)
        pe_x[:, 1::2] = torch.cos(x_embed.unsqueeze(1) * div_term)

        pe_y = pe_y.unsqueeze(1).repeat(1, W, 1)
        pe_x = pe_x.unsqueeze(0).repeat(H, 1, 1)

        pos_embed = torch.cat([pe_y, pe_x], dim=2)
        pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos_embed


class CrossModalPriorSeeding(nn.Module):
    """Cross-modal prior seeding for the Text-Guided Spatial Prior Generator (SPG)."""

    def __init__(self, img_dim, text_dim=768, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.img_proj = nn.Conv2d(img_dim, hidden_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.pos_encoder = SpatialCoordinateEncoding2D(channels=hidden_dim)
        self.sa_img = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_img_q = nn.LayerNorm(hidden_dim)
        self.norm_img_kv = nn.LayerNorm(hidden_dim)
        self.norm_img2 = nn.LayerNorm(hidden_dim)
        self.max_kv_size = 14
        self.sa_text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm_text1 = nn.LayerNorm(hidden_dim)
        self.norm_text2 = nn.LayerNorm(hidden_dim)
        self.ca_cross = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, img_feat, text_emb):
        B, C, H, W = img_feat.shape
        img_q_spatial = self.img_proj(img_feat)
        img_q_spatial = img_q_spatial + self.pos_encoder(img_q_spatial)
        img_q_seq = img_q_spatial.flatten(2).transpose(1, 2)

        img_kv_spatial = F.adaptive_avg_pool2d(
            img_q_spatial, (min(H, self.max_kv_size), min(W, self.max_kv_size))
        )
        img_kv_seq = img_kv_spatial.flatten(2).transpose(1, 2)

        text_seq = self.text_proj(text_emb)
        img_sa_out, _ = self.sa_img(
            query=self.norm_img_q(img_q_seq),
            key=self.norm_img_kv(img_kv_seq),
            value=self.norm_img_kv(img_kv_seq),
        )
        img_seq = self.norm_img2(img_q_seq + img_sa_out)

        text_sa_out, _ = self.sa_text(
            query=self.norm_text1(text_seq),
            key=self.norm_text1(text_seq),
            value=self.norm_text1(text_seq),
        )
        text_seq = self.norm_text2(text_seq + text_sa_out)

        attn_output, _ = self.ca_cross(query=img_seq, key=text_seq, value=text_seq)
        return self.out_proj(attn_output).transpose(1, 2).view(B, 1, H, W)


class SemanticPriorWeighting(nn.Module):
    """Semantic Prior Weighting (SPW) for text-adaptive propagation."""

    def __init__(self, text_dim=768, init_k=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.register_buffer("init_bias", torch.tensor(float(init_k)))

    def forward(self, x, text_emb):
        text_global = torch.mean(text_emb, dim=1)
        k = torch.clamp(self.mlp(text_global) + self.init_bias, min=-5.0, max=5.0).view(
            -1, 1, 1, 1
        )
        u = torch.clamp((x + 1) / 2.0, 0.0, 1.0)
        b = torch.exp(-k)
        return u / (b * (1 - u) + u + 1e-6)


class SemanticSpatialPropagation(nn.Module):
    """Semantic Spatial Propagation (SSP) for spatial prior evolution."""

    def __init__(self, in_channels, iterations, init_alpha=0.1, text_dim=768):
        super().__init__()
        self.iterations = iterations
        self.alpha_logit = nn.Parameter(
            torch.log(torch.tensor(init_alpha) / (1 - torch.tensor(init_alpha) + 1e-6))
        )
        self.spw_controller = SemanticPriorWeighting(text_dim=text_dim, init_k=5.0)
        self.beta = nn.Parameter(torch.tensor(2.0))

    def get_four_direction_affinity(self, img_feat):
        feat = F.normalize(img_feat, p=2, dim=1)
        fp = F.pad(feat, (1, 1, 1, 1), mode="replicate")
        return [
            torch.sum(feat * fp[:, :, 0:-2, 1:-1], dim=1, keepdim=True),
            torch.sum(feat * fp[:, :, 2:, 1:-1], dim=1, keepdim=True),
            torch.sum(feat * fp[:, :, 1:-1, 0:-2], dim=1, keepdim=True),
            torch.sum(feat * fp[:, :, 1:-1, 2:], dim=1, keepdim=True),
        ]

    def get_gradient(self, current):
        cp = F.pad(current, (1, 1, 1, 1), mode="replicate")
        return [
            cp[:, :, 0:-2, 1:-1] - current,
            cp[:, :, 2:, 1:-1] - current,
            cp[:, :, 1:-1, 0:-2] - current,
            cp[:, :, 1:-1, 2:] - current,
        ]

    def forward(self, seed, img_feat, text_emb):
        sim_maps = self.get_four_direction_affinity(img_feat)
        current = seed
        l_alpha = torch.sigmoid(self.alpha_logit)
        r_strength = F.softplus(self.beta)

        for _ in range(self.iterations):
            grads = self.get_gradient(current)
            div = 0.0
            for sm, g in zip(sim_maps, grads):
                div += self.spw_controller(sm, text_emb) * g

            u = torch.clamp(current, -1.0, 1.0)
            current = current + l_alpha * (div + r_strength * (u - u**3))
            current = torch.clamp(current, -1.1, 1.1)
        return current
