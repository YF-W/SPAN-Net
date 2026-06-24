import torch
import torch.nn as nn
from einops import rearrange

from networks.spatial_prior import CrossModalPriorSeeding, SemanticSpatialPropagation


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, num_groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(
        self, in_channels, out_channels, iterations=0, text_dim=768, use_injector=True, pool=True
    ):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, num_groups=8)

        self.use_injector = use_injector
        if self.use_injector:
            self.injector = SpatialPriorAwareEncoderStage(out_channels, iterations, text_dim)

        self.pool = pool
        if self.pool:
            self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, text_emb=None, original_image=None):
        feat = self.double_conv(x)
        prior, seed = None, None

        if self.use_injector:
            feat, prior, seed = self.injector(feat, text_emb, original_image)

        if self.pool:
            out = self.maxpool(feat)
        else:
            out = feat

        if self.use_injector:
            return out, feat, prior, seed
        return out, feat


class PriorModulatedEdgeAttentionBlock(nn.Module):
    """Decoder upsampling block with Prior-Modulated Edge Attention (PMEA)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(
            in_channels + (in_channels // 2), out_channels, in_channels // 2
        )

        self.pmea_gate = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def apply_pmea(self, img_feat, seed, prior):
        """Apply PMEA using signed prior maps."""
        raw_diff = torch.abs(prior - seed)
        resistance_map = 1.0 - (raw_diff / 2.0)
        edge_potential = 1.0 - prior ** 2
        synergy_map = resistance_map * edge_potential

        edge_attention_weight = self.pmea_gate(synergy_map)
        feat_enhanced = img_feat * (1 + edge_attention_weight)
        feat_enhanced = self.fusion_conv(feat_enhanced)

        return feat_enhanced, synergy_map

    def forward(self, x1, x2, seed=None, prior_map=None):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if seed is not None:
            x, diff = self.apply_pmea(x, seed, prior_map)

        return x, diff


class PriorGuidedSemanticFusion(nn.Module):
    """Prior-Guided Semantic Fusion (PGSF)."""

    def __init__(self, dim, num_heads=4, pool_size=7, num_groups=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pool_size = pool_size

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim + 1, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim + 1, dim, kernel_size=1)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(0.1)
        self.proj_trans = nn.Conv2d(dim, dim, kernel_size=1)

        self.cnn_local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(num_groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, dim),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img_feat, prior_map):
        B, C, H, W = img_feat.shape
        q = self.q_proj(img_feat)
        q = rearrange(q, "b c h w -> b (h w) c").contiguous()

        kv_input = torch.cat([img_feat, prior_map], dim=1)
        pool_h = min(H, self.pool_size)
        pool_w = min(W, self.pool_size)
        kv_pooled = torch.nn.functional.adaptive_avg_pool2d(kv_input, (pool_h, pool_w))

        k = self.k_proj(kv_pooled)
        v = self.v_proj(kv_pooled)
        k = rearrange(k, "b c h w -> b (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b (h w) c").contiguous()

        k = self.norm_kv(k)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads).contiguous()
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads).contiguous()
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads).contiguous()

        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        feat_trans = attn @ v
        feat_trans = rearrange(feat_trans, "b h n d -> b n (h d)").contiguous()
        feat_trans = rearrange(feat_trans, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        feat_trans = self.proj_trans(feat_trans)

        uncertainty_map = 4.0 * prior_map * (1.0 - prior_map)
        feat_gated = img_feat * uncertainty_map
        feat_cnn = self.cnn_local(feat_gated)

        feat_concat = torch.cat([feat_trans, feat_cnn], dim=1)
        feat_synergy = self.fusion_conv(feat_concat)

        out = img_feat + self.gamma * feat_synergy
        return out.contiguous()


class LayerTextAdapter(nn.Module):
    def __init__(self, text_dim=768, num_groups=8):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv1d(in_channels=text_dim, out_channels=text_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, text_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=text_dim, out_channels=text_dim, kernel_size=1),
            nn.GroupNorm(num_groups, text_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, text_emb):
        x = text_emb.transpose(1, 2)
        x = self.adapter(x)
        x = x.transpose(1, 2)
        return text_emb + x


class SpatialPriorAwareEncoderStage(nn.Module):
    """Encoder-stage SPG and PGSF block."""

    def __init__(self, channels, iterations, text_dim=768):
        super().__init__()
        self.text_adapter = LayerTextAdapter(text_dim=text_dim)
        self.spg_seeding = CrossModalPriorSeeding(img_dim=channels, text_dim=text_dim)
        self.ssp_propagation = SemanticSpatialPropagation(in_channels=channels, iterations=iterations)
        self.pgsf_fusion = PriorGuidedSemanticFusion(dim=channels)

    def forward(self, img_feat, text_emb, original_image):
        text_emb = self.text_adapter(text_emb)
        seed = self.spg_seeding(img_feat, text_emb)
        prior = self.ssp_propagation(seed, img_feat, text_emb)

        normalized_prior = (prior + 1.0) / 2.0
        normalized_prior = torch.clamp(normalized_prior, 0.0, 1.0)
        out_feat = self.pgsf_fusion(img_feat, normalized_prior)

        return out_feat, prior, seed


class SPANNet(nn.Module):
    def __init__(self, n_channels, n_classes, text_dim=768):
        super(SPANNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(in_channels=n_channels, out_channels=64, iterations=16, text_dim=text_dim, use_injector=True)
        self.down2 = Down(in_channels=64, out_channels=128, iterations=8, text_dim=text_dim, use_injector=True)
        self.down3 = Down(in_channels=128, out_channels=256, iterations=4, text_dim=text_dim, use_injector=True)
        self.down4 = Down(in_channels=256, out_channels=512, iterations=2, text_dim=text_dim, use_injector=True)

        self.down5 = Down(in_channels=512, out_channels=1024, use_injector=False, pool=False)

        self.up1 = PriorModulatedEdgeAttentionBlock(1024, 512)
        self.up2 = PriorModulatedEdgeAttentionBlock(512, 256)
        self.up3 = PriorModulatedEdgeAttentionBlock(256, 128)
        self.up4 = PriorModulatedEdgeAttentionBlock(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x, text_emb):
        priors, features, seeds = [], [], []

        x_pool1, x1_sharp, p1, s1 = self.down1(x, text_emb)
        priors.append(p1)
        features.append(x1_sharp)
        seeds.append(s1)

        x_pool2, x2_sharp, p2, s2 = self.down2(x_pool1, text_emb)
        priors.append(p2)
        features.append(x2_sharp)
        seeds.append(s2)

        x_pool3, x3_sharp, p3, s3 = self.down3(x_pool2, text_emb)
        priors.append(p3)
        features.append(x3_sharp)
        seeds.append(s3)

        x_pool4, x4_sharp, p4, s4 = self.down4(x_pool3, text_emb)
        priors.append(p4)
        features.append(x4_sharp)
        seeds.append(s4)

        x5, _ = self.down5(x_pool4)

        up_priors = []
        x, up_p4 = self.up1(x5, x4_sharp, s4, p4)
        x, up_p3 = self.up2(x, x3_sharp, s3, p3)
        x, up_p2 = self.up3(x, x2_sharp, s2, p2)
        x, up_p1 = self.up4(x, x1_sharp, s1, p1)

        up_priors.extend([up_p1, up_p2, up_p3, up_p4])

        logits = self.outc(x)
        return logits, priors, up_priors, seeds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).to(device)
    text_emb = torch.randn(batch_size, 196, 768).to(device)

    model = SPANNet(n_channels=3, n_classes=1, text_dim=768).to(device)
    logits, priors, up_priors, seeds = model(img, text_emb)

    def print_shape(name, x):
        if isinstance(x, (list, tuple)):
            print(f"{name}:")
            for i, t in enumerate(x):
                if hasattr(t, "shape"):
                    print(f"  [{i}] {tuple(t.shape)}")
                else:
                    print(f"  [{i}] type={type(t)}")
        else:
            if hasattr(x, "shape"):
                print(f"{name}: {tuple(x.shape)}")
            else:
                print(f"{name}: type={type(x)}")

    print_shape("logits", logits)
    print_shape("priors", priors)
    print_shape("up_priors", up_priors)
    print_shape("seeds", seeds)

    print("Forward pass test succeeded.")
    print(f"Prediction Shape: {logits.shape}")
