import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj      = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, head_dim=64):
        super().__init__()
        self.head_dim = head_dim
        self.scale    = head_dim ** -0.5           # 1 / sqrt(64)
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        Q = self.W_Q(x)                            # [B, 197, 64]
        K = self.W_K(x)                            # [B, 197, 64]
        V = self.W_V(x)                            # [B, 197, 64]

        scores  = (Q @ K.transpose(-2, -1)) * self.scale  # [B, 197, 197]
        weights = F.softmax(scores, dim=-1)                # [B, 197, 197]
        output  = weights @ V                              # [B, 197, 64]

        return output, weights


# ─── Load image ───────────────────────────────────────────────────────────────

img_bgr     = cv2.imread("Image/Buddha2.PNG")
img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))

x = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
x = x.unsqueeze(0)


# ─── Forward pass ─────────────────────────────────────────────────────────────

torch.manual_seed(42)
patch_embed = PatchEmbedding()
ln          = nn.LayerNorm(768)
attn        = SingleHeadAttention(embed_dim=768, head_dim=64)

with torch.no_grad():
    tokens   = patch_embed(x)                     # [1, 197, 768]
    tokens_ln = ln(tokens)                         # [1, 197, 768]
    out, weights = attn(tokens_ln)                 # [1, 197, 64], [1, 197, 197]

print(f"Input shape          : {tokens_ln.shape}")
print(f"Attention out shape  : {out.shape}")       # [1, 197, 64]
print(f"Attention weights    : {weights.shape}")   # [1, 197, 197]
print(f"Weights sum (row 0)  : {weights[0, 0].sum():.4f}")   # must be 1.0


# ─── Visualize attention weights ──────────────────────────────────────────────

w = weights[0].detach().numpy()                    # [197, 197]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Single Head Self Attention on Buddha image", fontsize=13)

# 1. full attention matrix
im0 = axes[0].imshow(w, cmap="viridis")
axes[0].set_title("Attention matrix [197×197]\nrow i = how token i attends to all others")
axes[0].set_xlabel("Token j (key)")
axes[0].set_ylabel("Token i (query)")
plt.colorbar(im0, ax=axes[0])

# 2. CLS token attention — how CLS attends to all 196 patches
cls_attn = w[0, 1:]                                # [196] skip CLS attending to itself
cls_grid = cls_attn.reshape(14, 14)                # reshape to image grid
im1 = axes[1].imshow(cls_grid, cmap="hot")
axes[1].set_title("CLS attention over 196 patches\nreshaped to 14×14 grid\n(random init — no meaning yet)")
axes[1].set_xlabel("Patch column")
axes[1].set_ylabel("Patch row")
plt.colorbar(im1, ax=axes[1])

# 3. overlay attention on original image
cls_attn_resized = cv2.resize(cls_grid,
                               (224, 224),
                               interpolation=cv2.INTER_LINEAR)
axes[2].imshow(img_resized)
axes[2].imshow(cls_attn_resized,
               cmap="hot", alpha=0.5)
axes[2].set_title("CLS attention overlaid on image\n(random init — uniform after softmax)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("attention.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: attention.png")