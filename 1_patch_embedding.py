import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── The class ────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2          # 196
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)          # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)          # [B, embed_dim, N]
        x = x.transpose(1, 2)     # [B, N, embed_dim]
        return x


# ─── Load and preprocess image ────────────────────────────────────────────────

img_bgr     = cv2.imread("Image/Buddha2.PNG")
img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))

# numpy HWC → torch CHW, normalize to [0,1], add batch dim
x = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
x = x.unsqueeze(0)               # [1, 3, 224, 224]

print(f"Input shape  : {x.shape}")


# ─── Run through PatchEmbedding ───────────────────────────────────────────────

torch.manual_seed(42)
model = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)

with torch.no_grad():
    out = model(x)

print(f"Output shape : {out.shape}")        # [1, 196, 768]
print(f"N patches    : {out.shape[1]}")     # 196
print(f"Embed dim    : {out.shape[2]}")     # 768


# ─── Inspect a single patch embedding ────────────────────────────────────────

patch_idx = 85
emb = out[0, patch_idx].numpy()             # [768]

print(f"\nPatch #{patch_idx} embedding:")
print(f"  shape : {emb.shape}")
print(f"  min   : {emb.min():.4f}")
print(f"  max   : {emb.max():.4f}")
print(f"  mean  : {emb.mean():.4f}")
print(f"  first 8 values: {emb[:8].round(4)}")


# # ─── Visualize ────────────────────────────────────────────────────────────────
#
# fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#
# # 1. original image
# axes[0].imshow(img_resized)
# axes[0].set_title("Original image")
# axes[0].axis("on")
#
# # 2. highlight patch #85 on the image
# img_highlight = img_resized.copy()
# row = patch_idx // 14
# col = patch_idx % 14
# y1, y2 = row * 16, (row + 1) * 16
# x1, x2 = col * 16, (col + 1) * 16
# cv2.rectangle(img_highlight, (x1, y1), (x2, y2), (255, 0, 0), 2)
# axes[1].imshow(img_highlight)
# axes[1].set_title(f"Patch #{patch_idx} highlighted (row={row}, col={col})")
# axes[1].axis("on")
#
# # 3. embedding vector of that patch
# axes[2].bar(range(len(emb[:64])), emb[:64], color="steelblue")
# axes[2].set_title(f"Patch #{patch_idx} embedding (first 64 of 768 dims)")
# axes[2].set_xlabel("Dimension")
# axes[2].set_ylabel("Value")
#
# plt.tight_layout()
# plt.savefig("patch_embedding_test.png", dpi=150, bbox_inches="tight")
# plt.show()
#
# print("\nSaved: patch_embedding_test.png")
#
#
# # ─── Draw all 196 patches in a 14×14 grid ────────────────────────────────────
#
# p = 16                       # patch side length in pixels
# grid_size  = 224 // p        # 14
#
# fig2, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
# fig2.suptitle("All 196 patches (14 × 14 grid)", fontsize=14, y=1.01)
#
# for idx in range(grid_size * grid_size):
#     row = idx // grid_size
#     col = idx % grid_size
#     patch = img_resized[row*p:(row+1)*p,
#                         col*p:(col+1)*p]
#     ax = axes[row][col]
#     ax.imshow(patch)
#     ax.axis("off")
#     # label every patch with its index in small text
#     ax.text(0.5, 0.0, str(idx), transform=ax.transAxes,
#             fontsize=4, color="black", ha="center", va="bottom")
#
# plt.subplots_adjust(wspace=0.02, hspace=0.02)
# plt.savefig("all_patches.png", dpi=200, bbox_inches="tight")
# plt.show()
#
# print("Saved: all_patches.png")