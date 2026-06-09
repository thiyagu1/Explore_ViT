import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2          # 196
        self.proj       = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape

        # step 1 — patch embedding
        x = self.proj(x).flatten(2).transpose(1, 2)    # [B, 196, 768]

        # step 2 — prepend CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)            # [B, 197, 768]

        # ── save before adding position so we can compare ─────────────────────
        x_before_pos = x.clone()

        # step 3 — add positional embeddings
        x = x + self.pos_embed                          # [B, 197, 768]

        return x_before_pos, x                          # return both for comparison


# ─── Load image ───────────────────────────────────────────────────────────────

img_bgr     = cv2.imread("Image/Buddha2.PNG")
img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))

x = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
x = x.unsqueeze(0)                                     # [1, 3, 224, 224]


# ─── Run model ────────────────────────────────────────────────────────────────

torch.manual_seed(42)
model = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)

with torch.no_grad():
    before, after = model(x)

print(f"Shape before pos embed : {before.shape}")      # [1, 197, 768]
print(f"Shape after  pos embed : {after.shape}")       # [1, 197, 768]  ← same shape
print(f"Pos embed shape        : {model.pos_embed.shape}")  # [1, 197, 768]


# ─── Detach for numpy ─────────────────────────────────────────────────────────

before_np  = before[0].detach().numpy()                # [197, 768]
after_np   = after[0].detach().numpy()                 # [197, 768]
pos_np     = model.pos_embed[0].detach().numpy()       # [197, 768]
pos_tensor = model.pos_embed[0].detach()               # [197, 768] tensor


# ─── Print: compare one token before and after ───────────────────────────────

print(f"\nCLS token before pos embed  : {before_np[0, :6].round(4)}")
print(f"CLS positional embedding    : {pos_np[0,    :6].round(4)}")
print(f"CLS token after  pos embed  : {after_np[0,  :6].round(4)}")

print(f"\nPatch #1 before pos embed   : {before_np[1, :6].round(4)}")
print(f"Patch #1 positional embed   : {pos_np[1,    :6].round(4)}")
print(f"Patch #1 after  pos embed   : {after_np[1,  :6].round(4)}")


# ─── Figure 1: before vs after for a single token ────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("What positional embedding adds to a single token (patch #1)", fontsize=13)

axes[0].bar(range(64), before_np[1, :64], color="steelblue")
axes[0].set_title("Before: patch content only\n[768 dims from pixels]")
axes[0].set_xlabel("Dimension")
axes[0].set_ylabel("Value")

axes[1].bar(range(64), pos_np[1, :64], color="green")
axes[1].set_title("Positional embedding (random init)\n[768 learned values for position 1]")
axes[1].set_xlabel("Dimension")

axes[2].bar(range(64), after_np[1, :64], color="purple")
axes[2].set_title("After: content + position fused\n[same 768 dims, both signals inside]")
axes[2].set_xlabel("Dimension")

plt.tight_layout()
plt.savefig("pos_single_token.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── Figure 2: full positional embedding matrix ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Positional embedding matrix — all 197 positions", fontsize=13)

# all 197 positional vectors as a heatmap
im0 = axes[0].imshow(pos_np, aspect="auto", cmap="RdBu")
axes[0].set_title("Pos embed matrix [197 × 768]\nrandom init — no structure yet")
axes[0].set_xlabel("Embedding dimension (768)")
axes[0].set_ylabel("Position (0=CLS, 1–196=patches)")
plt.colorbar(im0, ax=axes[0])

# cosine similarity between every pair of positions
sim = np.zeros((197, 197))
for i in range(197):
    for j in range(197):
        sim[i, j] = cosine_similarity(
            pos_tensor[i].unsqueeze(0),
            pos_tensor[j].unsqueeze(0)
        ).item()

im1 = axes[1].imshow(sim, cmap="viridis", vmin=-1, vmax=1)
axes[1].set_title("Cosine similarity between positions\nrandom init — no spatial pattern yet")
axes[1].set_xlabel("Position j")
axes[1].set_ylabel("Position i")
plt.colorbar(im1, ax=axes[1])

# zoom into just the patch positions (1–196), reshape to 14x14 grid
# show similarity of center patch (position 98) to all other patches
center_patch = pos_tensor[98]
sims_to_center = np.zeros(196)
for i in range(196):
    sims_to_center[i] = cosine_similarity(
        center_patch.unsqueeze(0),
        pos_tensor[i + 1].unsqueeze(0)       # +1 to skip CLS
    ).item()

sim_grid = sims_to_center.reshape(14, 14)
im2 = axes[2].imshow(sim_grid, cmap="hot")
axes[2].set_title("Similarity of center patch (pos 98)\nto all other patch positions\n(random init — will show 2D structure after training)")
axes[2].set_xlabel("Patch column")
axes[2].set_ylabel("Patch row")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("pos_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── Figure 3: visualize what the addition does across all 197 tokens ─────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Effect of positional embedding across all 197 tokens", fontsize=13)

im0 = axes[0].imshow(before_np, aspect="auto", cmap="RdBu")
axes[0].set_title("Before pos embed [197 × 768]\npatch content only")
axes[0].set_xlabel("Dimension")
axes[0].set_ylabel("Position (0=CLS, 1–196=patches)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pos_np, aspect="auto", cmap="RdBu")
axes[1].set_title("Positional embeddings [197 × 768]\nwhat gets added")
axes[1].set_xlabel("Dimension")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(after_np, aspect="auto", cmap="RdBu")
axes[2].set_title("After pos embed [197 × 768]\ncontent + position fused")
axes[2].set_xlabel("Dimension")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("pos_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: pos_single_token.png")
print("Saved: pos_matrix.png")
print("Saved: pos_before_after.png")