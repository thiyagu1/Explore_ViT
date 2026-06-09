import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE HEAD SELF ATTENTION — COMPLETE FLOW
# Using the exact values from the ViT blog walkthrough
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(42)

print("=" * 60)
print("SELF ATTENTION — COMPLETE FLOW")
print("=" * 60)


# ─── Input ────────────────────────────────────────────────────────────────────
# Two tokens after Layer Normalization (first 4 dims of your actual values)
# CLS token and Patch #1 from the Buddha image run

Z = torch.tensor([
    [-0.1198, -1.8722, -0.5205,  0.5967],   # CLS   (position 0)
    [ 0.4821, -0.9234,  0.7823, -0.3412],   # patch1 (position 1)
], dtype=torch.float32)

print(f"\n── INPUT Z [2 tokens × 4 dims] ──────────────────────────")
print(f"CLS    : {Z[0].tolist()}")
print(f"patch1 : {Z[1].tolist()}")
print(f"shape  : {Z.shape}")


# ─── Weight Matrices ─────────────────────────────────────────────────────────
# In real ViT these are nn.Linear(768, 64, bias=False)
# Here we use [4×4] for readability
# These are RANDOM at init — meaning comes from training

d_model = 4
d_head  = 4
scale   = d_head ** -0.5    # 1 / sqrt(4) = 0.5

W_Q = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.2, 0.1, 0.4, 0.3],
    [0.6, 0.5, 0.8, 0.7],
], dtype=torch.float32)

W_K = torch.tensor([
    [0.3, 0.1, 0.4, 0.2],
    [0.7, 0.5, 0.8, 0.6],
    [0.1, 0.3, 0.2, 0.4],
    [0.5, 0.7, 0.6, 0.8],
], dtype=torch.float32)

W_V = torch.tensor([
    [0.4, 0.3, 0.2, 0.1],
    [0.8, 0.7, 0.6, 0.5],
    [0.3, 0.4, 0.1, 0.2],
    [0.7, 0.8, 0.5, 0.6],
], dtype=torch.float32)

print(f"\n── WEIGHT MATRICES [4×4] ────────────────────────────────")
print(f"W_Q shape : {W_Q.shape}  (random at init — shaped by training)")
print(f"W_K shape : {W_K.shape}  (random at init — shaped by training)")
print(f"W_V shape : {W_V.shape}  (random at init — shaped by training)")


# ─── Step 1: Compute Q, K, V ─────────────────────────────────────────────────
# Q = Z @ W_Q   → what am I looking for?   (role learned through training)
# K = Z @ W_K   → what do I contain?       (role learned through training)
# V = Z @ W_V   → what will I share?       (role learned through training)

Q = Z @ W_Q    # [2, 4] @ [4, 4] = [2, 4]
K = Z @ W_K    # [2, 4] @ [4, 4] = [2, 4]
V = Z @ W_V    # [2, 4] @ [4, 4] = [2, 4]

print(f"\n── STEP 1: PROJECT TO Q, K, V ───────────────────────────")
print(f"Q = Z @ W_Q → shape {Q.shape}")
print(f"  CLS    Q : {Q[0].tolist()}")
print(f"  patch1 Q : {Q[1].tolist()}")
print(f"K = Z @ W_K → shape {K.shape}")
print(f"  CLS    K : {K[0].tolist()}")
print(f"  patch1 K : {K[1].tolist()}")
print(f"V = Z @ W_V → shape {V.shape}")
print(f"  CLS    V : {V[0].tolist()}")
print(f"  patch1 V : {V[1].tolist()}")


# ─── Step 2: Raw scores QK^T ─────────────────────────────────────────────────
# score[i, j] = how much token i should attend to token j
# computed as dot product of Q[i] and K[j]

scores_raw = Q @ K.T    # [2, 4] @ [4, 2] = [2, 2]

print(f"\n── STEP 2: RAW SCORES  Q @ K^T ─────────────────────────")
print(f"shape : {scores_raw.shape}  (every token vs every token)")
print(f"\n  score[CLS,   CLS]    = Q_CLS   · K_CLS   = {scores_raw[0,0]:.4f}")
print(f"  score[CLS,   patch1] = Q_CLS   · K_patch1 = {scores_raw[0,1]:.4f}")
print(f"  score[patch1,CLS]    = Q_patch1 · K_CLS   = {scores_raw[1,0]:.4f}")
print(f"  score[patch1,patch1] = Q_patch1 · K_patch1 = {scores_raw[1,1]:.4f}")
print(f"\n  matrix:\n{scores_raw}")


# ─── Step 3: Scale ───────────────────────────────────────────────────────────
# Divide by sqrt(d_head) to prevent large dot products
# from pushing softmax into regions with near-zero gradients

scores_scaled = scores_raw * scale

print(f"\n── STEP 3: SCALE BY 1/sqrt({d_head}) = {scale:.4f} ──────────────────")
print(f"\n  before scaling:\n{scores_raw}")
print(f"\n  after  scaling (÷ {1/scale:.2f}):\n{scores_scaled}")
print(f"\n  why: large scores → softmax output near 0 or 1 → gradient vanishes")
print(f"       scaling keeps scores in a stable range")


# ─── Step 4: Softmax ─────────────────────────────────────────────────────────
# Convert scores to probabilities — each row sums to 1.0
# score[i, j] → how much token i attends to token j

attn_weights = F.softmax(scores_scaled, dim=-1)    # softmax over last dim (j)

print(f"\n── STEP 4: SOFTMAX (row-wise) ───────────────────────────")
print(f"\n  formula: softmax(a,b) = e^a/(e^a+e^b),  e^b/(e^a+e^b)")
print(f"\n  Row 0 — CLS attends to:")
print(f"    CLS    : e^{scores_scaled[0,0]:.4f} / sum = {attn_weights[0,0]:.4f}  ({attn_weights[0,0]*100:.1f}%)")
print(f"    patch1 : e^{scores_scaled[0,1]:.4f} / sum = {attn_weights[0,1]:.4f}  ({attn_weights[0,1]*100:.1f}%)")
print(f"    sum    : {attn_weights[0].sum():.4f} ✅")
print(f"\n  Row 1 — patch1 attends to:")
print(f"    CLS    : e^{scores_scaled[1,0]:.4f} / sum = {attn_weights[1,0]:.4f}  ({attn_weights[1,0]*100:.1f}%)")
print(f"    patch1 : e^{scores_scaled[1,1]:.4f} / sum = {attn_weights[1,1]:.4f}  ({attn_weights[1,1]*100:.1f}%)")
print(f"    sum    : {attn_weights[1].sum():.4f} ✅")
print(f"\n  full attention matrix:\n{attn_weights}")


# ─── Step 5: Weighted sum of Values ──────────────────────────────────────────
# output[i] = sum over j of (attn_weight[i,j] * V[j])
# each token becomes a weighted mix of all token values

output = attn_weights @ V    # [2, 2] @ [2, 4] = [2, 4]

print(f"\n── STEP 5: OUTPUT = ATTENTION WEIGHTS @ V ───────────────")
print(f"\n  new CLS = {attn_weights[0,0]:.4f} × V_CLS  +  {attn_weights[0,1]:.4f} × V_patch1")
print(f"          = {attn_weights[0,0]:.4f} × {V[0].tolist()}")
print(f"          + {attn_weights[0,1]:.4f} × {V[1].tolist()}")

manual_cls = attn_weights[0,0] * V[0] + attn_weights[0,1] * V[1]
print(f"          = {manual_cls.tolist()}")
print(f"  pytorch = {output[0].tolist()}")
print(f"  match   : {torch.allclose(manual_cls, output[0], atol=1e-4)} ✅")

print(f"\n  new patch1 = {attn_weights[1,0]:.4f} × V_CLS  +  {attn_weights[1,1]:.4f} × V_patch1")
print(f"             = {attn_weights[1,0]:.4f} × {V[0].tolist()}")
print(f"             + {attn_weights[1,1]:.4f} × {V[1].tolist()}")
manual_p1 = attn_weights[1,0] * V[0] + attn_weights[1,1] * V[1]
print(f"             = {manual_p1.tolist()}")
print(f"  pytorch    = {output[1].tolist()}")
print(f"  match      : {torch.allclose(manual_p1, output[1], atol=1e-4)} ✅")


# ─── Summary ──────────────────────────────────────────────────────────────────

print(f"\n── SUMMARY ──────────────────────────────────────────────")
print(f"\n  INPUT  Z (after LayerNorm):")
print(f"    CLS    : {Z[0].tolist()}")
print(f"    patch1 : {Z[1].tolist()}")
print(f"\n  ATTENTION WEIGHTS (who attends to whom):")
print(f"    CLS    → CLS    : {attn_weights[0,0]:.4f}  ({attn_weights[0,0]*100:.1f}%)")
print(f"    CLS    → patch1 : {attn_weights[0,1]:.4f}  ({attn_weights[0,1]*100:.1f}%)")
print(f"    patch1 → CLS    : {attn_weights[1,0]:.4f}  ({attn_weights[1,0]*100:.1f}%)")
print(f"    patch1 → patch1 : {attn_weights[1,1]:.4f}  ({attn_weights[1,1]*100:.1f}%)")
print(f"\n  OUTPUT (tokens enriched with context):")
print(f"    new CLS    : {output[0].tolist()}")
print(f"    new patch1 : {output[1].tolist()}")
print(f"\n  WHAT CHANGED:")
print(f"    CLS    before : {Z[0].tolist()}")
print(f"    CLS    after  : {output[0].tolist()}")
print(f"    patch1 before : {Z[1].tolist()}")
print(f"    patch1 after  : {output[1].tolist()}")
print(f"\n  KEY POINT:")
print(f"    old CLS    knew only about itself")
print(f"    new CLS    = {attn_weights[0,0]*100:.1f}% itself + {attn_weights[0,1]*100:.1f}% patch1")
print(f"    old patch1 knew only about itself")
print(f"    new patch1 = {attn_weights[1,0]*100:.1f}% CLS + {attn_weights[1,1]*100:.1f}% itself")


# ─── Verify with PyTorch nn.MultiheadAttention ────────────────────────────────
# confirms our manual math matches PyTorch's implementation

print(f"\n── VERIFY WITH PyTorch nn.MultiheadAttention ────────────")
import torch.nn as nn

# single head, no bias, embed_dim=4
mha = nn.MultiheadAttention(embed_dim=4, num_heads=1, bias=False, batch_first=True)

# manually set the weights to match our W_Q, W_K, W_V
# PyTorch packs Q,K,V weights into one matrix [3*d, d]
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.cat([W_Q.T, W_K.T, W_V.T], dim=0))
    mha.out_proj.weight.copy_(torch.eye(4))   # identity — no output projection

Z_batch = Z.unsqueeze(0)    # [1, 2, 4] — batch of 1
with torch.no_grad():
    pt_output, pt_weights = mha(Z_batch, Z_batch, Z_batch)

print(f"\n  our output    CLS    : {output[0].tolist()}")
print(f"  pytorch output CLS   : {pt_output[0, 0].tolist()}")
print(f"\n  our output    patch1 : {output[1].tolist()}")
print(f"  pytorch output patch1: {pt_output[0, 1].tolist()}")
print(f"\n  match : {torch.allclose(output, pt_output[0], atol=1e-4)}")