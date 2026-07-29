"""
CodeToAGI — ML Series EP98 Challenge Solution
Apply PCA to a High-Dimensional Dataset

Challenge:
  1) Pick any dataset with 4+ numeric features (Iris, Wine, or your own CSV).
  2) Standardize the features first (StandardScaler) — don't skip this.
  3) Fit PCA and print explained_variance_ratio_ for each component.
  4) Plot cumulative explained variance and pick n_components for 95%.
  5) Project onto 2 components and scatter-plot colored by the true label.
  6) Write one sentence: how much variance did you keep, and did classes separate?

Run:
  python ep98_challenge_solution.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 1) Load a dataset with 4+ numeric features ───────────────────────────────
# Wine has 13 numeric features and 3 classes — perfect for this challenge.
data = load_wine()
X = data.data          # shape (178, 13)
y = data.target        # 0, 1, 2
feature_names = data.feature_names
target_names = data.target_names

print(f"Dataset: Wine")
print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Classes: {len(target_names)}")
print(f"Features: {list(feature_names)}\n")

# ── 2) Standardize first (CRITICAL — PCA is variance-based) ──────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3) Fit PCA (keep all components first so we can inspect the ratios) ──────
pca_full = PCA()  # n_components = min(n_samples, n_features) by default
X_pca_full = pca_full.fit_transform(X_scaled)

ratios = pca_full.explained_variance_ratio_
print("Explained variance ratio per component:")
for i, r in enumerate(ratios):
    print(f"  PC{i+1}: {r:.4f}  ({r*100:.1f}%)")

# ── 4) Cumulative variance + pick n_components for ≥ 95% ─────────────────────
cumulative = np.cumsum(ratios)
n_95 = int(np.argmax(cumulative >= 0.95) + 1)  # first index that crosses 95%

print(f"\nCumulative explained variance:")
for i, c in enumerate(cumulative):
    marker = "  ← 95% threshold" if i + 1 == n_95 else ""
    print(f"  up to PC{i+1}: {c:.4f}  ({c*100:.1f}%){marker}")

print(f"\n→ Keep n_components = {n_95} to retain ≥ 95% of variance "
      f"({cumulative[n_95-1]*100:.1f}% kept).")

# Scree + cumulative plot
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(1, len(ratios) + 1)
ax.bar(x_pos, ratios * 100, color="#8be9fd", alpha=0.85, label="Individual")
ax.plot(x_pos, cumulative * 100, "o-", color="#f1fa8c", linewidth=2, label="Cumulative")
ax.axhline(95, color="#ff5555", linestyle="--", linewidth=1.5, label="95% cutoff")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("Scree Plot — Wine Dataset")
ax.set_xticks(x_pos)
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("ep98_scree_plot.png", dpi=150)
print("\nScree plot saved → ep98_scree_plot.png")
plt.show()

# ── 5) Project to 2D and scatter colored by true label ───────────────────────
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
var_kept_2d = pca_2d.explained_variance_ratio_.sum()

colors = ["#8be9fd", "#50fa7b", "#ffb86c"]  # cyan, green, orange
fig, ax = plt.subplots(figsize=(8, 6))
for cls, color, name in zip(range(3), colors, target_names):
    mask = y == cls
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name,
               alpha=0.8, edgecolors="none", s=50)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"Wine — PCA 2D Projection ({var_kept_2d*100:.1f}% variance kept)")
ax.legend()
ax.axhline(0, color="#6272a4", linewidth=0.6)
ax.axvline(0, color="#6272a4", linewidth=0.6)
plt.tight_layout()
plt.savefig("ep98_pca_2d_scatter.png", dpi=150)
print("2D scatter saved → ep98_pca_2d_scatter.png")
plt.show()

# ── 6) One-sentence summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print(
    f"SUMMARY: With 2 components we kept {var_kept_2d*100:.1f}% of the variance; "
    f"the three wine classes form clearly visible clusters in the PCA plane."
)
print("=" * 70)

# Optional: show how many components for exactly 95%
print(f"\nFor ≥95% variance you would keep {n_95} components "
      f"({cumulative[n_95-1]*100:.1f}%).")
