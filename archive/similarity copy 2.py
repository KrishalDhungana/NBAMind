from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pandas as pd
import numpy as np
import polars as pl
"""
Vunerabilities to keep in mind:
1. correlation between features is certainly apparent, and we must either explicitly or implicitly address this
2. not all features have the same importance (e.g., similar STL_PCT shouldn't be an equal driver of player similarity as similar box_creation)
2. although all features are scaled using a robust Z score, we need to identify whether or not this handles cases where a 0.1 difference in STL_PCT is treated the same as a 0.1 difference in ts_pct_rel

Use cases of similarity engine:
1. "Moneyball" arbitrage engine (cost efficiency) for injuries & trades: find player X, who has 90% similarity score to player Y but costs 15% of the salary
2. Insights: users can explore and find interesting comparisons, including detailed descriptions of why player X is similar to player Y; should also include detailed profiles of each player showing their play style, production stats, etc.
3. League leaders section: most ball dominant, best defensive, top players for each archetype (e.g., 3-and-D, rimrunning big)

Idea for baseline:
- allow user to adjust sliders for each bucket: Scoring Gravity, Shot Diet, Playmaking, Defense/Rim, Physical/Hustle

2. The "Standout" Approach: Variational Autoencoder (VAE) + Vector Search + SHAPTo impress a modern Data Science team, you need to move beyond linear algebra (PCA) and demonstrate competence in Deep Learning, Latent Representations, and MLOps.The Concept:We will train a Neural Network to "compress" a player's 70 features into a dense, lower-dimensional "DNA" (Latent Space).Why this outperforms the Baseline:PCA is linear. It assumes the relationship between features is a straight line. Basketball is non-linear.Example: The relationship between Usage and Efficiency is a curve (efficiency drops as usage becomes extreme). PCA flattens this; a Neural Network captures the curve.The Architecture (Recruiter "Wow" Factor)A. The Model: Variational Autoencoder (VAE)Input: 70 Features.Encoder: Compresses features into a small latent vector (e.g., size 8 or 16). This vector is the "Player Embedding."Decoder: Attempts to reconstruct the original 70 stats from the embedding.Loss Function: Reconstruction Loss (MSE) + KL Divergence (ensures the latent space is continuous and valid).B. The Storage: Vector Database (e.g., FAISS, Pinecone, or DuckDB)Instead of calculating a massive correlation matrix every time (which is $O(N^2)$), you store the VAE Embeddings in a Vector Database.This demonstrates you understand Production Engineering. You are building a system that can scale to historical comparisons instantly.C. The Explanation: SHAP (SHapley Additive exPlanations)Deep Learning is usually a "Black Box." Recruiters hate Black Boxes.You will use KernelSHAP to explain the similarity.Narrative: "My VAE identified Player B is similar to Player A. SHAP analysis shows the primary drivers were their identical 'Rim Deterrence' and 'Drive Frequency,' despite different scoring averages."Summary of ApproachesFeatureBaseline (PCA + Cosine)Standout (VAE + Vector DB)MathLinear Algebra (Orthogonal transformation)Non-Linear Manifold Learning (Neural Net)CollinearitySolved via Dimensionality ReductionSolved via Compression/encodingComplexityLow (Scikit-Learn)High (PyTorch/TensorFlow + Vector Store)Recruiter Signal"Solid Analyst. Knows stats.""ML Engineer. Can build production AI systems."VulnerabilityFails to capture complex non-linear interactions (e.g., usage vs. efficiency curves).Harder to interpret without SHAP; requires more data tuning.

"""

# Assume df_sim is your Polars DataFrame converted to Pandas
df_sim = pl.read_parquet("data/processed/features_similarity.parquet")
df_sim = df_sim.to_pandas()
features = [c for c in df_sim.columns if c not in ["PLAYER_ID", "SEASON_YEAR"]]
X = df_sim[features].values

# 1. Standardize (Your Robust Z-Scores are essentially this, but ensure unit variance)
# PCA is sensitive to scale.
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)

# 2. PCA for Dimensionality Reduction (Handling Multicollinearity)
# Keep 95% of variance
pca = PCA(n_components=0.80)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")

# 3. Calculate Distance Matrix (Euclidean on PCA = Mahalanobis on Raw)
dist_matrix = cosine_similarity(X_pca, X_pca)

# 4. Interpretability: Get Feature Loadings for PC1 and PC2
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], 
    index=features
)
print("Top features driving variance (PC1):")
for i in loadings:
    print(loadings[i].abs().sort_values(ascending=False)[:5])
    # print(loadings['PC1'].abs().sort_values(ascending=False).head(5))