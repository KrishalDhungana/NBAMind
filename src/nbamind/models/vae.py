"""
==========================================================================================
NBAMind: vae.py
------------------------------------------------------------------------------------------
Implements a Variational Autoencoder (VAE) for non-linear player similarity.

ARCHITECTURE:
1.  Encoder: Compresses ~50 robust features into a dense latent space (dim=16).
    Captures non-linear interactions (e.g., Usage vs Efficiency trade-offs).
2.  Latent Space: Regularized via KL-Divergence to approximate a standard Gaussian.
    This ensures Euclidean distance in latent space corresponds to statistical likelihood.
3.  Decoder: Reconstructs inputs to ensure the embedding retains semantic meaning.

INTERPRETABILITY:
Leverages SHAP (GradientExplainer) to attribute latent representations back to
original input features, explaining "Why" a player maps to a specific archetype.
==========================================================================================
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Optional SHAP import (graceful fallback if missing)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger(__name__)

# ----------------------------
# Neural Network Architecture
# ----------------------------

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dims: List[int] = None):
        super(VAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # --- Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                )
            )
            in_dim = h_dim
        self.encoder_body = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # --- Decoder ---
        decoder_layers = []
        hidden_dims.reverse()
        in_dim = latent_dim
        for h_dim in hidden_dims:
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1)
                )
            )
            in_dim = h_dim
        
        self.decoder_body = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder_body(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_body(z)
        return self.final_layer(result)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, x, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0) -> dict:
        """
        Computes VAE loss: MSE (Reconstruction) + Beta * KLD (Regularization).
        """
        recons_loss = F.mse_loss(recons, input, reduction='mean')
        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}


# ----------------------------
# Model Wrapper
# ----------------------------

class VAESimilarityModel:
    """
    Production-grade wrapper for VAE training, inference, and explanation.
    """
    def __init__(
        self, 
        latent_dim: int = 16, 
        epochs: int = 50, 
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[VAE] = None
        self.scaler = StandardScaler()
        
        # Metadata storage
        self.ids: pd.DataFrame = pd.DataFrame()
        self.feature_names: List[str] = []
        self._id_to_idx: Dict[Tuple[int, str], int] = {}
        
        # Embeddings cache
        self.embeddings: Optional[np.ndarray] = None
        self.X_scaled: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        
        # SHAP explainer cache
        self._shap_explainer = None
        self._shap_background = None

    def fit(self, df: Union[pl.DataFrame, pd.DataFrame], id_cols: Optional[List[str]] = None) -> "VAESimilarityModel":
        """
        Trains the VAE on the provided player data.
        """
        logger.info(f"Initializing VAE training on {self.device}...")
        
        # 1. Data Prep
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
            
        if id_cols is None:
            id_cols = ["PLAYER_ID", "SEASON_YEAR", "PLAYER_NAME", "TEAM_ABBREVIATION", "SALARY"]
        
        existing_id_cols = [c for c in id_cols if c in df.columns]
        self.ids = df[existing_id_cols].copy().reset_index(drop=True)
        feature_df = df.drop(columns=existing_id_cols)
        self.feature_names = feature_df.columns.tolist()

        # Index for fast lookup
        if "PLAYER_ID" in self.ids.columns and "SEASON_YEAR" in self.ids.columns:
            self._id_to_idx = {
                (row.PLAYER_ID, row.SEASON_YEAR): idx 
                for idx, row in self.ids.iterrows()
            }

        # Scale Data (StandardScaler on top of Robust Z-scores ensures NN stability)
        self.X_scaled = self.scaler.fit_transform(feature_df).astype(np.float32)
        
        # Convert to Tensor
        dataset = TensorDataset(torch.from_numpy(self.X_scaled))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2. Model Init
        input_dim = self.X_scaled.shape[1]
        self.model = VAE(input_dim=input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, steps_per_epoch=len(dataloader), epochs=self.epochs
        )

        # 3. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            # KL Annealing: Slowly increase KL weight to prevent posterior collapse
            kld_weight = min(1.0, (epoch / (self.epochs * 0.5))) 
            
            for batch in dataloader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                
                recons, input_x, mu, log_var = self.model(x)
                loss_dict = self.model.loss_function(recons, input_x, mu, log_var, kld_weight=kld_weight)
                
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss_dict['loss'].item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | KLD Weight: {kld_weight:.2f}")

        # 4. Generate Embeddings (Inference Mode)
        self.model.eval()
        with torch.no_grad():
            full_tensor = torch.from_numpy(self.X_scaled).to(self.device)
            mu, _ = self.model.encode(full_tensor)
            self.embeddings = mu.cpu().numpy()

        self._is_fitted = True
        logger.info(f"VAE Fitted. Embeddings shape: {self.embeddings.shape}")
        return self

    def search(
        self, 
        player_id: int, 
        season: str, 
        top_n: int = 10,
        exclude_self_history: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Performs vector search in the latent space.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        idx = self._id_to_idx.get((player_id, season))
        if idx is None:
            raise ValueError(f"Player {player_id} ({season}) not found.")

        query_vec = self.embeddings[idx].reshape(1, -1)
        
        # Euclidean distance in VAE latent space corresponds to likelihood
        dists = euclidean_distances(query_vec, self.embeddings).flatten()
        
        # Convert distance to similarity score (Gaussian Kernel)
        # Sigma is heuristic; median distance is usually good, hardcoded 2.0 for latent space stability
        sigma = 2.0 
        scores = np.exp(-(dists ** 2) / (2 * sigma ** 2))

        # Sort
        candidate_indices = np.argsort(scores)[::-1]
        
        results = []
        for cand_idx in candidate_indices:
            if len(results) >= top_n:
                break
            
            # Skip exact self
            if cand_idx == idx:
                continue
                
            cand_row = self.ids.iloc[cand_idx]
            
            # Skip same player, different season (optional)
            if exclude_self_history and cand_row["PLAYER_ID"] == player_id:
                continue
                
            res = cand_row.to_dict()
            res["similarity_score"] = float(scores[cand_idx])
            res["euclidean_distance"] = float(dists[cand_idx])
            results.append(res)
            
        return results

    def explain(self, player_id_a: int, season_a: str, player_id_b: int, season_b: str) -> Dict[str, Any]:
        """
        Explains the similarity using SHAP (if available) and raw feature comparison.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        idx_a = self._id_to_idx.get((player_id_a, season_a))
        idx_b = self._id_to_idx.get((player_id_b, season_b))
        
        if idx_a is None or idx_b is None:
            raise ValueError("Player not found.")

        # --- 1. Raw Feature Comparison (Human Readable) ---
        vec_a = self.X_scaled[idx_a]
        vec_b = self.X_scaled[idx_b]
        
        diffs = np.abs(vec_a - vec_b)
        sums = vec_a + vec_b
        
        df_exp = pd.DataFrame({
            "feature": self.feature_names,
            "val_a": vec_a,
            "val_b": vec_b,
            "diff": diffs,
            "sum": sums
        })
        
        # Shared Strengths: Both high (>0.5 Z) and close
        strengths = df_exp[(df_exp["val_a"] > 0.5) & (df_exp["val_b"] > 0.5)].copy()
        strengths["relevance"] = strengths["sum"] / (1.0 + strengths["diff"])
        shared_strengths = strengths.sort_values("relevance", ascending=False).head(5)
        
        # Key Differences
        key_diffs = df_exp.sort_values("diff", ascending=False).head(3)

        # --- 2. SHAP Explanation of Latent Representation ---
        shap_values = None
        if HAS_SHAP:
            try:
                # Lazy init SHAP explainer
                if self._shap_explainer is None:
                    # Use a background sample of 100 random players
                    rng = np.random.RandomState(42)
                    bg_indices = rng.choice(len(self.X_scaled), 100, replace=False)
                    self._shap_background = torch.from_numpy(self.X_scaled[bg_indices]).to(self.device)
                    
                    # Explain the Encoder's Mu output
                    # We wrap the encoder to return just Mu for SHAP
                    class EncoderWrapper(nn.Module):
                        def __init__(self, vae):
                            super().__init__()
                            self.vae = vae
                        def forward(self, x):
                            mu, _ = self.vae.encode(x)
                            return mu
                    
                    self._shap_explainer = shap.GradientExplainer(
                        EncoderWrapper(self.model), 
                        self._shap_background
                    )

                # Compute SHAP for Player A
                input_tensor = torch.from_numpy(vec_a.reshape(1, -1)).to(self.device)
                # shap_vals is list of arrays (one per latent dim)
                shap_vals = self._shap_explainer.shap_values(input_tensor)
                
                # Aggregate importance: Sum of absolute SHAP values across all latent dims
                # This tells us: "Which features contributed most to this player's embedding?"
                feature_importance = np.sum([np.abs(s[0]) for s in shap_vals], axis=0)
                top_indices = np.argsort(feature_importance)[::-1][:5]
                
                shap_values = [
                    {"feature": self.feature_names[i], "importance": float(feature_importance[i])}
                    for i in top_indices
                ]
            except Exception as e:
                logger.warning(f"SHAP calculation failed: {e}")

        return {
            "shared_strengths": shared_strengths.to_dict(orient="records"),
            "key_differences": key_diffs.to_dict(orient="records"),
            "embedding_drivers": shap_values  # The "Why" behind the embedding
        }