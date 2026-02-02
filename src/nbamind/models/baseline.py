"""
==========================================================================================
NBAMind: baseline.py
------------------------------------------------------------------------------------------
Implements a robust baseline for Player Similarity using PCA and Euclidean Distance.

RATIONALE:
1.  Curse of Dimensionality: With ~70 features, distance metrics in raw space become
    meaningless as contrast diminishes. PCA reduces this to ~15-25 dense components.
2.  Collinearity: Features like USG_PCT and FGA_Per100 are highly correlated.
    PCA collapses these into single orthogonal components, preventing "double counting"
    of specific skills (e.g., volume scoring).
3.  Impact vs Style: We use Euclidean Distance (L2) rather than Cosine Similarity.
    Cosine normalizes vectors, discarding magnitude (Production/Impact). Euclidean
    respects that a Superstar (high magnitude) is distinct from a Role Player (low magnitude)
    even if their playstyle ratios are similar.
==========================================================================================
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaselineSimilarityModel:
    """
    A production-grade baseline model using PCA-reduced latent space for search
    and raw feature space for explanation.
    """

    def __init__(self, variance_threshold: float = 0.95):
        """
        Args:
            variance_threshold: The amount of variance to retain in PCA (0.0 to 1.0).
                                Lower values filter more noise but might lose signal.
                                0.95 is a conservative baseline.
        """
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)
        
        # State
        self.ids: Optional[pd.DataFrame] = None  # Stores PLAYER_ID, SEASON_YEAR
        self.feature_names: List[str] = []
        self.X_raw: Optional[np.ndarray] = None  # Scaled raw features (for explanation)
        self.X_pca: Optional[np.ndarray] = None  # Latent features (for search)
        self._is_fitted = False

    def fit(self, df: Union[pl.DataFrame, pd.DataFrame]) -> "BaselineSimilarityModel":
        """
        Fits the scaler and PCA on the provided data.
        Expects the input DF to contain 'PLAYER_ID' and 'SEASON_YEAR'.
        """
        logger.info("Fitting Baseline Similarity Model...")
        
        # Convert to Pandas for Scikit-Learn compatibility
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Separate Metadata from Features
        id_cols = ["PLAYER_ID", "SEASON_YEAR", "PLAYER_NAME"]
        # Handle case where PLAYER_NAME might not be in the similarity parquet
        self.ids = df[[c for c in id_cols if c in df.columns]].copy().reset_index(drop=True)
        
        feature_data = df.drop(columns=[c for c in df.columns if c in id_cols])
        self.feature_names = feature_data.columns.tolist()

        # 1. Standardize
        # Even though input is RobustScaled, we enforce Unit Variance for PCA.
        # This ensures 'offensive_load' (range -4 to 4) doesn't dominate 'freq_dunk'
        # (range 0 to 0.1) purely due to scale differences.
        self.X_raw = self.scaler.fit_transform(feature_data)

        # 2. PCA
        self.X_pca = self.pca.fit_transform(self.X_raw)
        self._is_fitted = True

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        n_components = self.pca.n_components_
        logger.info(
            f"PCA Fit Complete: Retained {n_components} components explaining "
            f"{explained_var:.1%} of variance from {len(self.feature_names)} features."
        )
        
        return self

    def search(
        self, 
        player_id: int, 
        season: str, 
        top_n: int = 10
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Finds the most similar players to the query player/season.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before searching.")

        # Locate Query Index
        query_mask = (self.ids["PLAYER_ID"] == player_id) & (self.ids["SEASON_YEAR"] == season)
        if not query_mask.any():
            raise ValueError(f"Player {player_id} ({season}) not found in dataset.")
        
        query_idx = query_mask.idxmax() # Returns integer index
        query_vec = self.X_pca[query_idx].reshape(1, -1)

        # Compute Euclidean Distance (Vectorized)
        # shape: (1, n_samples)
        dists = euclidean_distances(query_vec, self.X_pca).flatten()

        # Convert to Similarity Score (Inverse Distance)
        # 1.0 = Identical. As distance grows, score -> 0.
        sim_scores = 1.0 / (1.0 + dists)

        # Get Top N (excluding self)
        # argsort returns indices of sorted array (ascending), so we take tail
        top_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        count = 0
        for idx in top_indices:
            if idx == query_idx:
                continue # Skip self

            # Skip same player ID (different seasons)
            if self.ids.iloc[idx]["PLAYER_ID"] == player_id:
                continue
            
            if count >= top_n:
                break

            rec = self.ids.iloc[idx].to_dict()
            rec["similarity_score"] = float(sim_scores[idx])
            results.append(rec)
            count += 1
            
        return results

    def explain_match(self, player_id_a: int, season_a: str, player_id_b: int, season_b: str) -> Dict[str, Any]:
        """
        Provides an interpretable explanation for why two players are similar.
        
        Methodology:
        We look for "Shared Archetypes" in the raw feature space.
        A feature is a "Shared Strength" if both players have high Z-scores (> 0.5) 
        and the values are close.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before explaining.")

        # Get Indices
        idx_a = self.ids[(self.ids["PLAYER_ID"] == player_id_a) & (self.ids["SEASON_YEAR"] == season_a)].index[0]
        idx_b = self.ids[(self.ids["PLAYER_ID"] == player_id_b) & (self.ids["SEASON_YEAR"] == season_b)].index[0]

        # Get Raw Scaled Vectors (Z-scores)
        vec_a = self.X_raw[idx_a]
        vec_b = self.X_raw[idx_b]

        # 1. Calculate "Shared Signal"
        # Logic: If both are +2.0 (High), product is 4.0. If both are -2.0 (Low), product is 4.0.
        # If one is +2 and one is -2, product is -4.0 (Mismatch).
        # We weight this by how close they are (1 / (1 + abs_diff)).
        
        abs_diff = np.abs(vec_a - vec_b)
        magnitude_signal = (vec_a * vec_b) # High if both are outliers in same direction
        closeness_weight = 1.0 / (1.0 + abs_diff)
        
        # Explanation Score: High magnitude (both outliers) * High closeness
        explanation_score = magnitude_signal * closeness_weight

        # Create DataFrame for sorting
        df_exp = pd.DataFrame({
            "feature": self.feature_names,
            "val_a": vec_a,
            "val_b": vec_b,
            "score": explanation_score,
            "diff": abs_diff
        })

        # Extract Insights
        # Filter for actual strengths (both > 0) vs weaknesses (both < 0)
        shared_strengths = df_exp[(df_exp["score"] > 0) & (df_exp["val_a"] > 0)].sort_values("score", ascending=False).head(5)
        shared_weaknesses = df_exp[(df_exp["score"] > 0) & (df_exp["val_a"] < 0)].sort_values("score", ascending=False).head(3)
        
        divergent_traits = df_exp.sort_values("diff", ascending=False).head(3)

        return {
            "shared_strengths": shared_strengths[["feature", "val_a", "val_b"]].to_dict(orient="records"),
            "shared_weaknesses": shared_weaknesses[["feature", "val_a", "val_b"]].to_dict(orient="records"),
            "key_differences": divergent_traits[["feature", "val_a", "val_b"]].to_dict(orient="records")
        }