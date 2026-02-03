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
    A robust, PCA-based similarity engine for NBA players.
    
    Attributes:
        variance_threshold (float): Variance to retain in PCA (0.0-1.0).
        ids (pd.DataFrame): Metadata for each player/season in the fitted set.
        feature_names (List[str]): Names of the numerical features used.
        X_scaled (np.ndarray): Standardized feature matrix (Z-scores).
        X_pca (np.ndarray): Latent space representations.
        sigma (float): RBF Kernel width for score normalization.
        _id_to_idx (Dict): Fast lookup mapping (player_id, season) -> index.
    """

    def __init__(self, variance_threshold: float = 0.90):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold)
        
        # Data storage
        self.ids: pd.DataFrame = pd.DataFrame()
        self.feature_names: List[str] = []
        self.X_scaled: Optional[np.ndarray] = None
        self.X_pca: Optional[np.ndarray] = None
        
        # Model parameters
        self.sigma: float = 1.0
        self._id_to_idx: Dict[Tuple[int, str], int] = {}
        self._is_fitted: bool = False

    def fit(self, df: Union[pl.DataFrame, pd.DataFrame], id_cols: Optional[List[str]] = None) -> "BaselineSimilarityModel":
        """
        Fits the model pipeline on the provided dataset.
        
        Args:
            df: Input DataFrame (Polars or Pandas).
            id_cols: List of column names to treat as metadata (not features). 
                     Defaults to standard identifiers.
        """
        logger.info("Fitting Baseline Similarity Model...")

        # Standardization & setup
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        
        # Identify metadata vs features
        if id_cols is None:
            id_cols = ["PLAYER_ID", "SEASON_YEAR", "PLAYER_NAME", "TEAM_ABBREVIATION", "SALARY"]
        existing_id_cols = [c for c in id_cols if c in df.columns]
        self.ids = df[existing_id_cols].copy().reset_index(drop=True)
        feature_df = df.drop(columns=existing_id_cols)
        self.feature_names = feature_df.columns.tolist()
        
        # Create fast lookup index
        if "PLAYER_ID" in self.ids.columns and "SEASON_YEAR" in self.ids.columns:
            self._id_to_idx = {
                (row.PLAYER_ID, row.SEASON_YEAR): idx 
                for idx, row in self.ids.iterrows()
            }

        # Pipeline execution: standardize (Z-Score) -> PCA
        self.X_scaled = self.scaler.fit_transform(feature_df)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        # Create sample to auto-tune Sigma (RBF Kernel Width)
        n_samples = min(5000, len(self.X_pca))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.X_pca), n_samples, replace=False)
        sample_vecs = self.X_pca[indices]
        
        # Set Sigma using sample pairwise distances; we want the similarity score to be meaningful, so a score of 0.5 should represent a "median" level of similarity in the league
        dists = euclidean_distances(sample_vecs, sample_vecs)
        flat_dists = dists[np.triu_indices_from(dists, k=1)]
        if len(flat_dists) > 0:
            self.sigma = float(np.median(flat_dists))
        else:
            self.sigma = 1.0
        self._is_fitted = True
        
        # Logging
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"Model Fitted: {self.X_pca.shape[0]} players, {self.X_pca.shape[1]} components. "
            f"Explained Variance: {explained_var:.1%}. Sigma: {self.sigma:.3f}"
        )
        
        return self

    def search(
        self, 
        player_id: int, 
        season: str, 
        top_n: int = 10,
        exclude_self_history: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Finds nearest neighbors in the latent PCA space.
        
        Args:
            player_id: Target Player ID.
            season: Target Season Year.
            top_n: Number of matches to return.
            exclude_self_history: If True, excludes other seasons of the same player.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        # Get query vector
        idx = self._id_to_idx.get((player_id, season))
        if idx is None:
            raise ValueError(f"Player {player_id} ({season}) not found in index.")
        query_vec = self.X_pca[idx].reshape(1, -1)

        # Compute distances and convert to similarity scores (0.0 to 1.0) using RBF Kernel: exp(-d^2 / (2*sigma^2))
        dists = euclidean_distances(query_vec, self.X_pca).flatten()
        scores = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))

        # Rank and filter top_n * 3 candidates (to allow space for filtering self & duplicates)
        candidate_indices = np.argsort(scores)[::-1]
        results = []
        for cand_idx in candidate_indices:
            if len(results) >= top_n:
                break
            if cand_idx == idx: # Skip exact self match (distance 0)
                continue
            cand_row = self.ids.iloc[cand_idx]
            if exclude_self_history and cand_row["PLAYER_ID"] == player_id: # Skip same player in different seasons (if specified)
                continue
            res = cand_row.to_dict()
            res["similarity_score"] = float(scores[cand_idx])
            res["euclidean_distance"] = float(dists[cand_idx])
            results.append(res)
            
        return results

    def explain(self, player_id_a: int, season_a: str, player_id_b: int, season_b: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates a human-readable explanation of the similarity (or difference) between two players.
        Uses the raw scaled feature space (Z-scores) to identify shared traits.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        # Fetch both player vectors
        idx_a = self._id_to_idx.get((player_id_a, season_a))
        idx_b = self._id_to_idx.get((player_id_b, season_b))
        if idx_a is None or idx_b is None:
            raise ValueError("One or both players not found in index.")
        vec_a = self.X_scaled[idx_a]
        vec_b = self.X_scaled[idx_b]
        
        # Metrics for Explanation
        # 1. Magnitude: How "strong" is this trait for both? (Product of Z-scores)
        # 2. Diff: How different are they? (Absolute difference)
        
        # Shared Strengths: Both are high (>0.5 Z) and close
        # We define a "Strength Score" = (Min(A, B) / (1 + Diff)) * Indicator(A>0, B>0)
        # Actually, simple heuristic: Sort by (A+B) where |A-B| is small.
        
        diffs = np.abs(vec_a - vec_b)
        sums = vec_a + vec_b
        df_exp = pd.DataFrame({
            "feature": self.feature_names,
            "val_a": vec_a,
            "val_b": vec_b,
            "diff": diffs,
            "sum": sums
        })
        
        # Shared Strengths: both positive, low difference (sort by 'sum' descending, filter for diff < 1.0)
        strengths = df_exp[
            (df_exp["val_a"] > 0.5) & 
            (df_exp["val_b"] > 0.5)
        ].copy()
        strengths["relevance"] = strengths["sum"] / (1.0 + strengths["diff"])
        shared_strengths = strengths.sort_values("relevance", ascending=False).head(5)
        
        # Shared Weaknesses: both negative, low difference
        weaknesses = df_exp[
            (df_exp["val_a"] < -0.5) & 
            (df_exp["val_b"] < -0.5)
        ].copy()
        weaknesses["relevance"] = np.abs(weaknesses["sum"]) / (1.0 + weaknesses["diff"])
        shared_weaknesses = weaknesses.sort_values("relevance", ascending=False).head(3)
        
        # Key Differences: high difference (we want features where one is high and the other is low/average)
        key_diffs = df_exp.sort_values("diff", ascending=False).head(3)
        
        return {
            "shared_strengths": shared_strengths.to_dict(orient="records"),
            "shared_weaknesses": shared_weaknesses.to_dict(orient="records"),
            "key_differences": key_diffs.to_dict(orient="records")
        }