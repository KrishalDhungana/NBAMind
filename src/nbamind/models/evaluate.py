"""
==========================================================================================
NBAMind: evaluate.py
------------------------------------------------------------------------------------------
Evaluation script for the Baseline Similarity Model.

This script performs a qualitative assessment of the model by:
1. Fitting the model on the full processed dataset.
2. Running search queries for known, distinct player archetypes (e.g., Curry, Gobert).
3. Displaying the top matches and their similarity scores.
4. explaining the "Why" behind the top match using the model's interpretability layer.

Usage:
    python -m nbamind.models.evaluate
==========================================================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl

from nbamind.models.baseline import BaselineSimilarityModel
from nbamind.models.vae import VAESimilarityModel

# ----------------------------
# Configuration
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
SIMILARITY_FILE = DATA_DIR / "features_similarity.parquet"
PROFILE_FILE = DATA_DIR / "features_profile.parquet"

def load_data() -> pd.DataFrame:
    """
    Loads similarity features and joins them with profile data to get Player Names.
    Returns a Pandas DataFrame ready for the model.
    """
    if not SIMILARITY_FILE.exists():
        logger.error(f"Similarity data not found at {SIMILARITY_FILE}. Run engineering pipeline first.")
        return pd.DataFrame()

    logger.info(f"Loading data from {SIMILARITY_FILE}...")
    df_sim = pl.read_parquet(SIMILARITY_FILE)
    
    if PROFILE_FILE.exists():
        logger.info(f"Loading names from {PROFILE_FILE}...")
        # We only need names from the profile to make the output readable
        df_prof = pl.read_parquet(PROFILE_FILE).select(["PLAYER_ID", "SEASON_YEAR", "PLAYER_NAME", "SALARY"])
        df_sim = df_sim.join(df_prof, on=["PLAYER_ID", "SEASON_YEAR"], how="left")
    else:
        logger.warning("Profile data not found. Player names will be missing in output.")
        df_sim = df_sim.with_columns([
            pl.lit("Unknown").alias("PLAYER_NAME"),
            pl.lit(None).cast(pl.Float64).alias("SALARY")
        ])

    return df_sim.to_pandas()


def print_header(title: str):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def format_currency(val: float) -> str:
    if pd.isna(val) or val == 0:
        return "N/A"
    return f"${val:,.0f}"


def run_evaluation():
    df = load_data()
    if df.empty:
        return

    # 1. Initialize and Fit Model
    # Exclude SALARY from the model features so similarity is based purely on on-court performance.
    # We keep the full 'df' for metadata lookups (Names, Salaries).
    train_cols = [c for c in df.columns if c != "SALARY"]
    
    models = {
        "Baseline (PCA)": BaselineSimilarityModel(variance_threshold=0.95),
        "VAE (Neural Net)": VAESimilarityModel(latent_dim=16, epochs=30)
    }

    for name, model in models.items():
        print_header(f"TRAINING MODEL: {name}")
        model.fit(df[train_cols])
        print(f"Input Rows:      {df.shape[0]}")
        print(f"Input Features:  {len(model.feature_names)}")
        if hasattr(model, "pca"):
            print(f"PCA Components:  {model.pca.n_components_}")
            print(f"Explained Var:   {sum(model.pca.explained_variance_ratio_):.2%}")
        elif hasattr(model, "embeddings"):
            print(f"Latent Dim:      {model.latent_dim}")
            print(f"Embeddings:      {model.embeddings.shape}")

    # 2. Evaluation on Random Players
    print_header("EVALUATION: Random Player Samples")
    
    if not df.empty:
        # Sample 5 random players
        sample_size = min(5, len(df))
        targets = df.sample(n=sample_size)

        for _, target in targets.iterrows():
            t_id = target["PLAYER_ID"]
            t_season = target["SEASON_YEAR"]
            t_name = target["PLAYER_NAME"]

            print(f"\nQuery Player: {t_name} ({t_season})")
            
            for name, model in models.items():
                print(f"\n  --- Model: {name} ---")
                try:
                    # Fetch more results to allow for filtering duplicates (same player, different seasons)
                    raw_results = model.search(t_id, t_season, top_n=50)
                    
                    results = []
                    seen_players = set()
                    for res in raw_results:
                        pid = res["PLAYER_ID"]
                        if pid not in seen_players:
                            results.append(res)
                            seen_players.add(pid)
                        if len(results) >= 5:
                            break
                    
                    print(f"  {'-'*76}")
                    print(f"  {'Rank':<5} {'Player':<30} {'Season':<10} {'Score':<10}")
                    print(f"  {'-'*76}")
                    
                    for i, res in enumerate(results):
                        r_name = res.get("PLAYER_NAME", "Unknown")
                        print(f"  {i + 1:<5} {r_name:<30} {res['SEASON_YEAR']:<10} {res['similarity_score']:.4f}")

                    if results:
                        top = results[0]
                        top_name = top.get("PLAYER_NAME", "Unknown")
                        print(f"\n  >> EXPLAINING MATCH: {t_name} <--> {top_name}")
                        
                        explanation = model.explain(t_id, t_season, top["PLAYER_ID"], top["SEASON_YEAR"])
                        
                        if "embedding_drivers" in explanation and explanation["embedding_drivers"]:
                            print("    [Latent Space Drivers (SHAP)]")
                            for t in explanation["embedding_drivers"]:
                                print(f"      * {t['feature']:<30} (Importance={t['importance']:.4f})")

                        print("    [Shared Strengths]")
                        for t in explanation.get("shared_strengths", []):
                            print(f"      * {t['feature']:<30} ({t_name}={t['val_a']:.2f}, {top_name}={t['val_b']:.2f})")

                        if "shared_weaknesses" in explanation:
                            print("    [Shared Weaknesses]")
                            for t in explanation["shared_weaknesses"]:
                                print(f"      * {t['feature']:<30} ({t_name}={t['val_a']:.2f}, {top_name}={t['val_b']:.2f})")

                        print("    [Key Differences]")
                        for t in explanation.get("key_differences", []):
                            print(f"      * {t['feature']:<30} ({t_name}={t['val_a']:.2f}, {top_name}={t['val_b']:.2f})")

                except Exception as e:
                    logger.error(f"Search failed for {t_name} on {name}: {e}")

    # 3. Moneyball Analysis
    print_header("MONEYBALL ANALYSIS: High Production, Low Cost")
    # Find players similar to high-salary stars but at a fraction of the cost.
    
    if "SALARY" in df.columns:
        # Define "Expensive" as top 10% of salaries in the dataset
        salary_threshold = df["SALARY"].quantile(0.90)
        expensive_players = df[df["SALARY"] >= salary_threshold]
        
        if not expensive_players.empty:
            # Pick 3 random expensive players to analyze
            targets = expensive_players.sample(n=min(3, len(expensive_players)))
            
            for _, target in targets.iterrows():
                t_name = target['PLAYER_NAME']
                t_season = target['SEASON_YEAR']
                t_salary = target['SALARY']
                
                print(f"\nTarget: {t_name} ({t_season}) | Salary: {format_currency(t_salary)}")
                
                for name, model in models.items():
                    print(f"  [{name}] Finding bargains (>85% sim, <50% cost)...")
                    try:
                        # Search for top 50 similar players to cast a wide net
                        results = model.search(target["PLAYER_ID"], t_season, top_n=50)
                        
                        found_bargain = False
                        for res in results:
                            # Requirement: Only consider players in the same season
                            if res["SEASON_YEAR"] != t_season:
                                continue

                            # Lookup salary for the match
                            match_row = df[(df["PLAYER_ID"] == res["PLAYER_ID"]) & (df["SEASON_YEAR"] == res["SEASON_YEAR"])]
                            if match_row.empty:
                                continue
                                
                            m_salary = match_row["SALARY"].iloc[0]
                            
                            # Moneyball Criteria: High Similarity (>0.85) AND Low Cost (<50%)
                            if res["similarity_score"] > 0.85 and m_salary < (t_salary * 0.5):
                                savings = t_salary - m_salary
                                m_name = res.get("PLAYER_NAME", "Unknown")
                                print(f"    -> FOUND: {m_name:<25} ({res['SEASON_YEAR']})")
                                print(f"       Score: {res['similarity_score']:.4f} | Cost: {format_currency(m_salary)} ({(m_salary/t_salary):.1%})")
                                found_bargain = True
                        
                        if not found_bargain:
                            print("    -> No distinct bargains found.")
                            
                    except Exception as e:
                        logger.error(f"Moneyball search failed for {t_name} on {name}: {e}")
    else:
        print("Salary data not available for Moneyball analysis.")

if __name__ == "__main__":
    run_evaluation()