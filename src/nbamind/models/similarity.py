"""
==========================================================================================
NBAMind: similarity.py
------------------------------------------------------------------------------------------
This module handles the logic for the Player Similarity Engine.
Currently, it includes data quality checks to ensure the input features are ready for
dimensionality reduction (PCA) and clustering (K-Means).
==========================================================================================
"""

import logging
import sys
from pathlib import Path

import polars as pl

# ----------------------------
# Configuration
# ----------------------------

DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "features_similarity.parquet"

# ----------------------------
# Data Quality Checks
# ----------------------------

def check_data_quality(df: pl.DataFrame) -> bool:
    """
    Performs comprehensive data quality checks on the similarity features DataFrame.
    Returns True if data looks good (warnings might be logged), False if critical issues found.
    """
    logging.info(f"Starting data quality check on {df.shape[0]} rows and {df.shape[1]} columns...")
    
    issues_found = False

    # 0. Quick visual
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    print(df.shape)
    print(df.columns)
    for col in df.columns:
        print(f"\n--- {col} ---")
        try:
            stats = df.select([
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).null_count().alias("nulls")
            ]).row(0)
            print(f"Min: {stats[0]}, Max: {stats[1]}, Missing: {stats[2]}")
            
            mode_df = df[col].value_counts(sort=True).head(1)
            if not mode_df.is_empty():
                print(f"Most Common: {mode_df[0, 0]} (Count: {mode_df[0, 1]})")
        except Exception as e:
            print(f"Error computing stats: {e}")
    
    # 1. Identify Column Types
    id_cols = {"PLAYER_ID", "SEASON_YEAR"}
    feature_cols = [c for c in df.columns if c not in id_cols]
    
    # 2. Check for Nulls
    logging.info("Checking for missing values...")
    null_counts = df.select([pl.col(c).null_count() for c in df.columns]).row(0)
    cols_with_nulls = {
        df.columns[i]: count 
        for i, count in enumerate(null_counts) 
        if count > 0
    }
    
    if cols_with_nulls:
        logging.warning(f"Found {len(cols_with_nulls)} columns with null values:")
        for col, count in sorted(cols_with_nulls.items(), key=lambda x: x[1], reverse=True):
            pct = (count / df.height) * 100
            logging.warning(f"  - {col}: {count} nulls ({pct:.2f}%)")
        logging.error("CRITICAL: PCA/K-Means cannot handle missing values. These must be imputed or dropped.")
        issues_found = True
    else:
        logging.info("No missing values found.")

    # 2.5 Check for NaNs (Floating point errors)
    logging.info("Checking for NaN values...")
    nan_counts = df.select([
        pl.col(c).is_nan().sum().alias(c) for c in feature_cols
    ]).row(0)
    
    cols_with_nans = {
        feature_cols[i]: count 
        for i, count in enumerate(nan_counts) 
        if count > 0
    }
    
    if cols_with_nans:
        logging.error(f"Found {len(cols_with_nans)} columns with NaN values:")
        for col, count in cols_with_nans.items():
            logging.error(f"  - {col}: {count} NaNs")
        issues_found = True
    else:
        logging.info("No NaN values found.")

    # 3. Check for Infinite Values
    logging.info("Checking for infinite values...")
    # Polars expression to sum is_infinite across feature columns
    inf_counts = df.select([
        pl.col(c).is_infinite().sum().alias(c) for c in feature_cols
    ]).row(0)
    
    cols_with_inf = {
        feature_cols[i]: count 
        for i, count in enumerate(inf_counts) 
        if count > 0
    }
    
    if cols_with_inf:
        logging.error(f"Found {len(cols_with_inf)} columns with infinite values:")
        for col, count in cols_with_inf.items():
            logging.error(f"  - {col}: {count} inf values")
        issues_found = True
    else:
        logging.info("No infinite values found.")

    # 4. Check for Zero Variance (Constant Columns)
    logging.info("Checking for zero-variance (constant) features...")
    stats = df.select([
        pl.col(c).std().alias(c) for c in feature_cols
    ]).row(0)
    
    constant_cols = [
        feature_cols[i] for i, std in enumerate(stats) 
        if std is not None and std == 0
    ]
    
    if constant_cols:
        logging.warning(f"Found {len(constant_cols)} constant columns (will provide no signal):")
        for col in constant_cols:
            logging.warning(f"  - {col}")
    else:
        logging.info("No constant columns found.")

    # 5. Check Distributions (Z-Score Verification)
    # Since engineering.py applies Z-scores, we expect Mean ~ 0 and Std ~ 1.
    logging.info("Checking feature distributions (Expect Mean~0, Std~1)...")
    
    dist_stats = df.select([
        pl.col(c).mean().alias(f"{c}_mean") for c in feature_cols
    ] + [
        pl.col(c).std().alias(f"{c}_std") for c in feature_cols
    ]).to_dict(as_series=False)
    
    abnormal_dist = []
    for col in feature_cols:
        mean = dist_stats[f"{col}_mean"][0]
        std = dist_stats[f"{col}_std"][0]
        
        # Thresholds: Mean should be close to 0, Std close to 1
        if abs(mean) > 0.5 or not (0.5 < std < 1.5):
            abnormal_dist.append((col, mean, std))
            
    if abnormal_dist:
        logging.warning(f"Found {len(abnormal_dist)} features with suspicious distributions (potential normalization issues):")
        for col, m, s in abnormal_dist[:10]: # Limit output
            logging.warning(f"  - {col}: Mean={m:.2f}, Std={s:.2f}")
        if len(abnormal_dist) > 10:
            logging.warning(f"  ... and {len(abnormal_dist) - 10} more.")
    else:
        logging.info("Feature distributions look healthy.")

    # 6. Check for Duplicates
    logging.info("Checking for duplicate player-season keys...")
    if "PLAYER_ID" in df.columns and "SEASON_YEAR" in df.columns:
        n_dupes = df.select(["PLAYER_ID", "SEASON_YEAR"]).is_duplicated().sum()
        if n_dupes > 0:
            logging.error(f"Found {n_dupes} duplicate entries for (PLAYER_ID, SEASON_YEAR).")
            issues_found = True
        else:
            logging.info("Keys are unique.")

    return not issues_found

def main():
    if not INPUT_FILE.exists():
        logging.error(f"Input file not found at {INPUT_FILE}. Run the engineering pipeline first.")
        sys.exit(1)
        
    try:
        df = pl.read_parquet(INPUT_FILE)

        # Temporary filter for specific seasons
        logging.info("Filtering for 2015-16 and 2016-17 seasons only...")
        df = df.filter(pl.col("SEASON_YEAR").is_in(["2015-16", "2016-17"]))

        success = check_data_quality(df)
        
        if success:
            logging.info(">> Data Quality Check PASSED. Ready for modeling.")
        else:
            logging.error(">> Data Quality Check FAILED. Please review logs.")
            sys.exit(1)
            
    except Exception as e:
        logging.exception(f"An error occurred during data checks: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()