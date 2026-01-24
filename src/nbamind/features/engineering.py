"""
==========================================================================================
NBAMind: engineering.py
------------------------------------------------------------------------------------------
This script consumes the master player analytics table and produces two outputs:
1. features_similarity.parquet: High-signal, normalized features for the similarity engine.
2. features_profile.parquet: Rich, descriptive features for player profile UI (Base, Advanced, Playstyle).

It handles unit conversions (PerGame <-> Per100), season-level normalizations (Z-scores),
and complex feature derivation (Offensive Load, Box Creation, etc.) in a single pass.
==========================================================================================
"""

import logging
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import polars as pl

# ----------------------------
# Configuration
# ----------------------------

DATA_DIR = Path("data/processed")
INPUT_FILE = DATA_DIR / "master_player_analytics.parquet"
OUTPUT_SIMILARITY = DATA_DIR / "features_similarity.parquet"
OUTPUT_PROFILE = DATA_DIR / "features_profile.parquet"

# Columns strictly required from the master file
# Expanded to include raw stats for the profile view
REQUIRED_COLUMNS = [
    # Identifiers
    "PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "PLAYER_POSITION",
    # Base Stats (Per Game / Rates)
    "PTS_PerGame", "AST_PerGame", "REB_PerGame", "STL_PerGame", "BLK_PerGame",
    "MIN1_PerGame", "PACE", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS_Per100Possessions",
    # Per 100
    "PTS_Per100Possessions", "AST_Per100Possessions", "FGA_Per100Possessions", "FTA_Per100Possessions",
    "TOV_Per100Possessions", "FG3A_Per100Possessions", "FG3M_Per100Possessions", "OREB_Per100Possessions", "DREB_Per100Possessions",
    "STL_Per100Possessions", "BLK_Per100Possessions", "PF_Per100Possessions",
    # Advanced / Rates
    "TS_PCT", "USG_PCT", "AST_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING", "PIE", "POTENTIAL_AST_PerGame",
    "OREB_PCT", "DREB_PCT", "PCT_BLK", "PCT_STL", "PCT_PLUSMINUS",
    # Shooting Zones (Volumes & Efficiency)
    "FGA_Per100Possessions_Restricted_Area", "FG_PCT_Restricted_Area", "FGM_Per100Possessions_Restricted_Area",
    "FGA_Per100Possessions_In_The_Paint_(Non_RA)", "FG_PCT_In_The_Paint_(Non_RA)",
    "FGA_Per100Possessions_Mid_Range", "FG_PCT_Mid_Range", "FGM_Per100Possessions_Mid_Range",
    "FGA_Per100Possessions_Left_Corner_3", "FG_PCT_Left_Corner_3",
    "FGA_Per100Possessions_Right_Corner_3", "FG_PCT_Right_Corner_3",
    "FGA_Per100Possessions_Above_the_Break_3", "FG_PCT_Above_the_Break_3",
    # Playtypes / Tracking (Per Game & PCT)
    "PULL_UP_FGA_PerGame", "PULL_UP_FG_PCT",
    "CATCH_SHOOT_FGA_PerGame", "CATCH_SHOOT_FG_PCT",
    "DRIVES_PerGame", "DRIVE_FG_PCT", "DRIVE_PASSES_PCT",
    "PAINT_TOUCH_FGA_PerGame", "POST_TOUCH_FGA_PerGame", "ELBOW_TOUCH_FGA_PerGame",
    "TOUCHES_PerGame", "ELBOW_TOUCHES_PerGame", "POST_TOUCHES_PerGame", "PAINT_TOUCHES_PerGame", "TIME_OF_POSS_PerGame",
    # Defense Tracking
    "DEF_RIM_FGA_PerGame", "DEF_RIM_FG_PCT",
    # Hustle / Physical
    "DEFLECTIONS_PerGame", "CHARGES_DRAWN_PerGame", "SCREEN_ASSISTS_PerGame",
    "LOOSE_BALLS_RECOVERED_PerGame", "BOX_OUTS_PerGame", "CONTESTED_SHOTS_PerGame",
    "DIST_MILES_OFF_PerGame", "DIST_MILES_DEF_PerGame",
    "AVG_SPEED", "AVG_SPEED_OFF", "AVG_SPEED_DEF",
    "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "AGE",
    # Assisted/Unassisted (Optional but good for profile)
    "PCT_UAST_2PM_Restricted_Area", "PCT_UAST_3PM_Above_the_Break_3", "PCT_UAST_FGM",
    # Granular Shot Types (Style)
    "FGA_Per100Possessions_Alley_Oop", "FG_PCT_Alley_Oop",
    "FGA_Per100Possessions_Bank_Shot", "FG_PCT_Bank_Shot",
    "FGA_Per100Possessions_Dunk", "FG_PCT_Dunk",
    "FGA_Per100Possessions_Fadeaway", "FG_PCT_Fadeaway",
    "FGA_Per100Possessions_Finger_Roll", "FG_PCT_Finger_Roll",
    "FGA_Per100Possessions_Hook_Shot", "FG_PCT_Hook_Shot",
    "FGA_Per100Possessions_Jump_Shot", "FG_PCT_Jump_Shot",
    "FGA_Per100Possessions_Layup", "FG_PCT_Layup",
    "FGA_Per100Possessions_Tip_Shot", "FG_PCT_Tip_Shot"
]

# ----------------------------
# Module-Level Helper Functions (Reusable Expressions)
# ----------------------------

def calculate_zscore(col_name: str, group_col: str = "SEASON_YEAR") -> pl.Expr:
    """Calculates Z-Score for a column grouped by season."""
    return (
        (pl.col(col_name) - pl.col(col_name).mean().over(group_col)) / 
        (pl.col(col_name).std().over(group_col) + 1e-6)
    )

def convert_per_game_to_per_100(col_name: str, poss_col: str = "poss_PerGame") -> pl.Expr:
    """Converts a PerGame stat to Per100Possessions."""
    return (
        pl.when(pl.col(poss_col) > 0)
        .then(pl.col(col_name) * 100.0 / pl.col(poss_col))
        .otherwise(None)
    )

def get_league_ratio_sum_expr(numerator_col: str, denominator_col: str) -> pl.Expr:
    """Returns an expression for league-wide efficiency (Sum(A)/Sum(B)) per season."""
    return (
        pl.col(numerator_col).sum().over("SEASON_YEAR") / 
        (pl.col(denominator_col).sum().over("SEASON_YEAR") + 1e-6)
    )

def load_data(file_path: Path) -> pl.DataFrame:
    """Loads the master dataset and validates required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find input file at {file_path}")
    logging.info(f"Loading data from: {file_path}")
    df = pl.read_parquet(file_path)
    # Column existence check (Non-blocking warning for optional profile fields, blocking for core)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        # We strict fail if any column is missing to ensure data integrity
        raise ValueError(f"Missing required columns in master dataset: {missing}")
    # Ensure all statistical columns are Float64
    skip_cols = {"PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "PLAYER_POSITION"}
    cols_to_cast = [c for c in df.columns if c in REQUIRED_COLUMNS and c not in skip_cols]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_cast
    ])
    return df

# ----------------------------
# Main Pipeline Function
# ----------------------------

def feature_engineering_pipeline(df: pl.DataFrame) -> Tuple[Path, Path]:
    """
    Computes all engineered features sequentially in a single function.
    
    1. Unit Handling
    2. Offensive Archetypes (Proficiency, Load, Box Creation)
    3. Season Context (Heliocentricity, Gravity)
    4. Playmaking & Efficiency
    5. Shot Diet & Tendencies
    6. Defense & Rebounding
    7. Hustle & Movement
    8. Relative Efficiency
    9. Final Selection & Saving
    """
    logging.info(f"Starting pipeline with shape: {df.shape}")

    # -------------------------------------------------------------------------
    # 0. Data Cleaning & Type Casting
    # -------------------------------------------------------------------------
    # Ensure all statistical columns are Float64 to prevent "division by String" errors.
    skip_cols = {"PLAYER_NAME", "SEASON_YEAR", "TEAM_ABBREVIATION", "PLAYER_POSITION"}
    cols_to_cast = [c for c in df.columns if c in REQUIRED_COLUMNS and c not in skip_cols]
    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_cast
    ])

    # =========================================================================
    # 1. Unit Handling & Estimations
    # =========================================================================
    # Estimate Possessions Per Game: PACE * MIN / 48
    df = df.with_columns([
        (pl.col("PACE") * pl.col("MIN1_PerGame") / 48.0).alias("poss_PerGame")
    ])

    # Calculate TOV_PerGame (missing in raw, needed for profile)
    df = df.with_columns([
        (pl.col("TOV_Per100Possessions") * pl.col("poss_PerGame") / 100.0).alias("TOV_PerGame")
    ])

    # =========================================================================
    # 2. Advanced Offensive Archetyping (Ben Taylor / Backpicks inspired)
    # =========================================================================
    logging.info("Computing advanced offensive metrics (Proficiency, Load, Box Creation)...")
    
    # 3PT Proficiency: Logistic curve weighting of volume * accuracy
    # Formula: (2/(1 + exp(-FG3A_p100)) - 1) * FG3_PCT
    three_pt_prof_expr = ((2.0 / (1.0 + (-pl.col("FG3A_Per100Possessions")).exp()) - 1.0) * pl.col("FG3_PCT")).alias("three_pt_proficiency")
    df = df.with_columns(three_pt_prof_expr)

    # Box Creation (approximate open shots created for teammates + self)
    # Formula: AST*0.1843 + (PTS+TOV)*0.0969 - 2.3021*3pt_prof + 0.0582*(AST*(PTS+TOV)*3pt_prof) - 1.1942
    box_creation_expr = (
        (pl.col("AST_Per100Possessions") * 0.1843) +
        ((pl.col("PTS_Per100Possessions") + pl.col("TOV_Per100Possessions")) * 0.0969) -
        (pl.col("three_pt_proficiency") * 2.3021) +
        ((pl.col("AST_Per100Possessions") * (pl.col("PTS_Per100Possessions") + pl.col("TOV_Per100Possessions")) * pl.col("three_pt_proficiency")) * 0.0582) - 
        1.1942
    ).alias("box_creation")
    df = df.with_columns(box_creation_expr)

    # Offensive Load (percentage of possessions a player is directly involved in)
    # Formula: (AST - (0.38 * box_creation))**0.75 + FGA + (FTA*0.44) + box_creation + TOV
    off_load_expr = (
        (pl.col("AST_Per100Possessions") - (0.38 * pl.col("box_creation"))).pow(0.75) +
        pl.col("FGA_Per100Possessions") +
        (pl.col("FTA_Per100Possessions") * 0.44) +
        pl.col("box_creation") +
        pl.col("TOV_Per100Possessions")
    ).alias("offensive_load")
    df = df.with_columns(off_load_expr)

    # =========================================================================
    # 3. Season-Level Context & Normalization
    # =========================================================================
    logging.info("Computing season averages and Z-scores...")
    
    # We calculate Z-scores temporarily for aggregation
    time_poss_rate = (pl.col("TIME_OF_POSS_PerGame") / pl.col("MIN1_PerGame")) # (total minutes player possesses ball) / (player minutes per game)
    df = df.with_columns([
        calculate_zscore("USG_PCT").alias("z_usg"),
        calculate_zscore("AST_PCT").alias("z_ast_pct"),
        ( (time_poss_rate - time_poss_rate.mean().over("SEASON_YEAR")) / 
          (time_poss_rate.std().over("SEASON_YEAR") + 1e-6) ).alias("z_time_poss")
    ])

    # Heliocentricity: Mean of Z(USG), Z(AST%), Z(TimeOfPoss/Min)
    df = df.with_columns([
        ((pl.col("z_usg") + pl.col("z_ast_pct") + pl.col("z_time_poss")) / 3.0).alias("heliocentricity")
    ])

    # TS% Relative
    df = df.with_columns([
        (pl.col("TS_PCT") - pl.col("TS_PCT").mean().over("SEASON_YEAR")).alias("ts_pct_rel")
    ])

    # Scoring Gravity: Volume * Efficiency relative
    df = df.with_columns([
        (pl.col("PTS_Per100Possessions") * pl.col("ts_pct_rel")).alias("scoring_gravity")
    ])

    # =========================================================================
    # 4. Passing & Playmaking Efficiency
    # =========================================================================
    logging.info("Computing passing and playmaking metrics...")

    # Passing Efficiency: AST (pg) / Potential AST (pg)
    # Convert AST p100 back to PG to match POTENTIAL_AST units
    ast_pg = (pl.col("AST_Per100Possessions") * (pl.col("poss_PerGame") / 100.0))
    df = df.with_columns([
        (ast_pg / (pl.col("POTENTIAL_AST_PerGame") + 1e-6)).alias("passing_efficiency"),
        (pl.col("AST_Per100Possessions") / (pl.col("offensive_load") + 1e-6)).alias("assist_to_load"),
        (pl.col("TOV_Per100Possessions") / (pl.col("offensive_load") + 1e-6)).alias("tov_economy")
    ])

    # =========================================================================
    # 5. Shot Diet & Tendencies
    # =========================================================================
    logging.info("Computing shot diet frequencies...")
    
    # Denominator: FGA Per 100
    fga_total = pl.col("FGA_Per100Possessions") + 1e-6

    # Calculate total FGA from zones (since zone columns appear to be totals)
    fga_zones_sum = (
        pl.col("FGA_Per100Possessions_Restricted_Area") +
        pl.col("FGA_Per100Possessions_In_The_Paint_(Non_RA)") +
        pl.col("FGA_Per100Possessions_Mid_Range") +
        pl.col("FGA_Per100Possessions_Left_Corner_3") +
        pl.col("FGA_Per100Possessions_Right_Corner_3") +
        pl.col("FGA_Per100Possessions_Above_the_Break_3")
    ) + 1e-6

    # Granular Style Metrics (Tendency & Efficiency)
    # Isolating Style from Volume by using Frequencies (FGA_Type / Total_FGA)
    df = df.with_columns([
        (pl.col("FGA_Per100Possessions_Alley_Oop") / fga_total).alias("freq_alley_oop"),
        pl.col("FG_PCT_Alley_Oop").alias("eff_alley_oop"),
        (pl.col("FGA_Per100Possessions_Bank_Shot") / fga_total).alias("freq_bank_shot"),
        pl.col("FG_PCT_Bank_Shot").alias("eff_bank_shot"),
        (pl.col("FGA_Per100Possessions_Dunk") / fga_total).alias("freq_dunk"),
        pl.col("FG_PCT_Dunk").alias("eff_dunk"),
        (pl.col("FGA_Per100Possessions_Fadeaway") / fga_total).alias("freq_fadeaway"),
        pl.col("FG_PCT_Fadeaway").alias("eff_fadeaway"),
        (pl.col("FGA_Per100Possessions_Finger_Roll") / fga_total).alias("freq_finger_roll"),
        pl.col("FG_PCT_Finger_Roll").alias("eff_finger_roll"),
        (pl.col("FGA_Per100Possessions_Hook_Shot") / fga_total).alias("freq_hook_shot"),
        pl.col("FG_PCT_Hook_Shot").alias("eff_hook_shot"),
        (pl.col("FGA_Per100Possessions_Jump_Shot") / fga_total).alias("freq_jump_shot"),
        pl.col("FG_PCT_Jump_Shot").alias("eff_jump_shot"),
        (pl.col("FGA_Per100Possessions_Layup") / fga_total).alias("freq_layup"),
        pl.col("FG_PCT_Layup").alias("eff_layup"),
        (pl.col("FGA_Per100Possessions_Tip_Shot") / fga_total).alias("freq_tip_shot"),
        pl.col("FG_PCT_Tip_Shot").alias("eff_tip_shot"),
    ])

    # Zone Frequencies (Already Per100 inputs)
    # FIX: These columns appear to be totals, so we normalize by the sum of zone attempts.
    df = df.with_columns([
        (pl.col("FGA_Per100Possessions_Restricted_Area") / fga_zones_sum).alias("fga_rim_freq"),
        (pl.col("FGA_Per100Possessions_In_The_Paint_(Non_RA)") / fga_zones_sum).alias("fga_floater_freq"),
        (pl.col("FGA_Per100Possessions_Mid_Range") / fga_zones_sum).alias("fga_mid_freq"),
        ((pl.col("FGA_Per100Possessions_Left_Corner_3") + pl.col("FGA_Per100Possessions_Right_Corner_3")) / fga_zones_sum).alias("fga_corner_freq"),
        (pl.col("FGA_Per100Possessions_Above_the_Break_3") / fga_zones_sum).alias("fga_ab3_freq"),
    ])

    # Playtype Frequencies (PerGame -> Per100 -> Ratio); NOTE: the sum of these DO NOT add up to fga_total (e.g., Aaron Gordon 2019-2020 averages 12.37 fga/game, but the sum of these fga's/game is 13.8)
    df = df.with_columns([
        (convert_per_game_to_per_100("PULL_UP_FGA_PerGame") / fga_total).alias("pullup_fga_freq"),
        (convert_per_game_to_per_100("DRIVE_FGA_PerGame") / fga_total).alias("drive_fga_freq"),
        (convert_per_game_to_per_100("CATCH_SHOOT_FGA_PerGame") / fga_total).alias("catch_shoot_fga_freq"),
        (convert_per_game_to_per_100("PAINT_TOUCH_FGA_PerGame") / fga_total).alias("paint_touch_fga_freq"),
        (convert_per_game_to_per_100("POST_TOUCH_FGA_PerGame") / fga_total).alias("post_touch_fga_freq"),
        (convert_per_game_to_per_100("ELBOW_TOUCH_FGA_PerGame") / fga_total).alias("elbow_touch_fga_freq"),
        
        # Drive Rate (Drives per 100 poss)
        convert_per_game_to_per_100("DRIVES_PerGame").alias("drive_rate"),
        
        # FT Rate
        (pl.col("FTA_Per100Possessions") / fga_total).alias("ft_rate")
    ])

    # Touch Frequencies (Ratios of PerGame counts)
    total_touches = pl.col("TOUCHES_PerGame") + 1e-6
    df = df.with_columns([
        (pl.col("ELBOW_TOUCHES_PerGame") / total_touches).alias("elbow_touch_freq"),
        (pl.col("POST_TOUCHES_PerGame") / total_touches).alias("post_touch_freq"),
        (pl.col("PAINT_TOUCHES_PerGame") / total_touches).alias("paint_touch_freq"),
    ])

    # =========================================================================
    # 6. Defense & Rebounding
    # =========================================================================
    logging.info("Computing defense and rebounding metrics...")

    # Stocks (Steals + Blocks per 100)
    df = df.with_columns([
        (pl.col("STL_Per100Possessions") + pl.col("BLK_Per100Possessions")).alias("stocks")
    ])

    # Defensive Versatility: 1 / (1 + abs(Z(BLK) - Z(STL)))
    df = df.with_columns([
        calculate_zscore("BLK_Per100Possessions").alias("z_blk"),
        calculate_zscore("STL_Per100Possessions").alias("z_stl"),
    ])
    df = df.with_columns([
        (1.0 / (1.0 + (pl.col("z_blk") - pl.col("z_stl")).abs())).alias("def_versatility")
    ])

    # Rim Deterrence: Player Rim FG% - League Rim FG%
    # League stat calculation: Sum(FGM_Rim) / Sum(FGA_Rim)
    league_rim_pct = get_league_ratio_sum_expr(
        "FGM_Per100Possessions_Restricted_Area", 
        "FGA_Per100Possessions_Restricted_Area"
    )

    df = df.with_columns([
        (pl.col("DEF_RIM_FG_PCT") - league_rim_pct).alias("rim_deterrence"),
        # Rim Contest Rate: Contests / Minute
        (pl.col("DEF_RIM_FGA_PerGame") / (pl.col("MIN1_PerGame") + 1e-6)).alias("rim_contests_per_min")
    ])

    # =========================================================================
    # 7. Hustle & Movement
    # =========================================================================
    logging.info("Computing hustle and movement metrics...")

    # Hustle stats: PerGame -> Per100
    hustle_cols = [
        "CONTESTED_SHOTS_PerGame", "DEFLECTIONS_PerGame", "CHARGES_DRAWN_PerGame",
        "SCREEN_ASSISTS_PerGame", "LOOSE_BALLS_RECOVERED_PerGame", "BOX_OUTS_PerGame"
    ]
    hustle_exprs = [convert_per_game_to_per_100(col).alias(col.replace("_PerGame", "_Per100Possessions")) for col in hustle_cols]
    df = df.with_columns(hustle_exprs)

    # Movement: Miles per minute
    df = df.with_columns([
        (pl.col("DIST_MILES_DEF_PerGame") / (pl.col("MIN1_PerGame") + 1e-6)).alias("def_miles_per_min"),
        (pl.col("DIST_MILES_OFF_PerGame") / (pl.col("MIN1_PerGame") + 1e-6)).alias("off_miles_per_min"),
        
        # Physical Z-Scores
        calculate_zscore("PLAYER_HEIGHT_INCHES").alias("z_height"),
        calculate_zscore("PLAYER_WEIGHT").alias("z_weight"),
        calculate_zscore("AGE").alias("z_age")
    ])

    # =========================================================================
    # 8. Relative Efficiency by Zone
    # =========================================================================
    logging.info("Computing relative zone efficiency metrics...")

    # Relative Efficiency: Player FG% - League FG% (Ratio of Sums)
    # NOTE: columns like FGM_Per100Possessions_Restricted_Area aren't represented as Per100Possessions, but rather seem like totals. However, since we're computing a ratio, this shouldn't matter. But, make sure !!!!!!!!!!!!!!!!
    lg_rim_pct = get_league_ratio_sum_expr("FGM_Per100Possessions_Restricted_Area", "FGA_Per100Possessions_Restricted_Area")
    lg_mid_pct = get_league_ratio_sum_expr("FGM_Per100Possessions_Mid_Range", "FGA_Per100Possessions_Mid_Range")
    lg_fg3_pct = get_league_ratio_sum_expr("FG3M_Per100Possessions", "FG3A_Per100Possessions")

    df = df.with_columns([
        (pl.col("FG_PCT_Restricted_Area") - lg_rim_pct).alias("rim_efficiency_rel"),
        (pl.col("FG_PCT_Mid_Range") - lg_mid_pct).alias("mid_efficiency_rel"),
        (pl.col("FG3_PCT") - lg_fg3_pct).alias("three_pt_efficiency_rel")
    ])

    # =========================================================================
    # 9. Output Selection & Save
    # =========================================================================
    # Define features for similarity
    sim_features = [
        # Identifiers
        "PLAYER_ID", "SEASON_YEAR",
        
        # Offensive Archetype
        "three_pt_proficiency", "box_creation", "offensive_load", 
        "heliocentricity", "passing_efficiency", "assist_to_load",
        "ts_pct_rel", "scoring_gravity", "tov_economy", "ft_rate", "PCT_UAST_FGM",
        
        # Shot Diet
        "fga_rim_freq", "fga_floater_freq", "fga_mid_freq", "fga_corner_freq", "fga_ab3_freq",
        "drive_rate",
        
        # Granular Style (New)
        "freq_alley_oop", "eff_alley_oop",
        "freq_bank_shot", "eff_bank_shot",
        "freq_dunk", "eff_dunk",
        "freq_fadeaway", "eff_fadeaway",
        "freq_finger_roll", "eff_finger_roll",
        "freq_hook_shot", "eff_hook_shot",
        "freq_jump_shot", "eff_jump_shot",
        "freq_layup", "eff_layup",
        "freq_tip_shot", "eff_tip_shot",
        
        # Playtype Freq
        "pullup_fga_freq", "drive_fga_freq", "catch_shoot_fga_freq", 
        "paint_touch_fga_freq", "post_touch_fga_freq", "elbow_touch_fga_freq",
        
        # Touch Distribution
        "elbow_touch_freq", "post_touch_freq", "paint_touch_freq",
        
        # Defense/Reb
        "OREB_PCT", "DREB_PCT", "PCT_BLK", "PCT_STL", 
        "stocks", "def_versatility", "rim_deterrence", "rim_contests_per_min", "PCT_PLUSMINUS",
        
        # Hustle
        "CONTESTED_SHOTS_Per100Possessions", "DEFLECTIONS_Per100Possessions", "CHARGES_DRAWN_Per100Possessions", 
        "SCREEN_ASSISTS_Per100Possessions", "LOOSE_BALLS_RECOVERED_Per100Possessions", "BOX_OUTS_Per100Possessions",
        "PF_Per100Possessions",
        
        # Physical/Movement
        "def_miles_per_min", "off_miles_per_min", 
        "AVG_SPEED", "AVG_SPEED_OFF", "AVG_SPEED_DEF",
        "z_height", "z_weight", "z_age",
        
        # Relative Efficiency
        "rim_efficiency_rel", "mid_efficiency_rel", "three_pt_efficiency_rel"
    ]

    # Select similarity features & apply Z-Score Normalization to ALL similarity features (excluding ID cols) for search optimization
    id_cols = ["PLAYER_ID", "SEASON_YEAR"]
    numeric_cols = [c for c in sim_features if c not in id_cols]
    df_similarity = df.select(sim_features).with_columns([
        calculate_zscore(col).alias(col) 
        for col in numeric_cols
    ])
    
    # Fill NaNs resulting from Z-score (e.g. std=0) with 0
    # df_similarity = df_similarity.fill_nan(0.0).fill_null(0.0)

    # ---------------------------------------------------------
    # Profile Output (Rich stats for UI)
    # ---------------------------------------------------------
    profile_cols = [
        # Identity
        "PLAYER_ID", "PLAYER_NAME", "SEASON_YEAR", "TEAM_ABBREVIATION", "PLAYER_POSITION", "AGE", "GP", "MIN1_PerGame",
        
        # Base Stats (Per Game)
        "PTS_PerGame", "AST_PerGame", "REB_PerGame", "STL_PerGame", "BLK_PerGame", "TOV_PerGame",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        
        # Advanced
        "TS_PCT", "USG_PCT", "AST_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING", "PIE", "PACE",
        
        # Archetype / Computed
        "offensive_load", "box_creation", "passing_efficiency", "scoring_gravity", "def_versatility", "three_pt_proficiency",
        "drive_rate",
        
        # Shooting Zones (Volumes & Efficiency)
        "FGA_Per100Possessions_Restricted_Area", "FG_PCT_Restricted_Area",
        "FGA_Per100Possessions_Mid_Range", "FG_PCT_Mid_Range",
        "FGA_Per100Possessions_Above_the_Break_3", "FG_PCT_Above_the_Break_3",
        "FGA_Per100Possessions_Left_Corner_3", "FG_PCT_Left_Corner_3",
        "FGA_Per100Possessions_Right_Corner_3", "FG_PCT_Right_Corner_3",
        
        # Playtypes
        "PULL_UP_FGA_PerGame", "PULL_UP_FG_PCT",
        "CATCH_SHOOT_FGA_PerGame", "CATCH_SHOOT_FG_PCT",
        "DRIVES_PerGame", "DRIVE_FG_PCT", "DRIVE_PASSES_PCT",
        
        # Physical & Hustle
        "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT",
        "AVG_SPEED_OFF", "AVG_SPEED_DEF", "DIST_MILES_PerGame",
        "CONTESTED_SHOTS_PerGame", "DEFLECTIONS_PerGame",
        
        # Shot Creation Context
        "PCT_UAST_2PM_Restricted_Area", "PCT_UAST_3PM_Above_the_Break_3"
    ]
    
    # Filter columns to ensure they exist (e.g. if some optional tracking stats are missing)
    final_profile_cols = [c for c in profile_cols if c in df.columns]
    df_profile = df.select(final_profile_cols)

    # Validation
    if df_similarity.select(["PLAYER_ID", "SEASON_YEAR"]).is_duplicated().any():
        raise ValueError("Duplicate PLAYER_ID + SEASON_YEAR detected in output!")
    null_counts = df_similarity.null_count().transpose(include_header=True)
    null_counts.columns = ["feature", "null_count"]
    logging.info("Null counts per feature (Top 5):")
    logging.info(null_counts.filter(pl.col("feature") != "PLAYER_ID").sort("null_count", descending=True).head(5))
    logging.info(f"Final Similarity Shape: {df_similarity.shape}")
    logging.info(f"Final Profile Shape: {df_profile.shape}")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_similarity.write_parquet(OUTPUT_SIMILARITY)
    logging.info(f"Saved similarity features to {OUTPUT_SIMILARITY}")
    df_profile.write_parquet(OUTPUT_PROFILE)
    logging.info(f"Saved profile features to {OUTPUT_PROFILE}")
    return OUTPUT_SIMILARITY, OUTPUT_PROFILE


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        df_master = load_data(INPUT_FILE)
        feature_engineering_pipeline(df_master)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)