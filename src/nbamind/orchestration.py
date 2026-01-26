"""
==========================================================================================
NBAMind: orchestration.py
------------------------------------------------------------------------------------------
Defines the Dagster assets, jobs, and schedules for the NBA data pipeline.
This orchestrates the flow from raw API ingestion -> master table -> feature engineering.
==========================================================================================
"""

from pathlib import Path
from typing import Tuple

from dagster import (
    AssetContext,
    Definitions,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

from nbamind.data import processing
from nbamind.features import engineering

# -------------------------------------------------------------------------
# Assets
# -------------------------------------------------------------------------

@asset(
    description="Fetches raw NBA player data, aggregates stats, and saves the master analytics parquet file.",
    group_name="ingestion",
)
def raw_nba_data(context: AssetContext) -> Path:
    """
    Orchestrates the data ingestion pipeline.
    Returns the path to the saved master player analytics file.
    """
    context.log.info("Starting NBA data ingestion pipeline...")
    
    output_path = processing.run_ingestion_pipeline()
    
    if not output_path:
        raise RuntimeError("Ingestion pipeline failed to produce an output path.")
    
    context.log.info(f"Ingestion complete. Master data saved to: {output_path}")
    return output_path


@asset(
    description="Generates normalized similarity features and rich profile features from master data.",
    group_name="engineering",
)
def engineered_features(context: AssetContext, raw_nba_data: Path) -> Tuple[Path, Path]:
    """
    Orchestrates the feature engineering pipeline.
    """
    context.log.info(f"Loading master dataset from: {raw_nba_data}")
    
    # Load data using the engineering module's loader to ensure schema consistency
    df = engineering.load_data(raw_nba_data)
    context.log.info(f"Loaded DataFrame with shape: {df.shape}")
    
    context.log.info("Running feature engineering pipeline...")
    sim_path, profile_path = engineering.run_feature_engineering_pipeline(df)
    
    context.log.info(f"Feature engineering complete.\nSimilarity: {sim_path}\nProfile: {profile_path}")
    
    return sim_path, profile_path


# -------------------------------------------------------------------------
# Jobs & Schedules
# -------------------------------------------------------------------------

# Define a job that materializes the entire pipeline
nba_pipeline_job = define_asset_job(
    name="nba_pipeline_job",
    selection=[raw_nba_data, engineered_features],
)

# Schedule: Weekly on Mondays at 08:00 UTC
# Active only during the NBA Season (October - April)
# Cron Expression Breakdown:
#   0 8       -> At 08:00
#   *         -> Every day of month
#   10-12,1-4 -> In Oct, Nov, Dec, Jan, Feb, Mar, Apr
#   1         -> On Mondays
season_schedule = ScheduleDefinition(
    job=nba_pipeline_job,
    cron_schedule="0 8 * 10-12,1-4 1",
    execution_timezone="UTC",
    name="weekly_in_season_update",
)

# -------------------------------------------------------------------------
# Definitions
# -------------------------------------------------------------------------

defs = Definitions(
    assets=[raw_nba_data, engineered_features],
    schedules=[season_schedule],
)
