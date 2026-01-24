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

import polars as pl
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
    description="Fetches raw NBA player data for all configured seasons, handles rate limiting, "
                "aggregates traded players, and saves the master analytics parquet file.",
    group_name="ingestion",
)
def raw_nba_data(context: AssetContext) -> Path:
    """
    Orchestrates the `processing.py` pipeline.
    Returns the path to the saved master player analytics file.
    """
    context.log.info("Starting NBA data ingestion pipeline...")
    
    # Execute the processing pipeline
    output_path = processing.run_pipeline()
    
    if output_path is None:
        raise RuntimeError("Pipeline finished but returned no output path. Check logs for errors.")
    
    context.log.info(f"Ingestion complete. Master data saved to: {output_path}")
    return output_path


@asset(
    description="Consumes the master player data to generate normalized similarity features "
                "and rich profile features.",
    group_name="engineering",
)
def engineered_features(context: AssetContext, raw_nba_data: Path) -> Tuple[Path, Path]:
    """
    Orchestrates the `engineering.py` pipeline.
    Depends on `raw_nba_data` to ensure ingestion runs first.
    """
    context.log.info(f"Loading master dataset from: {raw_nba_data}")
    
    # Load the dataframe to pass into the engineering pipeline
    # This ensures we are using the file exactly as produced by the upstream asset
    df = pl.read_parquet(raw_nba_data)
    context.log.info(f"Loaded DataFrame with shape: {df.shape}")
    
    context.log.info("Running feature engineering pipeline...")
    sim_path, profile_path = engineering.feature_engineering_pipeline(df)
    
    context.log.info("Feature engineering complete.")
    context.log.info(f"Similarity Features: {sim_path}")
    context.log.info(f"Profile Features:    {profile_path}")
    
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
