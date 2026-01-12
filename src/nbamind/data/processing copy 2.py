"""
==========================================================================================
NBAMind: processing.py
------------------------------------------------------------------------------------------
This script constitutes the primary data ingestion pipeline for the Player Similarity Engine.
It systematically fetches NBA player statistics for all seasons from 2000-01 to the present,
focusing on data points that reveal player playstyle, archetypes, and on-court tendencies.

The pipeline operates in two main stages to maximize efficiency:
1.  **League-Wide Data Fetch:** For each season, it makes single API calls to 'LeagueDash'
    endpoints. These endpoints provide a wide array of statistics for all players in the
    league, including traditional, advanced, shooting location, defensive, and hustle stats.
    This data is saved to a structured directory per season.

2.  **Targeted Player-Specific Fetch:** After gathering the league-wide data, the script
    identifies players who meet a minimum threshold for games played and total minutes.
    This filters out players with insignificant sample sizes. For this qualified subset
    of players, it then makes more granular, individual 'PlayerDash' API calls to obtain
    critical playstyle context not available in the league-wide data, such as shot-creation
    habits (pull-ups vs. catch-and-shoot), detailed rebounding stats (contested vs.
    uncontested), and shot-assist context (assisted vs. unassisted).

All data fetching is routed through the robust `fetcher.py` wrapper, which handles
caching, rate-limiting, and retries automatically. The final output is stored in the
`data/processed` directory, organized by season and data type in the efficient Parquet format.

This modular and idempotent approach ensures that the data foundation for the similarity
model is both comprehensive and efficiently acquired.
==========================================================================================
"""
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from nba_api.stats.endpoints import (
    commonallplayers,
    leaguedashplayerbiostats,
    leaguehustlestatsplayer,
    leaguedashplayershotlocations,
    leaguedashplayerstats,
    leaguedashptdefend,
    leaguedashptstats,
    playerdashboardbyshootingsplits,
    playerdashptreb,
    playerdashptshots,
)

from nbamind.data.fetcher import fetch

# ----------------------------
# Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
PROCESSED_DIR = Path("data/processed")
# Minimum thresholds to qualify a player for detailed, per-player API calls
MIN_GAMES_PLAYED = 15
MIN_MINUTES_PLAYED = 300 # ~20 min over 15 games

# Endpoint definitions now contain all static parameters.
# The 'Season' and 'PlayerID' parameters are added dynamically during execution.
LEAGUE_WIDE_ENDPOINTS = {
    # "all_players": {
    #     "cls": commonallplayers.CommonAllPlayers, # ['PERSON_ID', 'DISPLAY_LAST_COMMA_FIRST', 'DISPLAY_FIRST_LAST', 'ROSTERSTATUS', 'FROM_YEAR', 'TO_YEAR', 'PLAYERCODE', 'PLAYER_SLUG', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CODE', 'TEAM_SLUG', 'GAMES_PLAYED_FLAG', 'OTHERLEAGUE_EXPERIENCE_CH']
    #     "params": {
    #         "league_id": "00",
    #     },
    #     "result_set": "CommonAllPlayers",
    # },
    "bio_stats": {
        "cls": leaguedashplayerbiostats.LeagueDashPlayerBioStats, # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'PLAYER_HEIGHT', 'PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT', 'COLLEGE', 'COUNTRY', 'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'GP', 'PTS', 'REB', 'AST', 'NET_RATING', 'OREB_PCT', 'DREB_PCT', 'USG_PCT', 'TS_PCT', 'AST_PCT']
        "params": {
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id": "00",
        },
        "result_set": "LeagueDashPlayerBioStats",
    },
    "adv_stats": {
        "cls": leaguedashplayerstats.LeagueDashPlayerStats, # ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'E_OFF_RATING', 'OFF_RATING', 'sp_work_OFF_RATING', 'E_DEF_RATING', 'DEF_RATING', 'sp_work_DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'sp_work_NET_RATING', 'AST_PCT', 'AST_TO', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'E_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'sp_work_PACE', 'PIE', 'POSS', 'FGM', 'FGA', 'FGM_PG', 'FGA_PG', 'FG_PCT', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'E_OFF_RATING_RANK', 'OFF_RATING_RANK', 'sp_work_OFF_RATING_RANK', 'E_DEF_RATING_RANK', 'DEF_RATING_RANK', 'sp_work_DEF_RATING_RANK', 'E_NET_RATING_RANK', 'NET_RATING_RANK', 'sp_work_NET_RATING_RANK', 'AST_PCT_RANK', 'AST_TO_RANK', 'AST_RATIO_RANK', 'OREB_PCT_RANK', 'DREB_PCT_RANK', 'REB_PCT_RANK', 'TM_TOV_PCT_RANK', 'E_TOV_PCT_RANK', 'EFG_PCT_RANK', 'TS_PCT_RANK', 'USG_PCT_RANK', 'E_USG_PCT_RANK', 'E_PACE_RANK', 'PACE_RANK', 'sp_work_PACE_RANK', 'PIE_RANK', 'FGM_RANK', 'FGA_RANK', 'FGM_PG_RANK', 'FGA_PG_RANK', 'FG_PCT_RANK', 'TEAM_COUNT']
        "params": {
            "measure_type_detailed_defense": "Advanced",
            "per_mode_detailed": "Per100Possessions",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPlayerStats",
    },
    "base_stats": {
        "cls": leaguedashplayerstats.LeagueDashPlayerStats, # ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'WNBA_FANTASY_PTS', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK', 'WNBA_FANTASY_PTS_RANK', 'TEAM_COUNT']
        "params": {
            "measure_type_detailed_defense": "Base",
            "per_mode_detailed": "Per100Possessions",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPlayerStats",
    },
    # measure_type_detailed_defense = Misc: ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'BLK', 'BLKA', 'PF', 'PFD', 'NBA_FANTASY_PTS', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'PTS_OFF_TOV_RANK', 'PTS_2ND_CHANCE_RANK', 'PTS_FB_RANK', 'PTS_PAINT_RANK', 'OPP_PTS_OFF_TOV_RANK', 'OPP_PTS_2ND_CHANCE_RANK', 'OPP_PTS_FB_RANK', 'OPP_PTS_PAINT_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'NBA_FANTASY_PTS_RANK', 'TEAM_COUNT']
    # measure_type_detailed_defense = Scoring: ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'FGM', 'FGA', 'FG_PCT', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'PCT_FGA_2PT_RANK', 'PCT_FGA_3PT_RANK', 'PCT_PTS_2PT_RANK', 'PCT_PTS_2PT_MR_RANK', 'PCT_PTS_3PT_RANK', 'PCT_PTS_FB_RANK', 'PCT_PTS_FT_RANK', 'PCT_PTS_OFF_TOV_RANK', 'PCT_PTS_PAINT_RANK', 'PCT_AST_2PM_RANK', 'PCT_UAST_2PM_RANK', 'PCT_AST_3PM_RANK', 'PCT_UAST_3PM_RANK', 'PCT_AST_FGM_RANK', 'PCT_UAST_FGM_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'TEAM_COUNT'] 
    # measure_type_detailed_defense = Usage: ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'USG_PCT', 'PCT_FGM', 'PCT_FGA', 'PCT_FG3M', 'PCT_FG3A', 'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB', 'PCT_REB', 'PCT_AST', 'PCT_TOV', 'PCT_STL', 'PCT_BLK', 'PCT_BLKA', 'PCT_PF', 'PCT_PFD', 'PCT_PTS', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'USG_PCT_RANK', 'PCT_FGM_RANK', 'PCT_FGA_RANK', 'PCT_FG3M_RANK', 'PCT_FG3A_RANK', 'PCT_FTM_RANK', 'PCT_FTA_RANK', 'PCT_OREB_RANK', 'PCT_DREB_RANK', 'PCT_REB_RANK', 'PCT_AST_RANK', 'PCT_TOV_RANK', 'PCT_STL_RANK', 'PCT_BLK_RANK', 'PCT_BLKA_RANK', 'PCT_PF_RANK', 'PCT_PFD_RANK', 'PCT_PTS_RANK', 'TEAM_COUNT']
    # measure_type_detailed_defense = Defense: ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'DEF_RATING', 'DREB', 'DREB_PCT', 'PCT_DREB', 'STL', 'PCT_STL', 'BLK', 'PCT_BLK', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'DEF_WS', 'DEF_WS_RAW', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'DEF_RATING_RANK', 'DREB_RANK', 'DREB_PCT_RANK', 'PCT_DREB_RANK', 'STL_RANK', 'PCT_STL_RANK', 'BLK_RANK', 'PCT_BLK_RANK', 'OPP_PTS_OFF_TOV_RANK', 'OPP_PTS_2ND_CHANCE_RANK', 'OPP_PTS_FB_RANK', 'OPP_PTS_PAINT_RANK', 'DEF_WS_RANK']
    # "shot_locations": {
    #     "cls": leaguedashplayershotlocations.LeagueDashPlayerShotLocations, # format is a bit iffy, but it shows fgm, fga, and fg_pct in the following areas: "Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3", "Above the Break 3", "Backcourt"
    #     "params": {
    #         "measure_type_simple": "Base",
    #         "per_mode_detailed": "Per100Possessions",
    #         "season_type_all_star": "Regular Season",
    #         "league_id_nullable": "00",
    #     },
    #     "result_set": "ShotLocations",
    # },
    "defending": {
        "cls": leaguedashptdefend.LeagueDashPtDefend, # ['CLOSE_DEF_PERSON_ID', 'PLAYER_NAME', 'PLAYER_LAST_TEAM_ID', 'PLAYER_LAST_TEAM_ABBREVIATION', 'PLAYER_POSITION', 'AGE', 'GP', 'G', 'FREQ', 'D_FGM', 'D_FGA', 'D_FG_PCT', 'NORMAL_FG_PCT', 'PCT_PLUSMINUS']
        "params": {
            "defense_category": "Overall", # for more granular defense data, we can replace Overall with 3 Pointers, 2 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft
            "per_mode_simple": "PerGame", # Totals or PerGame
            "season_type_all_star": "Regular Season",
            "league_id": "00",
        },
        "result_set": "LeagueDashPTDefend",
    },
    "hustle": {
        "cls": leaguehustlestatsplayer.LeagueHustleStatsPlayer, # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'G', 'MIN', 'CONTESTED_SHOTS', 'CONTESTED_SHOTS_2PT', 'CONTESTED_SHOTS_3PT', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS', 'SCREEN_AST_PTS', 'OFF_LOOSE_BALLS_RECOVERED', 'DEF_LOOSE_BALLS_RECOVERED', 'LOOSE_BALLS_RECOVERED', 'PCT_LOOSE_BALLS_RECOVERED_OFF', 'PCT_LOOSE_BALLS_RECOVERED_DEF', 'OFF_BOXOUTS', 'DEF_BOXOUTS', 'BOX_OUTS', 'BOX_OUT_PLAYER_TEAM_REBS', 'BOX_OUT_PLAYER_REBS', 'PCT_BOX_OUTS_OFF', 'PCT_BOX_OUTS_DEF', 'PCT_BOX_OUTS_TEAM_REB', 'PCT_BOX_OUTS_REB']
        "params": {
            "per_mode_time": "Per48", # Totals, PerGame, Per48, Per40, Per36, or PerMinute
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "HustleStatsPlayer",
    },
    "speed_distance": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'MIN1', 'DIST_FEET', 'DIST_MILES', 'DIST_MILES_OFF', 'DIST_MILES_DEF', 'AVG_SPEED', 'AVG_SPEED_OFF', 'AVG_SPEED_DEF']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "SpeedDistance", # ^(SpeedDistance)|(Rebounding)|(Possessions)|(CatchShoot)|(PullUpShot)|(Defense)|(Drives)|(Passing)|(ElbowTouch)|(PostTouch)|(PaintTouch)|(Efficiency)$
            "per_mode_simple": "PerGame", # Totals or PerGame
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "rebounding": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'OREB', 'OREB_CONTEST', 'OREB_UNCONTEST', 'OREB_CONTEST_PCT', 'OREB_CHANCES', 'OREB_CHANCE_PCT', 'OREB_CHANCE_DEFER', 'OREB_CHANCE_PCT_ADJ', 'AVG_OREB_DIST', 'DREB', 'DREB_CONTEST', 'DREB_UNCONTEST', 'DREB_CONTEST_PCT', 'DREB_CHANCES', 'DREB_CHANCE_PCT', 'DREB_CHANCE_DEFER', 'DREB_CHANCE_PCT_ADJ', 'AVG_DREB_DIST', 'REB', 'REB_CONTEST', 'REB_UNCONTEST', 'REB_CONTEST_PCT', 'REB_CHANCES', 'REB_CHANCE_PCT', 'REB_CHANCE_DEFER', 'REB_CHANCE_PCT_ADJ', 'AVG_REB_DIST']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Rebounding",
            "per_mode_simple": "PerGame", 
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "possessions": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'POINTS', 'TOUCHES', 'FRONT_CT_TOUCHES', 'TIME_OF_POSS', 'AVG_SEC_PER_TOUCH', 'AVG_DRIB_PER_TOUCH', 'PTS_PER_TOUCH', 'ELBOW_TOUCHES', 'POST_TOUCHES', 'PAINT_TOUCHES', 'PTS_PER_ELBOW_TOUCH', 'PTS_PER_POST_TOUCH', 'PTS_PER_PAINT_TOUCH']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Possessions",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "catch_shoot": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'CATCH_SHOOT_FGM', 'CATCH_SHOOT_FGA', 'CATCH_SHOOT_FG_PCT', 'CATCH_SHOOT_PTS', 'CATCH_SHOOT_FG3M', 'CATCH_SHOOT_FG3A', 'CATCH_SHOOT_FG3_PCT', 'CATCH_SHOOT_EFG_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "CatchShoot",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "pull_up_shot": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'PULL_UP_FGM', 'PULL_UP_FGA', 'PULL_UP_FG_PCT', 'PULL_UP_PTS', 'PULL_UP_FG3M', 'PULL_UP_FG3A', 'PULL_UP_FG3_PCT', 'PULL_UP_EFG_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "PullUpShot",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "defense": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'STL', 'BLK', 'DREB', 'DEF_RIM_FGM', 'DEF_RIM_FGA', 'DEF_RIM_FG_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Defense",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "drives": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'DRIVES', 'DRIVE_FGM', 'DRIVE_FGA', 'DRIVE_FG_PCT', 'DRIVE_FTM', 'DRIVE_FTA', 'DRIVE_FT_PCT', 'DRIVE_PTS', 'DRIVE_PTS_PCT', 'DRIVE_PASSES', 'DRIVE_PASSES_PCT', 'DRIVE_AST', 'DRIVE_AST_PCT', 'DRIVE_TOV', 'DRIVE_TOV_PCT', 'DRIVE_PF', 'DRIVE_PF_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Drives",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "passing": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'PASSES_MADE', 'PASSES_RECEIVED', 'AST', 'FT_AST', 'SECONDARY_AST', 'POTENTIAL_AST', 'AST_POINTS_CREATED', 'AST_ADJ', 'AST_TO_PASS_PCT', 'AST_TO_PASS_PCT_ADJ']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Passing",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "elbow_touch": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'TOUCHES', 'ELBOW_TOUCHES', 'ELBOW_TOUCH_FGM', 'ELBOW_TOUCH_FGA', 'ELBOW_TOUCH_FG_PCT', 'ELBOW_TOUCH_FTM', 'ELBOW_TOUCH_FTA', 'ELBOW_TOUCH_FT_PCT', 'ELBOW_TOUCH_PTS', 'ELBOW_TOUCH_PASSES', 'ELBOW_TOUCH_AST', 'ELBOW_TOUCH_AST_PCT', 'ELBOW_TOUCH_TOV', 'ELBOW_TOUCH_TOV_PCT', 'ELBOW_TOUCH_FOULS', 'ELBOW_TOUCH_PASSES_PCT', 'ELBOW_TOUCH_FOULS_PCT', 'ELBOW_TOUCH_PTS_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "ElbowTouch",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "post_touch": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'TOUCHES', 'POST_TOUCHES', 'POST_TOUCH_FGM', 'POST_TOUCH_FGA', 'POST_TOUCH_FG_PCT', 'POST_TOUCH_FTM', 'POST_TOUCH_FTA', 'POST_TOUCH_FT_PCT', 'POST_TOUCH_PTS', 'POST_TOUCH_PTS_PCT', 'POST_TOUCH_PASSES', 'POST_TOUCH_PASSES_PCT', 'POST_TOUCH_AST', 'POST_TOUCH_AST_PCT', 'POST_TOUCH_TOV', 'POST_TOUCH_TOV_PCT', 'POST_TOUCH_FOULS', 'POST_TOUCH_FOULS_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "PostTouch",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "paint_touch": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'TOUCHES', 'PAINT_TOUCHES', 'PAINT_TOUCH_FGM', 'PAINT_TOUCH_FGA', 'PAINT_TOUCH_FG_PCT', 'PAINT_TOUCH_FTM', 'PAINT_TOUCH_FTA', 'PAINT_TOUCH_FT_PCT', 'PAINT_TOUCH_PTS', 'PAINT_TOUCH_PTS_PCT', 'PAINT_TOUCH_PASSES', 'PAINT_TOUCH_PASSES_PCT', 'PAINT_TOUCH_AST', 'PAINT_TOUCH_AST_PCT', 'PAINT_TOUCH_TOV', 'PAINT_TOUCH_TOV_PCT', 'PAINT_TOUCH_FOULS', 'PAINT_TOUCH_FOULS_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "PaintTouch",
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
    "efficiency": { # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN', 'POINTS', 'DRIVE_PTS', 'DRIVE_FG_PCT', 'CATCH_SHOOT_PTS', 'CATCH_SHOOT_FG_PCT', 'PULL_UP_PTS', 'PULL_UP_FG_PCT', 'PAINT_TOUCH_PTS', 'PAINT_TOUCH_FG_PCT', 'POST_TOUCH_PTS', 'POST_TOUCH_FG_PCT', 'ELBOW_TOUCH_PTS', 'ELBOW_TOUCH_FG_PCT', 'EFF_FG_PCT']
        "cls": leaguedashptstats.LeagueDashPtStats,
        "params": {
            "player_or_team": "Player",
            "pt_measure_type": "Efficiency", # this seems to summarize all the shot types: drive, catch_shoot, pull_up, paint, paint, post, elbow; we can probably just keep this and get rid of all the other related calls above
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPtStats",
    },
}

PLAYER_SPECIFIC_ENDPOINTS = {
    "shot_creation": {
        "cls": playerdashptshots.PlayerDashPtShots,
        "params": {"PerMode": "Per100Possessions", "SeasonType": "Regular Season"},
        "result_sets": [
            "ShotClockShooting", # shows fga, fgm, fg_pct data of 3's and 2's for shot clock ranges
            "DribbleShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken with different number of dribbles
            "GeneralShooting", # shows fga, fgm, fg_pct data of 3's and 2's for certain types (catch and shot, pull ups, less than 10 ft, other)
            "TouchTimeShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken for varying touch times (i.e., how long player held ball for before shooting)
            "ClosestDefenderShooting10ftPlus", # shows fga, fgm, fg_pct data of 3's and 2's taken from all distances with different closest defender distances
            "ClosestDefenderShooting10ftPlus", # shows fga, fgm, fg_pct data of 3's and 2's taken from more than 10 ft with different closest defender distances
        ],
    },
    # "rebounding": {
    #     "cls": playerdashptreb.PlayerDashPtReb,
    #     "params": {"PerMode": "Per100Possessions", "SeasonType": "Regular Season"},
    #     "result_sets": ["OverallRebounding", "ShotTypeRebounding", "ShotDistanceRebounding"],
    # },
    "shot_splits": {
        "cls": playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits,
        "params": {"PerMode": "Per100Possessions", "SeasonType": "Regular Season"},
        "result_sets": [
            # "ShotFiveFtData", # same as ShotEightFtData but just more granular distances
            "ShotEightFtData", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's from varying distances (<8 ft, 8-16 ft, 16-24 ft, 24+ ft, backcourt)
            "ShotAreaPlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's at varying court locations (restricted area, left corner, etc.)
            # "AssistedShotPlayerDashboard",
            "ShotTypeSummaryPlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's for varying shot types (alley oop, bank shot, dunk, fadeaway, etc.)
            "ShotTypePlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's for more specific varying shot types (alley oop dunk shot, cutting dunk shot, driving bank hook shot, etc.)
        ],
    },
}

# ----------------------------
# Helper Functions
# ----------------------------

def get_seasons_list(start_year: int = 2000) -> List[str]:
    """
    Generates NBA 'Season' strings from a start year up to the current season.
    e.g., '2023-24'.
    """
    today = date.today()
    current_start_year = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(start_year, current_start_year + 1)]


def extract_dataframe_from_response(
    response: Dict[str, Any], result_set_name: str
) -> Optional[pl.DataFrame]:
    """
    Safely extracts a polars DataFrame from the raw API response dictionary.
    """
    if not response or "data" not in response:
        logging.warning("Response was empty or did not contain a 'data' key.")
        return None
    try:
        raw = response["data"]
        rs = next(r for r in raw["resultSets"] if r["name"] == result_set_name)
        return pl.DataFrame(rs["rowSet"], schema=rs["headers"], orient="row")
    except (StopIteration, KeyError) as e:
        logging.error(f"Could not find result set '{result_set_name}': {e}")
        return None

# ----------------------------
# Core Fetching Logic
# ----------------------------

def fetch_and_save_data(
    endpoint_def: Dict[str, Any], output_path: Path, params: Dict[str, Any]
):
    """
    Generic fetch and save function. It takes endpoint definitions and parameters,
    fetches the data, and saves it to a specified parquet file.
    """
    if output_path.exists():
        logging.info(f"SKIPPING - already exists: {output_path.name}")
        return

    response = fetch(endpoint_def["cls"], params)

    if response:
        # Handle endpoints with single or multiple result sets
        result_set_names = endpoint_def.get("result_sets", [endpoint_def["result_set"]])
        
        if len(result_set_names) > 1:
            # For player-specific endpoints with multiple tables
            for i, name in enumerate(result_set_names):
                df = extract_dataframe_from_response(response, name)
                if df is not None and not df.is_empty():
                    # Suffix file path to avoid overwrites for the same endpoint
                    p = output_path.with_name(f"{output_path.stem}_{i}_{name}.parquet")
                    df.write_parquet(p)
            logging.info(f"SAVED multiple dataframes for: {output_path.stem}")
        else:
            # For league-wide endpoints with a single table
            df = extract_dataframe_from_response(response, result_set_names[0])
            if df is not None and not df.is_empty():
                df.write_parquet(output_path)
                logging.info(f"SAVED: {output_path.name}")


def run_pipeline():
    """
    Main function to execute the full data ingestion pipeline.
    """
    seasons = get_seasons_list()
    seasons = ["2019-20", "2020-21"] # just temporary
    logging.info(f"Pipeline starting for seasons: {seasons}")

    for season in seasons:
        logging.info(f"--- Processing Season: {season} ---")
        season_dir = PROCESSED_DIR / season
        season_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Fetch League-Wide Data ---
        logging.info("=> Stage 1: Fetching league-wide data...")
        for name, endpoint_def in LEAGUE_WIDE_ENDPOINTS.items():
            output_path = season_dir / f"{name}.parquet"
            params = endpoint_def["params"].copy()
            params["season"] = season
            fetch_and_save_data(endpoint_def, output_path, params)

        # # --- Stage 2: Fetch Player-Specific Data ---
        # logging.info("=> Stage 2: Fetching player-specific data...")
        # base_stats_path = season_dir / "base_stats.parquet"
        # if not base_stats_path.exists():
        #     logging.error(
        #         f"Cannot proceed to Stage 2: base_stats.parquet not found for {season}."
        #     )
        #     continue
        
        # # In polars, it's good practice to wrap file reads in try-except blocks
        # try:
        #     base_df = pl.read_parquet(base_stats_path)
        # except Exception as e:
        #     logging.error(f"Could not read parquet file {base_stats_path}: {e}")
        #     continue

        # qualified_players_df = base_df.filter(
        #     (pl.col("GP").cast(pl.Int32) >= MIN_GAMES_PLAYED)
        #     & (pl.col("MIN").cast(pl.Float32) >= MIN_MINUTES_PLAYED)
        # )

        # player_ids = qualified_players_df.get_column("PLAYER_ID").unique().to_list()
        # logging.info(
        #     f"Found {len(player_ids)} players meeting criteria (GP>={MIN_GAMES_PLAYED}, MIN>={MIN_MINUTES_PLAYED}) for {season}."
        # )

        # player_dir = season_dir / "player_specific"
        # player_dir.mkdir(exist_ok=True)

        # for player_id in player_ids:
        #     for name, endpoint_def in PLAYER_SPECIFIC_ENDPOINTS.items():
        #         output_stem = f"{player_id}_{name}"
        #         # We check for existence of the first file to see if we should skip
        #         if list(player_dir.glob(f"{output_stem}*.parquet")):
        #             logging.info(f"SKIPPING - already exists: {output_stem} files")
        #             continue
                
        #         output_path = player_dir / f"{output_stem}.parquet"
        #         params = endpoint_def["params"].copy()
        #         params["Season"] = season
        #         params["PlayerID"] = player_id
        #         fetch_and_save_data(endpoint_def, output_path, params)

    logging.info("--- Pipeline finished ---")


if __name__ == "__main__":
    run_pipeline()
