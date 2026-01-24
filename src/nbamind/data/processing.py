"""
==========================================================================================
NBAMind: processing.py
------------------------------------------------------------------------------------------
This script constitutes the primary data ingestion pipeline for the Player Similarity Engine.
It systematically fetches NBA player statistics for all seasons from 2000-01 to the present,
focuses on data points that reveal player playstyle, archetypes, and on-court tendencies,
and joins them into a master table.
==========================================================================================
"""
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
PROCESSED_DIR = Path("data/processed")
START_SEASON = 2015 # 2015/16 season
# Minimum thresholds to qualify a player for detailed, per-player API calls
MIN_GAMES_PLAYED = 15
MIN_MPG = 20

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
    "defense_stats": {
        "cls": leaguedashplayerstats.LeagueDashPlayerStats, # ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'DEF_RATING', 'DREB', 'DREB_PCT', 'PCT_DREB', 'STL', 'PCT_STL', 'BLK', 'PCT_BLK', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB', 'OPP_PTS_PAINT', 'DEF_WS', 'DEF_WS_RAW', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'DEF_RATING_RANK', 'DREB_RANK', 'DREB_PCT_RANK', 'PCT_DREB_RANK', 'STL_RANK', 'PCT_STL_RANK', 'BLK_RANK', 'PCT_BLK_RANK', 'OPP_PTS_OFF_TOV_RANK', 'OPP_PTS_2ND_CHANCE_RANK', 'OPP_PTS_FB_RANK', 'OPP_PTS_PAINT_RANK', 'DEF_WS_RANK']
        "params": {
            "measure_type_detailed_defense": "Defense",
            "per_mode_detailed": "Per100Possessions",
            "season_type_all_star": "Regular Season",
            "league_id_nullable": "00",
        },
        "result_set": "LeagueDashPlayerStats",
    },
    "scoring_stats": {
        "cls": leaguedashplayerstats.LeagueDashPlayerStats, # ['PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV', 'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'FGM', 'FGA', 'FG_PCT', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'PCT_FGA_2PT_RANK', 'PCT_FGA_3PT_RANK', 'PCT_PTS_2PT_RANK', 'PCT_PTS_2PT_MR_RANK', 'PCT_PTS_3PT_RANK', 'PCT_PTS_FB_RANK', 'PCT_PTS_FT_RANK', 'PCT_PTS_OFF_TOV_RANK', 'PCT_PTS_PAINT_RANK', 'PCT_AST_2PM_RANK', 'PCT_UAST_2PM_RANK', 'PCT_AST_3PM_RANK', 'PCT_UAST_3PM_RANK', 'PCT_AST_FGM_RANK', 'PCT_UAST_FGM_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'TEAM_COUNT']
        "params": {
            "measure_type_detailed_defense": "Scoring",
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
    "bio_stats": {
        "cls": leaguedashplayerbiostats.LeagueDashPlayerBioStats, # ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'AGE', 'PLAYER_HEIGHT', 'PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT', 'COLLEGE', 'COUNTRY', 'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'GP', 'PTS', 'REB', 'AST', 'NET_RATING', 'OREB_PCT', 'DREB_PCT', 'USG_PCT', 'TS_PCT', 'AST_PCT']
        "params": {
            "per_mode_simple": "PerGame",
            "season_type_all_star": "Regular Season",
            "league_id": "00",
        },
        "result_set": "LeagueDashPlayerBioStats",
    },
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
            "per_mode_time": "PerGame", # Totals, PerGame, Per48, Per40, Per36, or PerMinute
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
        "params": {"league_id": "00", "per_mode_simple": "PerGame", "season_type_all_star": "Regular Season"},
        "result_sets": [
            "ShotClockShooting", # shows fga, fgm, fg_pct data of 3's and 2's for shot clock ranges: [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	SHOT_CLOCK_RANGE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
            "DribbleShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken with different number of dribbles: [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	DRIBBLE_RANGE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
            "GeneralShooting", # shows fga, fgm, fg_pct data of 3's and 2's for certain types (catch and shot, pull ups, less than 10 ft, other): [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	SHOT_TYPE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
            "TouchTimeShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken for varying touch times (i.e., how long player held ball for before shooting): [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	TOUCH_TIME_RANGE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
            "ClosestDefenderShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken from all distances with different closest defender distances: [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	CLOSE_DEF_DIST_RANGE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
            "ClosestDefender10ftPlusShooting", # shows fga, fgm, fg_pct data of 3's and 2's taken from more than 10 ft with different closest defender distances: [PLAYER_ID	PLAYER_NAME_LAST_FIRST	SORT_ORDER	GP	G	CLOSE_DEF_DIST_RANGE	FGA_FREQUENCY	FGM	FGA	FG_PCT	EFG_PCT	FG2A_FREQUENCY	FG2M	FG2A	FG2_PCT	FG3A_FREQUENCY	FG3M	FG3A	FG3_PCT]
        ],
    },
    # "rebounding": {
    #     "cls": playerdashptreb.PlayerDashPtReb,
    #     "params": {"PerMode": "Per100Possessions", "SeasonType": "Regular Season"},
    #     "result_sets": ["OverallRebounding", "ShotTypeRebounding", "ShotDistanceRebounding"],
    # },
    "shot_splits": {
        "cls": playerdashboardbyshootingsplits.PlayerDashboardByShootingSplits,
        "params": {"league_id_nullable": "00", "per_mode_detailed": "Per100Possessions", "season_type_playoffs": "Regular Season"},
        "result_sets": [
            # "Shot5FTPlayerDashboard", # same as ShotEightFtData but just more granular distances
            "Shot8FTPlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's from varying distances (<8 ft, 8-16 ft, 16-24 ft, 24+ ft, backcourt): [GROUP_SET	GROUP_VALUE	FGM	FGA	FG_PCT	FG3M	FG3A	FG3_PCT	EFG_PCT	BLKA	PCT_AST_2PM	PCT_UAST_2PM	PCT_AST_3PM	PCT_UAST_3PM	PCT_AST_FGM	PCT_UAST_FGM	FGM_RANK	FGA_RANK	FG_PCT_RANK	FG3M_RANK	FG3A_RANK	FG3_PCT_RANK	EFG_PCT_RANK	BLKA_RANK	PCT_AST_2PM_RANK	PCT_UAST_2PM_RANK	PCT_AST_3PM_RANK	PCT_UAST_3PM_RANK	PCT_AST_FGM_RANK	PCT_UAST_FGM_RANK]
            "ShotAreaPlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's at varying court locations (restricted area, left corner, etc.): [GROUP_SET	GROUP_VALUE	FGM	FGA	FG_PCT	FG3M	FG3A	FG3_PCT	EFG_PCT	BLKA	PCT_AST_2PM	PCT_UAST_2PM	PCT_AST_3PM	PCT_UAST_3PM	PCT_AST_FGM	PCT_UAST_FGM	FGM_RANK	FGA_RANK	FG_PCT_RANK	FG3M_RANK	FG3A_RANK	FG3_PCT_RANK	EFG_PCT_RANK	BLKA_RANK	PCT_AST_2PM_RANK	PCT_UAST_2PM_RANK	PCT_AST_3PM_RANK	PCT_UAST_3PM_RANK	PCT_AST_FGM_RANK	PCT_UAST_FGM_RANK]
            # "AssistedShotPlayerDashboard",
            "ShotTypeSummaryPlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's for varying shot types (alley oop, bank shot, dunk, fadeaway, etc.): [GROUP_SET	GROUP_VALUE	FGM	FGA	FG_PCT	FG3M	FG3A	FG3_PCT	EFG_PCT	BLKA	PCT_AST_2PM	PCT_UAST_2PM	PCT_AST_3PM	PCT_UAST_3PM	PCT_AST_FGM	PCT_UAST_FGM]
            # "ShotTypePlayerDashboard", # shows fga, fgm, fg_pct data of both assisted and unassisted 3's and 2's for more specific varying shot types (alley oop dunk shot, cutting dunk shot, driving bank hook shot, etc.): [GROUP_SET	GROUP_VALUE	FGM	FGA	FG_PCT	FG3M	FG3A	FG3_PCT	EFG_PCT	BLKA	PCT_AST_2PM	PCT_UAST_2PM	PCT_AST_3PM	PCT_UAST_3PM	PCT_AST_FGM	PCT_UAST_FGM	FGM_RANK	FGA_RANK	FG_PCT_RANK	FG3M_RANK	FG3A_RANK	FG3_PCT_RANK	EFG_PCT_RANK	BLKA_RANK	PCT_AST_2PM_RANK	PCT_UAST_2PM_RANK	PCT_AST_3PM_RANK	PCT_UAST_3PM_RANK	PCT_AST_FGM_RANK	PCT_UAST_FGM_RANK]
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


def get_suffix_for_params(params: Dict[str, Any]) -> str:
    """
    Determines the appropriate column suffix based on the API parameters.
    """
    # Check various keys used by NBA API for mode settings
    mode = (
        params.get("per_mode_detailed")
        or params.get("per_mode_simple")
        or params.get("per_mode_time")
        or params.get("PerMode")
    )
    if mode:
        return f"_{mode}"
    return ""


def apply_unit_suffixes(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    """
    Renames columns to include unit suffixes (e.g., _PerGame, _Per100Possessions)
    based on the fetch parameters.
    """
    suffix = get_suffix_for_params(params)
    if not suffix:
        return df

    # Columns that should never be suffixed
    exact_exclusions = {
        # Identifiers & Metadata
        "PLAYER_ID", "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION",
        "AGE", "GP", "G", "W", "L", "W_PCT", "SEASON_YEAR", "LEAGUE_ID",
        "TEAM_CITY", "TEAM_NAME", "TEAM_CODE", "TEAM_SLUG", "TEAM_COUNT",
        "PLAYER_HEIGHT", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT",
        "COLLEGE", "COUNTRY", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER",
        "ROSTERSTATUS", "FROM_YEAR", "TO_YEAR", "PLAYERCODE", "PLAYER_SLUG",
        "GAMES_PLAYED_FLAG", "OTHERLEAGUE_EXPERIENCE_CH",
        "GROUP_SET", "GROUP_VALUE", 
        "CLOSE_DEF_PERSON_ID", "PLAYER_LAST_TEAM_ID", "PLAYER_LAST_TEAM_ABBREVIATION",
        "PLAYER_POSITION", "FREQ", "MIN", 
        "AST_TO", "PIE"  # Specific ratios/ratings
    }

    # Substring Matches for dimensionless Metrics: if a column name contains ANY of these, it remains unsuffixed.
    rate_indicators = [
        "PCT", "RATE", "RATIO", "RATING", "RANK", 
        "PACE", "AVG", "PER"
    ]

    new_columns = {}
    for col in df.columns:
        if (col in exact_exclusions) or any(x in col for x in rate_indicators):
            continue
        new_columns[col] = f"{col}{suffix}"

    return df.rename(new_columns)


def _pivot_and_widen_player_stats(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Pivots a long DataFrame from the NBA API into a wide, one-row DataFrame.
    This is used for player-specific endpoints that return data in a long format,
    e.g., shooting stats for different dribble counts.

    Args:
        df: The long-format DataFrame, must contain 'GROUP_VALUE'.

    Returns:
        A wide, one-row DataFrame or None if the input is invalid.
    """
    if df is None or "GROUP_VALUE" not in df.columns or df.is_empty():
        return None

    # Identify value columns to pivot (i.e., all columns except identifiers).
    value_cols = [
        c
        for c in df.columns
        if c
        not in [
            "GROUP_VALUE",
            "GROUP_SET",
            "PLAYER_ID",
            "PLAYER_NAME",
            "NICKNAME",
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "AGE",
        ]
    ]

    # Sanitize GROUP_VALUE strings to be valid parts of column names.
    df = df.with_columns(
        sanitized_group=pl.col("GROUP_VALUE").str.replace_all(r"[^a-zA-Z0-9_+()]", "_")
    )

    # Add a dummy index to pivot on, creating a single row output.
    df = df.with_columns(_pivot_index=pl.lit(0))

    # Perform the pivot.
    pivoted_df = df.pivot(
        values=value_cols, index="_pivot_index", on="sanitized_group"
    )

    return pivoted_df.drop("_pivot_index")


# ----------------------------
# Core Logic
# ----------------------------

def fetch_data_for_season(season: str) -> Dict[str, pl.DataFrame]:
    """
    Fetches all league-wide data for a given season and returns a dictionary
    of DataFrames.
    """
    logging.info(f"--- Processing Season: {season} ---")
    season_data = {}

    # Fetch League-Wide Data
    logging.info("=> Fetching league-wide data...")
    for name, endpoint_def in LEAGUE_WIDE_ENDPOINTS.items():
        params = endpoint_def["params"].copy()
        params["season"] = season
        response = fetch(endpoint_def["cls"], params)
        if response:
            df = extract_dataframe_from_response(response, endpoint_def["result_set"])
            if df is not None and not df.is_empty():
                df = apply_unit_suffixes(df, params)
                season_data[name] = df.with_columns(pl.lit(season).alias("SEASON_YEAR"))

    return season_data


def join_dataframes(
    dataframes: Dict[str, pl.DataFrame], season: str
) -> Optional[pl.DataFrame]:
    """
    Joins all league-wide dataframes for a season into a single master dataframe.
    """
    if not dataframes:
        logging.warning(f"No dataframes to join for season {season}.")
        return None

    # Use 'base_stats' as the base, as it contains fundamental player info.
    base_df_name = "base_stats"
    if base_df_name not in dataframes:
        logging.error(f"'{base_df_name}' not found for season {season}, cannot join.")
        return None
    base_df = dataframes[base_df_name]
    join_keys = ["PLAYER_ID", "SEASON_YEAR"]

    # Iteratively join the rest of the dataframes.
    for name, df in dataframes.items():
        if name == base_df_name:
            continue

        processed_df = df.clone()

        if "PLAYER_ID" not in processed_df.columns and "CLOSE_DEF_PERSON_ID" in processed_df.columns:
            logging.info(f"Renaming 'CLOSE_DEF_PERSON_ID' to 'PLAYER_ID' for '{name}' table.")
            processed_df = processed_df.rename({"CLOSE_DEF_PERSON_ID": "PLAYER_ID"})

        # Identify and warn about duplicate columns before the join.
        base_cols = set(base_df.columns)
        right_cols = set(processed_df.columns)
        common_cols = (base_cols & right_cols) - set(join_keys)

        if common_cols:
            logging.warning(
                f"Duplicate columns found when joining '{name}' for season {season}: {common_cols}. "
                f"These will be dropped from the right-side table."
            )
            processed_df = processed_df.drop(list(common_cols))

        # Perform the join
        base_df = base_df.join(processed_df, on=join_keys, how="left")

        # --- Data Integrity Check ---
        # After joining, check if any players from the base table are missing data
        # from the table that was just joined. This is identified by nulls in a
        # column that is unique to the right-side table.
        
        # Heuristic: Pick a column that is likely not null if the data exists.
        # This isn't perfect, but a good proxy.
        unique_col_to_check = next((col for col in processed_df.columns if col not in join_keys), None)
        
        if unique_col_to_check:
            missing_data_count = base_df.filter(pl.col(unique_col_to_check).is_null()).height
            if missing_data_count > 0:
                logging.warning(
                    f"Post-join check: {missing_data_count} players are missing data "
                    f"from the '{name}' table for season {season}. "
                    f"(Checked using column: '{unique_col_to_check}')"
                )

    return base_df


def fetch_and_join_player_specific_data(
    master_season_df: pl.DataFrame, season: str
) -> pl.DataFrame:
    """
    Fetches and joins player-specific data for eligible players.

    This function identifies players who meet minimum game and minute thresholds,
    fetches detailed data for them from player-specific endpoints, and joins
    this new data into the master DataFrame for the season. It is robust to
    endpoints requiring specific parameters like 'team_id' and to variations
    in the returned data columns for different players.

    Args:
        master_season_df: The DataFrame containing joined league-wide data.
        season: The season string (e.g., '2023-24').

    Returns:
        A DataFrame containing data for eligible players, now including the
        detailed player-specific stats.
    """
    logging.info("=> Identifying eligible players for detailed stats...")

    # Filter for players who meet the minimum participation criteria.
    eligible_df = master_season_df.filter(
        (pl.col("GP") >= MIN_GAMES_PLAYED) & (pl.col("MIN") >= MIN_MPG)
    )
    eligible_player_ids = eligible_df["PLAYER_ID"].to_list()
    # eligible_player_ids = [201935, 201952, 202695] # temp placeholder

    if not eligible_player_ids:
        logging.warning(f"No eligible players found for season {season}. Skipping detailed stats.")
        return eligible_df  # Return the empty (but correctly schema'd) DataFrame

    logging.info(f"Found {len(eligible_player_ids)} eligible players. Fetching detailed data...")

    # Create a mapping from player_id to team_id.
    player_team_df = master_season_df.filter(
        pl.col("PLAYER_ID").is_in(eligible_player_ids)
    ).select(["PLAYER_ID", "TEAM_ID"])
    player_id_to_team_id = dict(zip(player_team_df["PLAYER_ID"], player_team_df["TEAM_ID"]))


    all_players_detailed_stats = []
    for player_id in eligible_player_ids:
        logging.debug(f"Fetching detailed stats for Player ID: {player_id}")
        # Start with a base DataFrame for the player to join all pivoted results onto.
        player_dfs_to_join = [
            pl.DataFrame({"PLAYER_ID": [player_id], "SEASON_YEAR": [season]})
        ]

        for endpoint_name, endpoint_def in PLAYER_SPECIFIC_ENDPOINTS.items():
            params = endpoint_def["params"].copy()
            params["season"] = season
            params["player_id"] = player_id

            # The PlayerDashPtShots endpoint specifically requires a team_id.
            if endpoint_def["cls"] == playerdashptshots.PlayerDashPtShots:
                team_id = player_id_to_team_id.get(player_id)
                if team_id is None:
                    logging.warning(
                        f"No TEAM_ID found for player {player_id}. "
                        f"Skipping '{endpoint_name}' for season {season}."
                    )
                    continue
                params["team_id"] = team_id

            response = fetch(endpoint_def["cls"], params)

            if response:
                for rs_name in endpoint_def["result_sets"]:
                    rs_df = extract_dataframe_from_response(response, rs_name)
                    if rs_df is not None and not rs_df.is_empty():
                        rs_df = apply_unit_suffixes(rs_df, params)
                    # Pivot the long-form data into a single wide row.
                    pivoted_df = _pivot_and_widen_player_stats(
                        # rs_df, f"{endpoint_name}_{rs_name}_"
                        rs_df
                    )
                    if pivoted_df is not None:
                        player_dfs_to_join.append(pivoted_df)

        # Horizontally concatenate all pivoted DataFrames for the current player.
        if len(player_dfs_to_join) > 1:
            all_players_detailed_stats.append(pl.concat(player_dfs_to_join, how="horizontal"))

    if not all_players_detailed_stats:
        logging.warning(f"Failed to fetch any player-specific data for season {season}.")
        return eligible_df

    # Combine all individual player DataFrames into one large DataFrame.
    # Use `how="diagonal"` to handle cases where players have different sets of stats,
    # which results in different columns. Polars will fill missing values with nulls.
    player_specific_df = pl.concat(all_players_detailed_stats, how="diagonal")

    # Join the detailed player stats back to the filtered master DataFrame.
    final_df = eligible_df.join(
        player_specific_df, on=["PLAYER_ID", "SEASON_YEAR"], how="left"
    )
    logging.info(f"Successfully joined detailed stats for {final_df.height} players.")

    return final_df


def run_pipeline() -> Optional[Path]:
    """
    Main function to execute the full data ingestion and processing pipeline.
    It fetches data for each season, joins it into a master table per season,
    and then combines all seasons into a single parquet file.
    """
    seasons = get_seasons_list(START_SEASON) # 10 years worth of data: 2015-16 - 2025-26
    # seasons = ["2019-20", "2020-21"] # just temporary
    logging.info(f"Pipeline starting for seasons: {seasons}")

    all_season_master_dfs = []

    for season in seasons:
        # 1. Fetch all raw league-wide data for the season
        season_dataframes = fetch_data_for_season(season)

        if not season_dataframes:
            logging.warning(f"No league-wide data found for season {season}. Skipping.")
            continue
        
        # 2. Join all league-wide dataframes for the season
        master_season_df = join_dataframes(season_dataframes, season)
        
        if master_season_df is None:
            logging.error(f"Failed to join league-wide data for season {season}. Skipping.")
            continue
            
        # 3. Fetch and join player-specific data for eligible players
        # This returns a DataFrame containing only eligible players with all stats joined.
        final_season_df = fetch_and_join_player_specific_data(master_season_df, season)

        if final_season_df is not None and not final_season_df.is_empty():
            all_season_master_dfs.append(final_season_df)
            logging.info(f"Successfully processed data for season {season}.")

    # 4. Combine all seasons and save to a single file
    if all_season_master_dfs:
        final_master_df = pl.concat(all_season_master_dfs)
        output_path = PROCESSED_DIR / "master_player_analytics.parquet"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        final_master_df.write_parquet(output_path)
        logging.info(f"--- Pipeline finished ---")
        logging.info(f"Master player analytics table saved to: {output_path}")
        logging.info(f"Final table shape: {final_master_df.shape}")
        return output_path
    else:
        logging.warning("--- Pipeline finished, but no data was processed or saved. ---")
        return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_pipeline()
