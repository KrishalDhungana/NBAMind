from nba_api.stats.endpoints import commonallplayers, leaguedashplayerbiostats, leaguedashplayerstats, leaguedashplayershotlocations, leaguedashptdefend, leaguehustlestatsplayer, playbyplayv3
from nbamind.data.fetcher import fetch
import pandas as pd
import polars as pl
# pl.Config.set_tbl_cols(-1)
import duckdb

from datetime import date
from typing import List, Optional

def print_details(result, endpoint):
    raw = result["data"]
    rs = next(r for r in raw["resultSets"] if r["name"] == endpoint)
    df = pl.DataFrame(rs["rowSet"], schema=rs["headers"], orient="row")
    print(df)
    print(df.columns)
    return

def seasons_2000_to_now(today: Optional[date] = None) -> List[str]:
    """
    Return NBA 'Season' parameter strings from 2000-01 up through the *current* season.

    Rule:
    - If we're in Oct/Nov/Dec, the current season is year-(year+1) (e.g., Nov 2025 => 2025-26)
    - Otherwise, the current season is (year-1)-year (e.g., May 2025 => 2024-25)
    """
    d = today or date.today()
    current_start_year = d.year if d.month >= 10 else d.year - 1  # season starts in October
    start_year = 2000
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(start_year, current_start_year + 1)]


if __name__ == "__main__":

    seasons_list = seasons_2000_to_now()
    print(seasons_list)


    # fetches all active players in a season, but doesn't seem too useful tbh
    result = fetch(commonallplayers.CommonAllPlayers, {"league_id": "00", "season": "2019-20"})
    print_details(result, "CommonAllPlayers")
    # print("endpoint:", result["meta"]["endpoint"])
    # print("data:", result["data"])

    # produce one player/season row
    # collapse into one row if player was traded mid season

    # raw = result["data"]
    # rs = next(r for r in raw["resultSets"] if r["name"] == "CommonAllPlayers")

    # players = pl.DataFrame(rs["rowSet"], schema=rs["headers"], orient="row")
    # active_players = players.filter(
    #     (pl.col("ROSTERSTATUS") == 1) &
    #     (pl.col("GAMES_PLAYED_FLAG") == "Y")
    # )
    # print(active_players.head())
    # filter for ROSTERSTATUS=1 and GAMES_PLAYED_FLAG=Y

    # df.write_parquet("data/processed/common_all_players_2019_20.parquet")

    # con = duckdb.connect("data/nba.duckdb")
    # con.execute("""
    # create or replace table common_all_players as
    # select * from read_parquet('data/processed/common_all_players_2019_20.parquet')
    # """)
    # print(con.execute("select count(*) from common_all_players").fetchone())


    result = fetch(leaguedashplayerbiostats.LeagueDashPlayerBioStats, {"league_id": "00", 
                                                                       "per_mode_simple": "PerGame",
                                                                       "season": "2019-20",
                                                                       "season_type_all_star": "Regular Season",
                                                                       })
    print_details(result, "LeagueDashPlayerBioStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Base",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Advanced",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Advanced",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Misc",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    # result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Four Factors",
    #                                                              # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
    #                                                              "per_mode_detailed": "Per100Possessions",
    #                                                              "season": "2019-20",
    #                                                              "season_type_all_star": "Regular Season",
    #                                                              "league_id_nullable": "00",
    #                                                              })
    # print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Scoring",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    # result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Opponent",
    #                                                              # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
    #                                                              "per_mode_detailed": "Per100Possessions",
    #                                                              "season": "2019-20",
    #                                                              "season_type_all_star": "Regular Season",
    #                                                              "league_id_nullable": "00",
    #                                                              })
    # print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Usage",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayerstats.LeagueDashPlayerStats, {"measure_type_detailed_defense": "Defense",
                                                                 # "pace_adjust": "Y", # unsure if we should use this; since we're already using Per100Possessions, we can omit this and take pace_adjust="N" (default)
                                                                 "per_mode_detailed": "Per100Possessions",
                                                                 "season": "2019-20",
                                                                 "season_type_all_star": "Regular Season",
                                                                 "league_id_nullable": "00",
                                                                 })
    print_details(result, "LeagueDashPlayerStats")

    result = fetch(leaguedashplayershotlocations.LeagueDashPlayerShotLocations, {"measure_type_simple": "Base",
                                                                                 # "pace_adjust": "Y",
                                                                                 "per_mode_detailed": "Per100Possessions",
                                                                                 "season": "2019-20",
                                                                                 "season_type_all_star": "Regular Season",
                                                                                 "league_id_nullable": "00",
                                                                                })
    # print_details(result, "ShotLocations")

    result = fetch(leaguedashptdefend.LeagueDashPtDefend, {"defense_category": "Overall", # can also do 3 Pointers, 2 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft
                                                           "league_id": "00",
                                                           "per_mode_simple": "PerGame",
                                                           "season": "2019-20",
                                                           "season_type_all_star": "Regular Season",
                                                          })
    print_details(result, "LeagueDashPTDefend")

    result = fetch(leaguehustlestatsplayer.LeagueHustleStatsPlayer, {"per_mode_time": "Per36",
                                                                     "season": "2019-20",
                                                                     "season_type_all_star": "Regular Season",
                                                                     "league_id_nullable": "00",
                                                                    })
    print_details(result, "HustleStatsPlayer")

    # result = fetch(playbyplayv3.PlayByPlayV3, {"start_period": "all",
    #                                            "end_period": "all",
    #                                            "game_id": "SOME GAME ID HERE",
    #                                           })
    # print_details(result, "PlayByPlayV3")

    # consider playerdashboardbyshootingsplits; ShotTypePlayerDashboard breaks down the different types of shots they take (e.g., amount of alley oop dunks, amount of cutting finger roll layups, etc.), ShotTypeSummaryPlayerDashboard gives more general shot types (like just alley oop, bank shot, dunk, etc.) ShotAreaPlayerDashboard shows where on the court they shot from, and we can also see unassisted vs assisted shot frequency; this is very useful for determining play style
    # playerdashptreb gives more details on rebounding (e.g., contested vs uncontested rebounds, distance of shots rebounded, distance of rebounds themselves)
    # playerdashptshotdefend gives more nuance on defended shots I think; output isn't there but it gives defense category and the difference in percentage when the given player is the closest defender
    # playerdashptshots gives rich detail on shots a player takes (like shot types (catch and shoot, pull ups, etc.), time in shot clock, number of dribbles taken before shooting, distance of closest defender when shooting (either overall or for 10+ feet shots), and touch time before shooting)
    # playerindex seems like a better primary table than commonallplayers; it shows all players, if they're defunct, their current team, height/weight/position, draft year, roster status, years they played (from and to), and their career pts/reb/ast; potential risk here is that perhaps a player didn't play for a season or two before returning back to the league (in that case those missing years will be between FROM_YEAR and TO_YEAR, but data obviously won't exist)
    # shotchartdetail also shows rich shot type data and where on the court it was taken for a given player/game combo; however this might be overkill since we can just look at a player's shot distribution overall for a given season rather than granularly for each individual game

    # proposal: rough idea is to fetch ALL player/season combos from 2000 to now, filter for only seasons where players played a sufficient amount (e.g., minimum games and minutes), and then for those players only, fetch granular data for endpoints that require to be called individually for each player id 