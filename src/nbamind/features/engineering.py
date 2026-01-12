import polars as pl
from pathlib import Path
from nbamind.data.processing import extract_dataframe_from_response

# ---- config ----
parquet_path = Path("data/processed/master_player_analytics.parquet")  # <- change this
csv_out_path = parquet_path.with_suffix(".csv")      # e.g., final.csv

# ---- read parquet -> Polars DataFrame ----
df = pl.read_parquet(parquet_path)

# ---- inspect ----
print("\n== HEAD ==")
print(df.head(10))  # change row count if you want

print("\n== COLUMNS ==")
print(df.columns)

print("\n== SHAPE ==")
print(df.shape)

# ---- save as CSV ----
df.write_csv(csv_out_path)
print(f"\nSaved CSV to: {csv_out_path.resolve()}")

"""
- eliminate all features with _RANK in name.

USAGE
- offensive load: $$(Ast - (0.38 \times BoxCr)) \times 0.75 + FGA + FTA \times 0.44 + BoxCr + TOV$$
- true usage: $$(FGA + 0.44 \times FTA + TOV + PotentialAst) / TeamPoss$$
- heliocentricity index: $$(USG\% \times 0.5) + (AST\% \times 0.5)$$
- ball dominance: $$TimeOfPossession / Minutes$$
    - since TimeOfPossession in our data is per-game, we should use minutes per-game

CREATION
- box creation: Ast*0.1843 + (Pts+TOV)*0.0969 - 2.3021*(3pt proficiency) + 0.0582*(Ast*(Pts+TOV)*3pt proficiency) - 1.1942
- passing aggression: $$Assists / PotentialAssists$$
- self-created 3 freq: $$Unassisted3PM / 3PM_{Total}$$

VOLUME
- scoring volume: PTSPer100Poss (z-scored)
    - for z scores, to avoid distortion from low minute players and garbage time players, we only consider rotation players with at least 500 min played (however, we need to address this definition for current, in-progress season)
- total reb %: TRB%
    
STYLE
- assist-to-load ratio: $$AST / OffensiveLoad$$
- 3p attempt rate (3PAr): $$3PA / FGA$$
- NEW rim pressure: PCT_PTS_PAINT + FTA_RATE
      floater/midrange: PCT_PTS_MID_RANGE
      perimeter reliance: PCT_PTS_3PT
      remember to make features like PCT_PTS_3PT z-scored


EFFICIENCY
- turnover economy: $$TOV / OffensiveLoad$$
- true shooting added: $$(TS\% - LeagueTS\%) \times Volume_{coeff}$$
- effective fg% (relative): $$eFG\%_{Player} - eFG\%_{League}$$
- NEW true shooting % (relative): $PlayerTS\% - LeagueAverageTS\%$

GRAVITY
- scoring gravity: $$(PTS_{Per100Poss}) \times (TrueShooting_{Rel})$$
    - relative true shooting (???): player % - league average %
- 3pt proficiency: $$(2 / (1 + e^{-3PA_{Per100}})) - 1) \times 3P\%$$

SPATIAL (should we consider per100poss here???)
- rim frequency: $$FGA_{<3ft} / FGA_{Total}$$
- short mid frequency: $$FGA_{3-10ft} / FGA_{Total}$$
- long mid frequency: $$FGA_{10ft-3pt} / FGA_{Total}$$
- corner 3 frequency: $$FGA_{Corner3} / FGA_{Total}$$
- above the break 3: $$FGA_{AB3} / FGA_{Total}$$
- shot range variance: $$StdDev(ShotDistance)$$
- NEW mid-range frequency: $1 - (RimFreq + Corner3Freq + AB3Freq)$

SKILL
- rim efficiency (relative): $$PlayerFG\%_{Rim} - LeagueAvgFG\%_{Rim}$$
- mid efficiency (relative): $$PlayerFG\%_{Mid} - LeagueAvgFG\%_{Mid}$$
- 3pt efficiency (relative): $$PlayerFG\%_{3pt} - LeagueAvgFG\%_{3pt}$$

PHYSICALITY
- free throw rate (FTr): $$FTA / FGA$$
- offensive reb %: ORB%

STRATEGY
- moreyball index: $$(FGA_{Rim} + FGA_{3pt}) / FGA_{Total}$$

IMPACT
- stocks per 100 poss: $$(STL + BLK)$$
- defensive playmaking (per 100 poss): $$STL + BLK + Deflections + Charges$$
- defensive box +/-: $$BPM_{Def}$$
- rim deterrence: $$OppFG\%_{Rim}$$
    - use on/off diff; opponent rim fg% diff (i.e., average % vs % when guarded by player) rather than raw opponent rim fg%

AGGRESSION
- foul rate per 100 poss: PF

ROLE
- block rate: BLK%
- steal rate: STL%
- defensive reb %: DRB%
- defensive load: $$(Reb\% \times 0.4) + (BLK\% \times 0.3) + (STL\% \times 0.3)$$

VERSATILITY
- defensive versatility proxy: $$(BLK\%_{Z} - STL\%_{Z})^2$$ 
- NEW alternate defensive versatility: $$(Z_{\text{DREB}} \times 0.4) + (Z_{\text{STL}} \times 0.3) + (Z_{\text{BLK}} \times 0.3)$$

ACTIVITY
- defensive miles: $$DistTraveled_{Def} / Minutes$$
- NEW offensive miles: $$DistTraveled_{Off} / Minutes$$


PHYSICAL 
- height (z-scored): $$(Height - AvgHeight_{Pos}) / \sigma$$
- weight (z-scored): $$(Weight - AvgWeight_{Pos}) / \sigma$$

CONTEXT
- age: PlayerAge
- experience: YearsPro

4 FACTORS (I believe these are already factored in above)
- shooting: eFG%
- turnovers: TOV%
- rebounding: REB%
- free throws: FTRate

RAW DATA
- 

"""


