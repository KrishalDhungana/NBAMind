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


FINAL FEATURE SET TO BE USED FOR EACH PLAYER/SEASON COMBO:
1. offensive_load: (Ast-(0.38*BoxCr))0.75 + FGA + FTA0.44 + BoxCr + TOV
    - AST = assists per 100 possessions
    - BoxCr = box_creation
    - FGA = field goals attempted per 100 possessions
    - FTA = free throws attempted per 100 possessions
    - TOV = turnovers committed per 100 possessions
    - all of these metrics exist in our data as per-100 possessions; no changes needed (AST, FGA, FTA, TOV, PTS, FG3M, FG3A from base_stats)
2. Heliocentricity: Mean(Z(USG%), Z(AST%), Z(TimeOfPoss/MIN))
    - Z = z-score computed using all eligible players for that particular season
    - USG% and AST% exist in our data (USG_PCT and AST_PCT) and are percentages, so they're pace-independant and ok as-is
    - NOTE: TimeOfPoss appears to be divided by some value (perhaps per-minute or per-24sec) since the highest value is 9.1, but I'm not sure, we need to check through this (!!!!!!!!!!!!!!!!!!)
        - Answer: TimeOfPoss = minutes the player possesses the ball; rewards players that play more minutes, so divide by MIN
3. AVG_SEC_PER_TOUCH, AVG_DRIB_PER_TOUCH (from possessions)
    - these are PerGame, but represent a rate, so they're pace-independent and we can use them as-is
4. 3pt_proficiency: (2/(1 + e^(-3PA)) - 1) * 3P%
    - 3PA = 3-point field goals attempted per 100 possessions (FG3A from base_stats)
    - 3P% = 3-point field goal percentage (FG3_PCT from base_stats)
5. box_creation: Ast*0.1843 + (Pts+TOV)*0.0969 - 2.3021*(3pt_proficiency) + 0.0582*(Ast*(Pts+TOV)*3pt_proficiency) - 1.1942
    - all of these metrics exist in our data as per-100 possessions; no changes needed (AST, PTS, TOV from base_stats)
6. passing_efficiency: AST/POTENTIAL_AST
    - NOTE: AST is per-100 possessions, but POTENTIAL_AST is perGame; we need to convert one to make this a fair ratio: AST_pg = AST_p100 * (possessions_per_game)/100. Then, do AST_pg / POTENTIAL_AST
7. assist_to_load: AST/offensive_load
8. ts_pct_rel: player ts% - league average ts% in season (TS_PCT from adv_stats)
8. scoring_gravity: PTSPer100Poss * TrueShootingPctRel
    - PTSPer100Poss = PTS from base_stats
9. fga_rim_freq = FGA_Restricted / FGA_TOTAL
    - FGA_TOTAL = field goals attempted per 100 possessions (FGA from base_stats)
    - FGA_Restricted = restricted area field goals attempted per 100 possessions (FGA_Restricted_Area from shot_splits)
        - NOTE: the data shows totals for this, we need to convert this to per 100 possessions!!!!!!!
10. fga_floater_freq = FGA_PaintNonRA / FGA_Total
    - FGA_PaintNonRA = paint and non-ra field goals attempted per 100 possessions (FGA_In_The_Paint_(Non_RA) from shot_splits)
11.fga_mid_freq = FGA_MidRange / FGA_Total
    - FGA_MidRange = mid-range field goals attempted per 100 possessions (FGA_Mid_Range from shot_splits)
12 fga_corner_freq = FGA_Corner3 / FGA_Total
    - FGA_Corner3 = corner 3 field goals attempted per 100 possessions (FGA_Left_Corner_3 + FGA_Right_Corner_3 from shot_splits)
13. fga_ab3_freq = FGA_AboveTheBreak3 / FGA_Total
    - FGA_AboveTheBreak3 = above the break 3 field goals attempted per 100 possessions (FGA_Above_the_Break_3 from shot_splits)
14. PCT_UAST_FGM (from shot-splits)
    - NOTE: this field doesn't exist
15. pullup_fga_freq = PullUp_FGA / FGA; repeat for DRIVE_FGA, CATCH_SHOOT_FGA, PAINT_TOUCH_FGA, POST_TOUCH_FGA, and ELBOW_TOUCH_FGA
    - NOTE: PullUp_FGA comes from PULL_UP_FGA in pull_up_shot but is perGame, so we need to convert it to per 100 possessions since FGA is per 100 possessions; the same applies to DRIVE, CATCH_SHOOT, etc.
16. drive_rate = DRIVES per 100 possessions
    - NOTE: convert DRIVES in drives from perGame to per 100 possessions
17. elbow_touch_freq = ElbowTouches / TotalTouches; repeat for POST_TOUCHES and PAINT_TOUCHES
    - ElbowTouches = ELBOW_TOUCHES from possessions
    - TotalTouches = TOUCHES from possessions
    - NOTE: these are PerGame, but need to convert to Per 100 possessions since we're computing a ratio
19. tov_economy: TOV / offensive_load
    - no changes needed, TOV from base_stats is already per 100 possessions
20. ft_rate: FTA / FGA
    - no changes needed, FTA and FGA from base_stats are already per 100 possessions
21. OREB_PCT and DREB_PCT from adv_stats (already per 100 possessions)
22. PCT_BLOCK and PCT_STEAL from defense_stats (already per 100 possessions)
23. stocks: STL + BLK from base_stats (already per 100 possessions)
24. def_versatility: 1/(1+abs(Z_BLK - Z_STL))
    - Z_BLK and Z_STL = z scores of BLK and STL per 100 possessions
25. rim_deterrence: OppFG_Rim - LeagueAvgRim
    - OppFG_Rim = DEF_RIM_FG_PCT from defense (no need to convert to per 100 possessions)
    - LeagueAvgRim = col sum FGM_Restricted_Area / col sum FGA_Restricted_Area (no need to convert to per 100 possessions)
26. rim_contest_rate: DFGA_Rim / MIN1
    - DFGA_Rim = DEF_RIM_FGA from defense
    - NOTE: should we convert DFGA_Rim (currently perGame) to per 100 possessions? If so, do we still need to divide by MIN???????????
27. CONTESTED_SHOTS (from hustle)
    - NOTE: need to convert from perGame to per 100 possessions
27. PCT_PLUSMINUS (fg% when defended by player - normal fg% (lower is better))
28. def_miles_per_min: DIST_MILES_DEF / MIN1
    - NOTE: DIST_MILES_DEF is PerGame from speed_distance; is dividing by MIN1 from adv_stats fair?
29. offensive_miles_per_min: DIST_MILES_OFF / MIN1
30. AVG_SPEED, AVG_SPEED_OFF, AVG_SPEED_DEF (from speed_distance)
30. DEFLECTIONS, CHARGES_DRAWN, SCREEN_ASSISTS, LOOSE_BALLS_RECOVERED, BOX_OUTS (from hustle)
    - NOTE: need to convert from perGame to per 100 possessions
31. Z(PLAYER_HEIGHT_INCHES) and Z(PLAYER_WEIGHT) (from bio_stats)
    - z score of height and weight
32. Z(AGE) (from bio_stats)
33. PF (from base_stats)
34. rim_efficiency_rel: FG_PCT_Restricted_Area - LeagueAvgFG_PCT_Restricted_Area
    - LeagueAvgFG_PCT_Restricted_Area = col sum FGM_Restricted_Area / col sum FGA_Restricted_Area
35. mid_efficiency_rel: FG_PCT_Mid_Range - LeagueAvgFG_PCT_Mid_Range
    - LeagueAvgFG_PCT_Mid_Range = col sum FGM_Mid_Range / col sum FGA_Mid_Range
36. 3pt_efficiency_rel: FG3_PCT - LeagueAvgFG3_PCT
    - LeagueAvgFG3_PCT = col sum FG3M / col sum FG3A
37.




- rim efficiency (relative): $$PlayerFG\%_{Rim} - LeagueAvgFG\%_{Rim}$$
- mid efficiency (relative): $$PlayerFG\%_{Mid} - LeagueAvgFG\%_{Mid}$$
- 3pt efficiency (relative): $$PlayerFG\%_{3pt} - LeagueAvgFG\%_{3pt}$$

FG_PCT_Less_Than_8_ft_	FG_PCT_8_16_ft_	FG_PCT_16_24_ft_	FG_PCT_24+_ft_	FG_PCT_Back_Court_Shot
EFG_PCT_Less_Than_8_ft_	EFG_PCT_8_16_ft_	EFG_PCT_16_24_ft_	EFG_PCT_24+_ft_	EFG_PCT_Back_Court_Shot
PCT_AST_FGM_Less_Than_8_ft_	PCT_AST_FGM_8_16_ft_	PCT_AST_FGM_16_24_ft_	PCT_AST_FGM_24+_ft_	PCT_AST_FGM_Back_Court_Shot	PCT_UAST_FGM_Less_Than_8_ft_	PCT_UAST_FGM_8_16_ft_	PCT_UAST_FGM_16_24_ft_	PCT_UAST_FGM_24+_ft_	PCT_UAST_FGM_Back_Court_Shot
FG_PCT_Restricted_Area	FG_PCT_In_The_Paint_(Non_RA)	FG_PCT_Mid_Range	FG_PCT_Left_Corner_3	FG_PCT_Right_Corner_3	FG_PCT_Above_the_Break_3	FG_PCT_Backcourt
EFG_PCT_Restricted_Area	EFG_PCT_In_The_Paint_(Non_RA)	EFG_PCT_Mid_Range	EFG_PCT_Left_Corner_3	EFG_PCT_Right_Corner_3	EFG_PCT_Above_the_Break_3	EFG_PCT_Backcourt
PCT_AST_FGM_Restricted_Area	PCT_AST_FGM_In_The_Paint_(Non_RA)	PCT_AST_FGM_Mid_Range	PCT_AST_FGM_Left_Corner_3	PCT_AST_FGM_Right_Corner_3	PCT_AST_FGM_Above_the_Break_3	PCT_AST_FGM_Backcourt	PCT_UAST_FGM_Restricted_Area	PCT_UAST_FGM_In_The_Paint_(Non_RA)	PCT_UAST_FGM_Mid_Range	PCT_UAST_FGM_Left_Corner_3	PCT_UAST_FGM_Right_Corner_3	PCT_UAST_FGM_Above_the_Break_3	PCT_UAST_FGM_Backcourt
FG_PCT_Alley_Oop	FG_PCT_Bank_Shot	FG_PCT_Dunk	FG_PCT_Fadeaway	FG_PCT_Finger_Roll	FG_PCT_Hook_Shot	FG_PCT_Jump_Shot	FG_PCT_Layup	FG_PCT_Tip_Shot
EFG_PCT_Alley_Oop	EFG_PCT_Bank_Shot	EFG_PCT_Dunk	EFG_PCT_Fadeaway	EFG_PCT_Finger_Roll	EFG_PCT_Hook_Shot	EFG_PCT_Jump_Shot	EFG_PCT_Layup	EFG_PCT_Tip_Shot
PCT_AST_FGM_Alley_Oop	PCT_AST_FGM_Bank_Shot	PCT_AST_FGM_Dunk	PCT_AST_FGM_Fadeaway	PCT_AST_FGM_Finger_Roll	PCT_AST_FGM_Hook_Shot	PCT_AST_FGM_Jump_Shot	PCT_AST_FGM_Layup	PCT_AST_FGM_Tip_Shot	PCT_UAST_FGM_Alley_Oop	PCT_UAST_FGM_Bank_Shot	PCT_UAST_FGM_Dunk	PCT_UAST_FGM_Fadeaway	PCT_UAST_FGM_Finger_Roll	PCT_UAST_FGM_Hook_Shot	PCT_UAST_FGM_Jump_Shot	PCT_UAST_FGM_Layup	PCT_UAST_FGM_Tip_Shot


to get Possessions_PerGame, calculate (PACE (from adv_stats)/48)*MIN (from bio_stats); we use this to normalize necessary features to per_100_poss
NOTE: FGA, FGA_8_16_ft_, etc. are TOTALS; they don't accurately reflect per 100 poss. We need to convert these. Nevermind, I flipped the order so base_stats is first

MIN is per-100 poss, MIN1 is per game. Keep this in mind.

possessions per game = PACE/48 * MIN1
per_100_poss_stat = (per_game_stat/possessions_per_game) * 100

"""


