Notes:
1. to adjust for pace across eras, always use per 100 possessions for VOLUME metrics.
2. use z-score normalization (within each season) to fairly compare players across eras by comparing them relative to their peers. Use this for VOLUME and EFFICIENCY metrics.
    - to avoid distortion from low minute players and garbage time specialists, define the "league population" as rotation players: players that have logged at least 500 minutes
3. tracking stats (e.g., leaguedashptstats) is from 2013-present, play by play stats (e.g., shotchartdetail) is from 1996-present, and box score stats are historical
4. perform soft clustering of player into archetypes using Gaussian Mixture Models (GMM) rather than K-Means
5. normalization strategy for each metric to account for eras: 
    - efficiency (TS%, eFG%, 3P%, etc.) = relative calculations (player% - leagueAvg%)
    - volume (attempts per 100 poss, TRB%, AST%, etc.) = z-score against rotation player population (e.g., compare rebounding dominance of Ben Wallace and Andre Drummond, despite changes in total rebounds available)
    - frequency vectors for shot profile (% of shots at rim, etc.) = raw % of player's own shot diet (since we want to cluster players based on their intent and tendency)
6. utilize PCA and cluster on the PC's
7. initial v1 approach will leverage PCA, K means, and KNN. v2 will leverage
    - goal is to see if v2 actually outperforms v1, but it's hard to quantify. Potential ideas include estimating player's next season stats based on similar matches, and nice undervalued player findings that are actually like "hidden gold in the rough"
    - for undervalued player findings, focus on filtering for players that get traded next season or have major roster changes on their current team; otherwise its quite unlikely the player will actually have a different role and enough opportunity to break out

Engineered features:

USAGE
- Offensive Load ("percentage of team's offensive possessions a player is directly involved in"): Ast*0.1843 + (Pts+TOV)*0.0969 - 2.3021*(3pt proficiency) + 0.0582*(Ast*(Pts+TOV)*3pt proficiency) - 1.1942
    - what is 3pt proficiency?
- True usage ("incorporates potential assists (i.e., passes that lead to shots or fouls) into standard usage"): $$(FGA + 0.44 \times FTA + TOV + PotentialAst) / TeamPoss$$
    - for pre-2013, proxy PotetialAst as Assists*2

CREATION
- Box Creation ("estimation of open shots created per 100 possessions"): $$(Ast - (0.38 \times BoxCr)) \times 0.75 + FGA + FTA \times 0.44 + BoxCr + TOV$$
    - what is 3pt proficiency?

VOLUME
- Scoring volume ("pure scoring output"): PTS per 100 poss (z-scored against rotation player population)

STYLE
- Assist-to-load ratio ("high ratio identifies pass-first players"): $$AST / OffensiveLoad$$

EFFICIENCY
- 
- Four factors ("essential for efficiency analysis"): eFG%, TOV%, REB%, FTr
- Shot profile ("spatial distribution of player's shot attempts)



