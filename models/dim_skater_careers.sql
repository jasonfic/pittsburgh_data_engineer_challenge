{{ config(materialized='table') }}

with source_data as (

    SELECT DISTINCT nhl_player_id, player_name, position_group,
        SUM(gp) AS career_games_played, SUM(goals) AS career_goals,
        SUM(assists) AS career_assists, SUM(points) AS career_points,
        ROUND(SUM(toi_seconds) / 60.0, 1) AS career_toi_minutes,
        ROUND(SUM(points)/SUM(gp), 2) AS career_ppg
    FROM `trim-icon-437420-r5.pens_interview_skater_seasons.skater_seasons`
    GROUP BY nhl_player_id, player_name, position_group

)

select *
from source_data
where career_games_played > 10
