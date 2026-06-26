{{ config(materialized='table') }}

WITH source_data AS (

    SELECT nhl_player_id, player_name, position_group,
        SUM(gp) AS career_games_played, SUM(goals) AS career_goals,
        SUM(assists) AS career_assists, SUM(points) AS career_points,
        ROUND(SUM(toi_seconds) / 60.0, 1) AS career_toi_minutes,
        ROUND(SUM(points)/SUM(gp), 2) AS career_ppg
    FROM `skater_seasons`
    GROUP BY nhl_player_id, player_name, position_group
    HAVING SUM(toi_seconds) > 10000

)

SELECT *
FROM source_data
WHERE career_games_played > 25
    AND nhl_player_id IS NOT NULL
