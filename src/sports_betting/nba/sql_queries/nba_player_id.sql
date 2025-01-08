CREATE TABLE IF NOT EXISTS nba.player_id (
    player VARCHAR(255) NOT NULL,
    team VARCHAR(255) NOT NULL,
    unique_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_player_team UNIQUE (player, team)
);