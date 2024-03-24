CREATE TABLE basketball.public_id (
    player VARCHAR(255) NOT NULL,
    position VARCHAR(255) NOT NULL,
    team VARCHAR(255) NOT NULL,
    unique_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_player_team_position_triple UNIQUE (player, position, team)
);