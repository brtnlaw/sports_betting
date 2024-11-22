CREATE TABLE IF NOT EXISTS cfb.all_games_log (
    unique_id VARCHAR(32) DEFAULT NULL,
    date DATE,
    all_quarters_scrape BOOLEAN DEFAULT NULL,
    CONSTRAINT unique_id_date UNIQUE (unique_id, date),
    FOREIGN KEY (unique_id) REFERENCES cfb.all_games(unique_id)
);