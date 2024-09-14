CREATE TABLE IF NOT EXISTS cfb.all_games (
    date DATE,
    home VARCHAR(3),
    home_points INT,
    visitor VARCHAR(3),
    visitor_points INT,
    home_first_quarter INT DEFAULT 0,
    home_first_half INT DEFAULT 0,
    home_third_quarter INT DEFAULT 0,
    visitor_first_quarter INT DEFAULT 0,
    visitor_first_half INT DEFAULT 0,
    visitor_third_quarter INT DEFAULT 0,
    home_rank INT DEFAULT None,
    visitor_rank INT DEFAULT None,
    ot VARCHAR(3) DEFAULT No,
    unique_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_date_visitor_home UNIQUE (date, visitor, home)
);