CREATE TABLE IF NOT EXISTS cfb.all_games (
    date DATE,
    home VARCHAR,
    home_points INT,
    visitor VARCHAR,
    visitor_points INT,
    home_first_quarter INT DEFAULT 0,
    home_first_half INT DEFAULT 0,
    home_third_quarter INT DEFAULT 0,
    home_fourth_quarter INT DEFAULT 0,
    home_ot INT DEFAULT 0,
    visitor_first_quarter INT DEFAULT 0,
    visitor_first_half INT DEFAULT 0,
    visitor_third_quarter INT DEFAULT 0,
    visitor_fourth_quarter INT DEFAULT 0,
    visitor_ot INT DEFAULT 0,
    home_rank INT DEFAULT NULL,
    visitor_rank INT DEFAULT NULL,
    ot VARCHAR(3) DEFAULT NULL, --OT, 2OT, 3OT, etc. 
    unique_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_date_visitor_home UNIQUE (date, visitor, home)
);