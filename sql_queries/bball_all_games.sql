CREATE TABLE IF NOT EXISTS basketball.all_games (
    date DATE,
    start_time TIME,
    visitor VARCHAR(3),
    visitor_points INT,
    home VARCHAR(3),
    home_points INT,
    ot VARCHAR(3),
    attendance INT,
    unique_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_date_visitor_home UNIQUE (date, visitor, home)
);