CREATE TABLE IF NOT EXISTS cfb.venues (
    id                  INT PRIMARY KEY,
    name                VARCHAR(255),
    city                VARCHAR(100),
    state               VARCHAR(50),
    zip                 VARCHAR(20),
    countryCode         VARCHAR(10),
    latitude            DECIMAL(10, 6),
    longitude           DECIMAL(10, 6),
    capacity            INT,
    dome                BOOLEAN,
    timezone            VARCHAR(50),
    elevation           DECIMAL(10, 6),
    constructionYear    INT,
    grass               BOOLEAN
);
