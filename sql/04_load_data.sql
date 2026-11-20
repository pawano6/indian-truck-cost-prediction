-- ============================================================================
-- Script: 04_load_data.sql
-- Purpose: Load data into the Indian logistics star schema
-- Database: indian_logistics_db
-- ============================================================================

SET GLOBAL local_infile = 1;

USE indian_logistics_db;

-- ============================================================================
-- STEP 1: Populate dim_truck_type
-- ============================================================================

INSERT IGNORE INTO dim_truck_type (truck_type, typical_capacity_tons) VALUES
    ('Heavy',   16.0),
    ('Medium',   9.0),
    ('LCV',      3.5),
    ('Trailer', 25.0);

-- ============================================================================
-- STEP 2: Populate dim_route
-- ============================================================================

INSERT IGNORE INTO dim_route (origin_city, origin_state, destination_city, destination_state) VALUES
    ('Mumbai',      'Maharashtra',      'Delhi',        'Delhi'),
    ('Delhi',       'Delhi',            'Mumbai',       'Maharashtra'),
    ('Chennai',     'Tamil Nadu',       'Bangalore',    'Karnataka'),
    ('Bangalore',   'Karnataka',        'Chennai',      'Tamil Nadu'),
    ('Kolkata',     'West Bengal',      'Hyderabad',    'Telangana'),
    ('Hyderabad',   'Telangana',        'Kolkata',      'West Bengal'),
    ('Ahmedabad',   'Gujarat',          'Pune',         'Maharashtra'),
    ('Pune',        'Maharashtra',      'Ahmedabad',    'Gujarat'),
    ('Jaipur',      'Rajasthan',        'Lucknow',      'Uttar Pradesh'),
    ('Lucknow',     'Uttar Pradesh',    'Jaipur',       'Rajasthan'),
    ('Chandigarh',  'Punjab',           'Indore',       'Madhya Pradesh'),
    ('Indore',      'Madhya Pradesh',   'Chandigarh',   'Punjab'),
    ('Nagpur',      'Maharashtra',      'Bhopal',       'Madhya Pradesh'),
    ('Coimbatore',  'Tamil Nadu',       'Kochi',        'Kerala'),
    ('Kochi',       'Kerala',           'Coimbatore',   'Tamil Nadu'),
    ('Visakhapatnam','Andhra Pradesh',  'Bhubaneswar',  'Odisha');

-- ============================================================================
-- STEP 3: Populate fact_trip_cost via Staging Table
-- ============================================================================

DROP TEMPORARY TABLE IF EXISTS staging_trips;

CREATE TEMPORARY TABLE staging_trips (
    record_type                  VARCHAR(50),
    trip_id                      VARCHAR(20),
    origin_city                  VARCHAR(100),
    origin_state                 VARCHAR(100),
    destination_city             VARCHAR(100),
    destination_state            VARCHAR(100),
    distance_km                  INT,
    truck_type                   VARCHAR(50),
    truck_capacity_tons          DECIMAL(5,1),
    load_weight_tons             DECIMAL(5,2),
    fuel_efficiency_km_per_litre DECIMAL(4,2),
    diesel_price_inr_per_litre   DECIMAL(5,2),
    toll_cost_inr                DECIMAL(10,2),
    driver_allowance_inr         DECIMAL(10,2),
    maintenance_cost_inr         DECIMAL(10,2),
    loading_unloading_cost_inr   DECIMAL(10,2),
    other_cost_inr               DECIMAL(10,2),
    route_factor                 DECIMAL(4,3),
    traffic_factor               DECIMAL(4,3),
    month                        TINYINT,
    actual_trip_cost_inr         DECIMAL(12,2),
    freight_revenue_inr          DECIMAL(12,2),
    trip_profit_inr              DECIMAL(12,2)
);

-- UPDATE THE CSV FILENAME BELOW IF DIFFERENT IN YOUR FOLDER
LOAD DATA LOCAL INFILE 'C:/REPLACE_WITH_YOUR_PROJECT_PATH/indian-truck-cost-prediction/data/indian_truck_logistics_synthetic_15000.csv'
INTO TABLE staging_trips
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Ensure any additional routes from the CSV exist in dim_route
INSERT IGNORE INTO dim_route (origin_city, origin_state, destination_city, destination_state)
SELECT DISTINCT origin_city, origin_state, destination_city, destination_state
FROM staging_trips;

-- Insert into fact_trip_cost by joining staging with dimension tables
INSERT INTO fact_trip_cost (
    trip_id, route_id, truck_type_id, distance_km,
    truck_capacity_tons, load_weight_tons,
    fuel_efficiency_km_per_litre, diesel_price_inr_per_litre,
    toll_cost_inr, driver_allowance_inr, maintenance_cost_inr,
    loading_unloading_cost_inr, other_cost_inr,
    route_factor, traffic_factor, trip_month,
    actual_trip_cost_inr, freight_revenue_inr, trip_profit_inr,
    load_utilization_pct, estimated_fuel_cost_inr, record_type
)
SELECT
    s.trip_id,
    dr.route_id,
    dt.truck_type_id,
    s.distance_km,
    s.truck_capacity_tons,
    s.load_weight_tons,
    s.fuel_efficiency_km_per_litre,
    s.diesel_price_inr_per_litre,
    s.toll_cost_inr,
    s.driver_allowance_inr,
    s.maintenance_cost_inr,
    s.loading_unloading_cost_inr,
    s.other_cost_inr,
    s.route_factor,
    s.traffic_factor,
    s.month,
    s.actual_trip_cost_inr,
    s.freight_revenue_inr,
    s.trip_profit_inr,
    ROUND((s.load_weight_tons / NULLIF(s.truck_capacity_tons, 0)) * 100, 2),
    ROUND((s.distance_km / NULLIF(s.fuel_efficiency_km_per_litre, 0)) * s.diesel_price_inr_per_litre, 2),
    s.record_type
FROM staging_trips s
JOIN dim_route dr
    ON  s.origin_city       = dr.origin_city
    AND s.origin_state      = dr.origin_state
    AND s.destination_city  = dr.destination_city
    AND s.destination_state = dr.destination_state
JOIN dim_truck_type dt
    ON s.truck_type = dt.truck_type;

DROP TEMPORARY TABLE IF EXISTS staging_trips;

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

SELECT 'dim_truck_type'  AS table_name, COUNT(*) AS row_count FROM dim_truck_type
UNION ALL
SELECT 'dim_route',                     COUNT(*)              FROM dim_route
UNION ALL
SELECT 'fact_trip_cost',                COUNT(*)              FROM fact_trip_cost;

SELECT * FROM fact_trip_cost LIMIT 5;