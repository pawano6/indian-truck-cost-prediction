-- ============================================================================
-- Script: 02_create_tables.sql
-- Purpose: Create star schema tables for Indian truck logistics cost analysis
-- Database: indian_logistics_db
-- Schema: Star schema with 2 dimension tables and 1 fact table
--
-- Star Schema Design:
--   fact_trip_cost (center)
--       ├── dim_truck_type  (truck type dimension)
--       └── dim_route       (route/geography dimension)
--
-- NOTE ON CALENDAR DIMENSION:
--   A dim_calendar table is intentionally NOT included in this schema.
--   The dataset contains only month numbers (1-12) without actual dates,
--   years, or day-level granularity. Therefore, the month is stored as
--   a TINYINT (trip_month) directly in the fact table. If future datasets
--   include full date information, a dim_calendar table should be added.
-- ============================================================================

USE indian_logistics_db;

-- ============================================================================
-- DIMENSION TABLE: dim_truck_type
-- Stores the 4 truck categories used in Indian logistics
-- ============================================================================
DROP TABLE IF EXISTS fact_trip_cost;
DROP TABLE IF EXISTS dim_truck_type;
DROP TABLE IF EXISTS dim_route;

CREATE TABLE dim_truck_type (
    truck_type_id       INT AUTO_INCREMENT PRIMARY KEY,
    truck_type          VARCHAR(50) NOT NULL UNIQUE,
    typical_capacity_tons DECIMAL(5,1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- DIMENSION TABLE: dim_route
-- Stores unique origin-destination city/state combinations
-- Covers 16 cities across Indian states
-- ============================================================================
CREATE TABLE dim_route (
    route_id            INT AUTO_INCREMENT PRIMARY KEY,
    origin_city         VARCHAR(100) NOT NULL,
    origin_state        VARCHAR(100) NOT NULL,
    destination_city    VARCHAR(100) NOT NULL,
    destination_state   VARCHAR(100) NOT NULL,
    UNIQUE KEY uq_route (origin_city, origin_state, destination_city, destination_state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- FACT TABLE: fact_trip_cost
-- Central fact table storing 15,000 synthetic truck trip records
-- Contains all cost components, efficiency metrics, and computed fields
-- ============================================================================
CREATE TABLE fact_trip_cost (
    trip_id                     VARCHAR(20) PRIMARY KEY,
    route_id                    INT NOT NULL,
    truck_type_id               INT NOT NULL,
    distance_km                 INT NOT NULL,
    truck_capacity_tons         DECIMAL(5,1) NOT NULL,
    load_weight_tons            DECIMAL(5,2) NOT NULL,
    fuel_efficiency_km_per_litre DECIMAL(4,2) NOT NULL,
    diesel_price_inr_per_litre  DECIMAL(5,2) NOT NULL,
    toll_cost_inr               DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    driver_allowance_inr        DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    maintenance_cost_inr        DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    loading_unloading_cost_inr  DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    other_cost_inr              DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    route_factor                DECIMAL(4,3) NOT NULL,
    traffic_factor              DECIMAL(4,3) NOT NULL,
    trip_month                  TINYINT NOT NULL CHECK (trip_month BETWEEN 1 AND 12),
    actual_trip_cost_inr        DECIMAL(12,2) NOT NULL,
    freight_revenue_inr         DECIMAL(12,2) NOT NULL,
    trip_profit_inr             DECIMAL(12,2) NOT NULL,
    load_utilization_pct        DECIMAL(5,2) NOT NULL,
    estimated_fuel_cost_inr     DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    record_type                 VARCHAR(50) DEFAULT 'SYNTHETIC_FOR_EDUCATIONAL_USE',

    -- Foreign key constraints linking to dimension tables
    CONSTRAINT fk_route
        FOREIGN KEY (route_id) REFERENCES dim_route(route_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_truck_type
        FOREIGN KEY (truck_type_id) REFERENCES dim_truck_type(truck_type_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- INDEXES for query performance
-- ============================================================================

-- Index on route_id for JOIN performance with dim_route
CREATE INDEX idx_fact_route_id ON fact_trip_cost(route_id);

-- Index on truck_type_id for JOIN performance with dim_truck_type
CREATE INDEX idx_fact_truck_type_id ON fact_trip_cost(truck_type_id);

-- Index on trip_month for monthly trend analysis and filtering
CREATE INDEX idx_fact_trip_month ON fact_trip_cost(trip_month);

-- Composite index for common filter combinations
CREATE INDEX idx_fact_truck_month ON fact_trip_cost(truck_type_id, trip_month);

-- ============================================================================
-- Verify table creation
-- ============================================================================
SHOW TABLES;

DESCRIBE dim_truck_type;
DESCRIBE dim_route;
DESCRIBE fact_trip_cost;
