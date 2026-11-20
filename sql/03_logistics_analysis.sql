-- ============================================================================
-- Script: 03_logistics_analysis.sql
-- Purpose: Comprehensive analytics queries for Indian truck logistics data
-- Database: indian_logistics_db
-- Schema: Star schema (fact_trip_cost, dim_truck_type, dim_route)
--
-- This script demonstrates a wide range of SQL techniques including:
--   - Aggregations (GROUP BY, HAVING)
--   - JOINs (INNER, LEFT)
--   - Window functions (RANK, DENSE_RANK, rolling averages)
--   - CTEs (Common Table Expressions)
--   - Subqueries
--   - CASE WHEN conditional logic
--   - Percentage and weighted calculations
--
-- All monetary values are in Indian Rupees (INR - ₹)
-- ============================================================================

USE indian_logistics_db;

SET SESSION sql_mode = (SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''));
-- ============================================================================
-- QUERY 1: Average Cost by Truck Type
-- ============================================================================
-- Demonstrates: INNER JOIN with dimension table, GROUP BY, aggregate functions
-- Purpose: Compare average trip costs across the 4 truck categories
--          (Heavy, Medium, LCV, Trailer) to identify cost-intensive vehicle types
-- ============================================================================

SELECT
    dt.truck_type,
    dt.typical_capacity_tons,
    COUNT(f.trip_id)                            AS total_trips,
    ROUND(AVG(f.actual_trip_cost_inr), 2)       AS avg_trip_cost_inr,
    ROUND(AVG(f.distance_km), 0)                AS avg_distance_km,
    ROUND(AVG(f.load_weight_tons), 2)           AS avg_load_tons,
    ROUND(MIN(f.actual_trip_cost_inr), 2)       AS min_cost_inr,
    ROUND(MAX(f.actual_trip_cost_inr), 2)       AS max_cost_inr
FROM fact_trip_cost f
INNER JOIN dim_truck_type dt
    ON f.truck_type_id = dt.truck_type_id
GROUP BY dt.truck_type, dt.typical_capacity_tons
ORDER BY avg_trip_cost_inr DESC;


-- ============================================================================
-- QUERY 2: Top 10 Most Expensive Routes
-- ============================================================================
-- Demonstrates: JOIN with dim_route, GROUP BY, ORDER BY DESC, LIMIT
-- Purpose: Identify the costliest origin-destination pairs to flag
--          routes that may need operational optimization
-- ============================================================================

SELECT
    dr.origin_city,
    dr.origin_state,
    dr.destination_city,
    dr.destination_state,
    COUNT(f.trip_id)                            AS trip_count,
    ROUND(AVG(f.actual_trip_cost_inr), 2)       AS avg_cost_inr,
    ROUND(AVG(f.distance_km), 0)                AS avg_distance_km,
    ROUND(SUM(f.actual_trip_cost_inr), 2)       AS total_cost_inr
FROM fact_trip_cost f
JOIN dim_route dr
    ON f.route_id = dr.route_id
GROUP BY dr.origin_city, dr.origin_state, dr.destination_city, dr.destination_state
ORDER BY avg_cost_inr DESC
LIMIT 10;


-- ============================================================================
-- QUERY 3: Cost per KM by Truck Type with Ranking
-- ============================================================================
-- Demonstrates: Window function RANK(), computed columns, JOINs
-- Purpose: Rank each trip by its cost-per-km within its truck type category
--          to find the most and least efficient trips per vehicle class
-- ============================================================================

SELECT
    ranked.*
FROM (
    SELECT
        f.trip_id,
        dt.truck_type,
        f.distance_km,
        f.actual_trip_cost_inr,
        ROUND(f.actual_trip_cost_inr / NULLIF(f.distance_km, 0), 2) AS cost_per_km_inr,
        RANK() OVER (
            PARTITION BY dt.truck_type
            ORDER BY (f.actual_trip_cost_inr / NULLIF(f.distance_km, 0)) DESC
        ) AS cost_rank
    FROM fact_trip_cost f
    INNER JOIN dim_truck_type dt
        ON f.truck_type_id = dt.truck_type_id
) ranked
WHERE ranked.cost_rank <= 5
ORDER BY ranked.truck_type, ranked.cost_rank;


-- ============================================================================
-- QUERY 4: Monthly Cost Trend with Season Classification
-- ============================================================================
-- Demonstrates: GROUP BY on month, CASE WHEN for seasonal classification
-- Purpose: Analyze how trip costs vary across months and Indian seasons
--          (Winter, Summer, Monsoon, Post-Monsoon) to identify seasonal patterns
-- ============================================================================

SELECT
    f.trip_month,
    CASE
        WHEN f.trip_month IN (12, 1, 2)   THEN 'Winter'
        WHEN f.trip_month IN (3, 4, 5)    THEN 'Summer'
        WHEN f.trip_month IN (6, 7, 8, 9) THEN 'Monsoon'
        WHEN f.trip_month IN (10, 11)     THEN 'Post-Monsoon'
    END                                         AS season,
    COUNT(f.trip_id)                            AS total_trips,
    ROUND(AVG(f.actual_trip_cost_inr), 2)       AS avg_cost_inr,
    ROUND(SUM(f.actual_trip_cost_inr), 2)       AS total_cost_inr,
    ROUND(AVG(f.distance_km), 0)                AS avg_distance_km,
    ROUND(AVG(f.traffic_factor), 3)             AS avg_traffic_factor
FROM fact_trip_cost f
GROUP BY f.trip_month,
         CASE
             WHEN f.trip_month IN (12, 1, 2)   THEN 'Winter'
             WHEN f.trip_month IN (3, 4, 5)    THEN 'Summer'
             WHEN f.trip_month IN (6, 7, 8, 9) THEN 'Monsoon'
             WHEN f.trip_month IN (10, 11)     THEN 'Post-Monsoon'
         END
ORDER BY f.trip_month;


-- ============================================================================
-- QUERY 5: Load Utilization Analysis
-- ============================================================================
-- Demonstrates: CASE WHEN for banding/bucketing, GROUP BY, HAVING clause
-- Purpose: Classify trips by how fully the truck was loaded and analyze
--          cost efficiency at each utilization level. HAVING filters bands
--          with a meaningful number of trips (>= 10).
-- ============================================================================

SELECT
    CASE
        WHEN f.load_utilization_pct < 50                        THEN 'Under-loaded (<50%)'
        WHEN f.load_utilization_pct >= 50 AND f.load_utilization_pct < 75  THEN 'Moderate (50-75%)'
        WHEN f.load_utilization_pct >= 75 AND f.load_utilization_pct < 90  THEN 'Well-loaded (75-90%)'
        WHEN f.load_utilization_pct >= 90                       THEN 'Full (>90%)'
    END                                                         AS utilization_band,
    COUNT(f.trip_id)                                            AS trip_count,
    ROUND(AVG(f.actual_trip_cost_inr), 2)                       AS avg_cost_inr,
    ROUND(AVG(f.load_weight_tons), 2)                           AS avg_load_tons,
    ROUND(AVG(f.actual_trip_cost_inr / NULLIF(f.distance_km, 0)), 2) AS avg_cost_per_km_inr,
    ROUND(AVG(f.trip_profit_inr), 2)                            AS avg_profit_inr
FROM fact_trip_cost f
GROUP BY
    CASE
        WHEN f.load_utilization_pct < 50                        THEN 'Under-loaded (<50%)'
        WHEN f.load_utilization_pct >= 50 AND f.load_utilization_pct < 75  THEN 'Moderate (50-75%)'
        WHEN f.load_utilization_pct >= 75 AND f.load_utilization_pct < 90  THEN 'Well-loaded (75-90%)'
        WHEN f.load_utilization_pct >= 90                       THEN 'Full (>90%)'
    END
HAVING trip_count >= 10
ORDER BY avg_cost_inr DESC;


-- ============================================================================
-- QUERY 6: Route Cost Comparison with CTE
-- ============================================================================
-- Demonstrates: Common Table Expression (CTE), filtering against an aggregate
-- Purpose: Find routes whose average trip cost exceeds the overall average,
--          highlighting routes that are disproportionately expensive
-- ============================================================================

WITH route_costs AS (
    SELECT
        dr.route_id,
        CONCAT(dr.origin_city, ' → ', dr.destination_city)     AS route_name,
        dr.origin_state,
        dr.destination_state,
        COUNT(f.trip_id)                                        AS trip_count,
        ROUND(AVG(f.actual_trip_cost_inr), 2)                   AS avg_cost_inr,
        ROUND(AVG(f.distance_km), 0)                            AS avg_distance_km
    FROM fact_trip_cost f
    JOIN dim_route dr ON f.route_id = dr.route_id
    GROUP BY dr.route_id, dr.origin_city, dr.destination_city,
             dr.origin_state, dr.destination_state
)
SELECT
    rc.route_name,
    rc.origin_state,
    rc.destination_state,
    rc.trip_count,
    rc.avg_cost_inr,
    rc.avg_distance_km,
    ROUND(rc.avg_cost_inr - overall.overall_avg, 2)             AS cost_above_avg_inr
FROM route_costs rc
CROSS JOIN (
    SELECT ROUND(AVG(avg_cost_inr), 2) AS overall_avg
    FROM route_costs
) overall
WHERE rc.avg_cost_inr > overall.overall_avg
ORDER BY rc.avg_cost_inr DESC;


-- ============================================================================
-- QUERY 7: Expense Breakdown as Percentage of Total Cost
-- ============================================================================
-- Demonstrates: Subquery for grand total, percentage calculation
-- Purpose: Break down each expense component (fuel, toll, driver, maintenance,
--          loading/unloading, other) as a percentage of total logistics spend
-- ============================================================================

SELECT
    ROUND(SUM(f.estimated_fuel_cost_inr), 2)        AS total_fuel_cost_inr,
    ROUND(SUM(f.toll_cost_inr), 2)                  AS total_toll_cost_inr,
    ROUND(SUM(f.driver_allowance_inr), 2)           AS total_driver_cost_inr,
    ROUND(SUM(f.maintenance_cost_inr), 2)           AS total_maintenance_cost_inr,
    ROUND(SUM(f.loading_unloading_cost_inr), 2)     AS total_loading_cost_inr,
    ROUND(SUM(f.other_cost_inr), 2)                 AS total_other_cost_inr,
    -- Percentage of grand total
    ROUND(SUM(f.estimated_fuel_cost_inr)    * 100.0 / grand.grand_total, 2) AS fuel_pct,
    ROUND(SUM(f.toll_cost_inr)              * 100.0 / grand.grand_total, 2) AS toll_pct,
    ROUND(SUM(f.driver_allowance_inr)       * 100.0 / grand.grand_total, 2) AS driver_pct,
    ROUND(SUM(f.maintenance_cost_inr)       * 100.0 / grand.grand_total, 2) AS maintenance_pct,
    ROUND(SUM(f.loading_unloading_cost_inr) * 100.0 / grand.grand_total, 2) AS loading_pct,
    ROUND(SUM(f.other_cost_inr)             * 100.0 / grand.grand_total, 2) AS other_pct
FROM fact_trip_cost f
CROSS JOIN (
    SELECT SUM(actual_trip_cost_inr) AS grand_total
    FROM fact_trip_cost
) grand;


-- ============================================================================
-- QUERY 8: Weighted Average Cost per KM
-- ============================================================================
-- Demonstrates: Weighted average formula — SUM(cost * distance) / SUM(distance)
-- Purpose: Compute a distance-weighted average cost per km by truck type,
--          giving more influence to longer trips (which carry more cost weight)
-- ============================================================================

SELECT
    dt.truck_type,
    COUNT(f.trip_id)                                            AS trip_count,
    ROUND(SUM(f.actual_trip_cost_inr), 2)                       AS total_cost_inr,
    SUM(f.distance_km)                                          AS total_distance_km,
    ROUND(
        SUM(f.actual_trip_cost_inr * f.distance_km)
        / NULLIF(SUM(f.distance_km), 0),
    2)                                                          AS weighted_avg_cost_per_km_inr,
    ROUND(AVG(f.actual_trip_cost_inr / NULLIF(f.distance_km, 0)), 2) AS simple_avg_cost_per_km_inr
FROM fact_trip_cost f
INNER JOIN dim_truck_type dt
    ON f.truck_type_id = dt.truck_type_id
GROUP BY dt.truck_type
ORDER BY weighted_avg_cost_per_km_inr DESC;


-- ============================================================================
-- QUERY 9: Profitable vs Unprofitable Trips
-- ============================================================================
-- Demonstrates: CASE WHEN for binary classification, percentage calculation
-- Purpose: Categorize trips as profitable or unprofitable and compute the
--          share of each, along with average profit/loss per category
-- ============================================================================

SELECT
    CASE
        WHEN f.trip_profit_inr > 0 THEN 'Profitable'
        ELSE 'Unprofitable'
    END                                                         AS profitability,
    COUNT(f.trip_id)                                            AS trip_count,
    ROUND(COUNT(f.trip_id) * 100.0 / (SELECT COUNT(*) FROM fact_trip_cost), 2)
                                                                AS pct_of_total,
    ROUND(AVG(f.trip_profit_inr), 2)                            AS avg_profit_inr,
    ROUND(SUM(f.trip_profit_inr), 2)                            AS total_profit_inr,
    ROUND(AVG(f.actual_trip_cost_inr), 2)                       AS avg_cost_inr,
    ROUND(AVG(f.freight_revenue_inr), 2)                        AS avg_revenue_inr
FROM fact_trip_cost f
GROUP BY
    CASE
        WHEN f.trip_profit_inr > 0 THEN 'Profitable'
        ELSE 'Unprofitable'
    END
ORDER BY profitability;


-- ============================================================================
-- QUERY 10: Dense Rank of Routes by Trip Volume
-- ============================================================================
-- Demonstrates: DENSE_RANK() window function
-- Purpose: Rank routes by how frequently they are used (trip volume),
--          with ties receiving the same rank and no gaps in ranking
-- ============================================================================

SELECT
    ranked_routes.*
FROM (
    SELECT
        CONCAT(dr.origin_city, ' → ', dr.destination_city)     AS route_name,
        dr.origin_state,
        dr.destination_state,
        COUNT(f.trip_id)                                        AS trip_count,
        ROUND(AVG(f.actual_trip_cost_inr), 2)                   AS avg_cost_inr,
        ROUND(AVG(f.distance_km), 0)                            AS avg_distance_km,
        DENSE_RANK() OVER (ORDER BY COUNT(f.trip_id) DESC)      AS volume_rank
    FROM fact_trip_cost f
    JOIN dim_route dr ON f.route_id = dr.route_id
    GROUP BY dr.origin_city, dr.destination_city, dr.origin_state, dr.destination_state
) ranked_routes
WHERE ranked_routes.volume_rank <= 15
ORDER BY ranked_routes.volume_rank;


-- ============================================================================
-- QUERY 11: Cost Distribution by Distance Bands
-- ============================================================================
-- Demonstrates: CASE WHEN for distance range classification
-- Purpose: Segment trips into distance bands (Short, Medium, Long, Very Long)
--          and compare cost profiles across each band
-- ============================================================================

SELECT
    CASE
        WHEN f.distance_km < 300                               THEN 'Short (<300 km)'
        WHEN f.distance_km >= 300 AND f.distance_km < 800      THEN 'Medium (300-800 km)'
        WHEN f.distance_km >= 800 AND f.distance_km < 1200     THEN 'Long (800-1200 km)'
        WHEN f.distance_km >= 1200                             THEN 'Very Long (>1200 km)'
    END                                                         AS distance_band,
    COUNT(f.trip_id)                                            AS trip_count,
    ROUND(AVG(f.actual_trip_cost_inr), 2)                       AS avg_cost_inr,
    ROUND(AVG(f.distance_km), 0)                                AS avg_distance_km,
    ROUND(AVG(f.actual_trip_cost_inr / NULLIF(f.distance_km, 0)), 2) AS avg_cost_per_km_inr,
    ROUND(AVG(f.toll_cost_inr), 2)                              AS avg_toll_inr,
    ROUND(AVG(f.estimated_fuel_cost_inr), 2)                    AS avg_fuel_cost_inr,
    ROUND(AVG(f.trip_profit_inr), 2)                            AS avg_profit_inr
FROM fact_trip_cost f
GROUP BY
    CASE
        WHEN f.distance_km < 300                               THEN 'Short (<300 km)'
        WHEN f.distance_km >= 300 AND f.distance_km < 800      THEN 'Medium (300-800 km)'
        WHEN f.distance_km >= 800 AND f.distance_km < 1200     THEN 'Long (800-1200 km)'
        WHEN f.distance_km >= 1200                             THEN 'Very Long (>1200 km)'
    END
ORDER BY avg_distance_km;


-- ============================================================================
-- QUERY 12: Driver Allowance vs Total Cost Correlation
-- ============================================================================
-- Demonstrates: LEFT JOIN, computed ratios, correlation analysis
-- Purpose: Analyze the relationship between driver allowance and total trip cost
--          across truck types and routes to understand labor cost impact
-- ============================================================================

SELECT
    dt.truck_type,
    CONCAT(dr.origin_city, ' → ', dr.destination_city)         AS route_name,
    COUNT(f.trip_id)                                            AS trip_count,
    ROUND(AVG(f.driver_allowance_inr), 2)                       AS avg_driver_allowance_inr,
    ROUND(AVG(f.actual_trip_cost_inr), 2)                       AS avg_total_cost_inr,
    ROUND(
        AVG(f.driver_allowance_inr) * 100.0
        / NULLIF(AVG(f.actual_trip_cost_inr), 0),
    2)                                                          AS driver_cost_ratio_pct,
    ROUND(AVG(f.distance_km), 0)                                AS avg_distance_km,
    ROUND(
        AVG(f.driver_allowance_inr / NULLIF(f.distance_km, 0)),
    2)                                                          AS driver_cost_per_km_inr
FROM fact_trip_cost f
LEFT JOIN dim_truck_type dt ON f.truck_type_id = dt.truck_type_id
LEFT JOIN dim_route dr      ON f.route_id = dr.route_id
GROUP BY dt.truck_type, dr.origin_city, dr.destination_city
ORDER BY driver_cost_ratio_pct DESC
LIMIT 20;


-- ============================================================================
-- QUERY 13: Rolling Average Cost (3-Month Rolling Average)
-- ============================================================================
-- Demonstrates: Window function with ROWS BETWEEN for rolling averages
-- Purpose: Smooth out monthly cost fluctuations with a 3-month rolling average
--          to identify underlying cost trends without seasonal noise
-- ============================================================================

SELECT
    monthly.trip_month,
    monthly.trip_count,
    monthly.avg_cost_inr,
    monthly.total_cost_inr,
    ROUND(
        AVG(monthly.avg_cost_inr) OVER (
            ORDER BY monthly.trip_month
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ),
    2)                                                          AS rolling_3m_avg_cost_inr,
    ROUND(
        AVG(monthly.total_cost_inr) OVER (
            ORDER BY monthly.trip_month
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ),
    2)                                                          AS rolling_3m_total_cost_inr
FROM (
    SELECT
        f.trip_month,
        COUNT(f.trip_id)                        AS trip_count,
        ROUND(AVG(f.actual_trip_cost_inr), 2)   AS avg_cost_inr,
        ROUND(SUM(f.actual_trip_cost_inr), 2)   AS total_cost_inr
    FROM fact_trip_cost f
    GROUP BY f.trip_month
) monthly
ORDER BY monthly.trip_month;


-- ============================================================================
-- END OF ANALYTICS QUERIES
-- ============================================================================
