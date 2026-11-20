-- ============================================================================
-- Script: 01_create_database.sql
-- Purpose: Create the Indian Logistics database for truck cost prediction
-- Database: indian_logistics_db
-- Engine: MySQL 8.0+
-- Dataset: 15,000 synthetic Indian truck trip records (educational use)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS indian_logistics_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Verify creation
SHOW DATABASES LIKE 'indian_logistics_db';

-- Switch to the new database
USE indian_logistics_db;

-- Confirm charset settings
SELECT @@character_set_database AS db_charset,
       @@collation_database    AS db_collation;
