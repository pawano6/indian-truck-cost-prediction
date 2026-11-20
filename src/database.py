"""
database.py — MySQL database integration for logistics data.

Provides connection management and data loading utilities for the
star-schema warehouse design.

Requires MySQL server to be installed and running.
See sql/ directory for schema creation scripts.

Database credentials are loaded from environment variables:
  MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
"""

import os
from typing import Optional, Any
import pandas as pd
import numpy as np

# Optional imports for MySQL support
try:
    import mysql.connector  # type: ignore
    HAS_MYSQL = True
except ImportError:
    mysql = None  # type: ignore
    HAS_MYSQL = False

try:
    from sqlalchemy import create_engine  # type: ignore
    HAS_SQLALCHEMY = True
except ImportError:
    create_engine = None  # type: ignore
    HAS_SQLALCHEMY = False


def is_mysql_available() -> bool:
    """Check if MySQL connector dependencies are installed."""
    return HAS_MYSQL and HAS_SQLALCHEMY


def _get_db_config() -> dict:
    """Load database configuration from environment variables."""
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "indian_logistics_db"),
    }


def get_connection() -> Any:
    """Get a MySQL connection using environment-based credentials.

    Returns
    -------
    mysql.connector.connection.MySQLConnection

    Raises
    ------
    ImportError
        If mysql-connector-python is not installed.
    ConnectionError
        If connection fails.
    """
    if not HAS_MYSQL or mysql is None:
        raise ImportError(
            "mysql-connector-python is required for database operations. "
            "Install with: pip install mysql-connector-python"
        )

    config = _get_db_config()
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to MySQL at {config['host']}:{config['port']}. "
            f"Ensure MySQL is running and credentials are correct. Error: {e}"
        )


def execute_query(query: str, params: Optional[tuple] = None) -> list[dict]:
    """Execute a SQL query and return results as a list of dicts.

    Parameters
    ----------
    query : str
        SQL query to execute.
    params : tuple, optional
        Query parameters for parameterized queries.

    Returns
    -------
    list[dict]
        Query results. Empty list for non-SELECT queries.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        if cursor.description:
            results = cursor.fetchall()
        else:
            conn.commit()
            results = []
        return results
    finally:
        cursor.close()
        conn.close()


def load_dataframe_to_table(df: pd.DataFrame, table_name: str,
                            if_exists: str = "append") -> int:
    """Load a DataFrame into a MySQL table.

    Parameters
    ----------
    df : pd.DataFrame
    table_name : str
    if_exists : str
        'append' to add rows, 'replace' to drop and recreate.

    Returns
    -------
    int — number of rows inserted.
    """
    if not HAS_SQLALCHEMY or create_engine is None:
        raise ImportError(
            "SQLAlchemy is required for DataFrame loading. "
            "Install with: pip install sqlalchemy mysql-connector-python"
        )

    config = _get_db_config()
    url = (
        f"mysql+mysqlconnector://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )
    engine = create_engine(url)

    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    return len(df)


def export_star_schema(df: pd.DataFrame) -> dict:
    """Transform flat DataFrame into star-schema tables and load into MySQL.

    Creates and populates dim_truck_type, dim_route, and fact_trip_cost.

    Parameters
    ----------
    df : pd.DataFrame
        Full logistics dataset.

    Returns
    -------
    dict with counts: {'dim_truck_type': N, 'dim_route': N, 'fact_trip_cost': N}
    """
    counts = {}

    # Dimension: Truck Type
    truck_types = df[["truck_type", "truck_capacity_tons"]].drop_duplicates()
    truck_types = truck_types.groupby("truck_type")["truck_capacity_tons"].median()
    dim_truck = pd.DataFrame({
        "truck_type": truck_types.index,
        "typical_capacity_tons": truck_types.values,
    })
    counts["dim_truck_type"] = load_dataframe_to_table(
        dim_truck, "dim_truck_type", if_exists="replace"
    )

    # Dimension: Route
    route_cols = ["origin_city", "origin_state", "destination_city", "destination_state"]
    dim_route = df[route_cols].drop_duplicates().reset_index(drop=True)
    counts["dim_route"] = load_dataframe_to_table(
        dim_route, "dim_route", if_exists="replace"
    )

    # Reload dim tables to get auto-generated IDs
    truck_map = {
        row["truck_type"]: row["truck_type_id"]
        for row in execute_query("SELECT truck_type_id, truck_type FROM dim_truck_type")
    }
    route_rows = execute_query(
        "SELECT route_id, origin_city, origin_state, "
        "destination_city, destination_state FROM dim_route"
    )
    route_map = {
        (r["origin_city"], r["origin_state"],
         r["destination_city"], r["destination_state"]): r["route_id"]
        for r in route_rows
    }

    # Fact table
    fact = df.copy()
    fact["truck_type_id"] = fact["truck_type"].map(truck_map)
    fact["route_id"] = fact.apply(
        lambda r: route_map.get(
            (r["origin_city"], r["origin_state"],
             r["destination_city"], r["destination_state"])
        ), axis=1
    )
    fact["trip_month"] = fact["month"]

    # Add computed columns
    if "load_utilization_pct" not in fact.columns:
        fact["load_utilization_pct"] = np.where(
            fact["truck_capacity_tons"] > 0,
            (fact["load_weight_tons"] / fact["truck_capacity_tons"]) * 100,
            None,
        )
    if "estimated_fuel_cost_inr" not in fact.columns:
        fact["estimated_fuel_cost_inr"] = np.where(
            fact["fuel_efficiency_km_per_litre"] > 0,
            (fact["distance_km"] / fact["fuel_efficiency_km_per_litre"])
            * fact["diesel_price_inr_per_litre"],
            None,
        )

    fact_cols = [
        "trip_id", "route_id", "truck_type_id", "distance_km",
        "truck_capacity_tons", "load_weight_tons",
        "fuel_efficiency_km_per_litre", "diesel_price_inr_per_litre",
        "toll_cost_inr", "driver_allowance_inr", "maintenance_cost_inr",
        "loading_unloading_cost_inr", "other_cost_inr",
        "route_factor", "traffic_factor", "trip_month",
        "actual_trip_cost_inr", "freight_revenue_inr", "trip_profit_inr",
        "load_utilization_pct", "estimated_fuel_cost_inr", "record_type",
    ]
    fact_df = fact[[c for c in fact_cols if c in fact.columns]]
    counts["fact_trip_cost"] = load_dataframe_to_table(
        fact_df, "fact_trip_cost", if_exists="replace"
    )

    return counts
