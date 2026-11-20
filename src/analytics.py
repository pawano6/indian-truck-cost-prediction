"""
analytics.py — Analytics and reporting utilities for logistics data.

Provides reusable functions for KPIs, route analysis, and cost breakdowns
used by the Streamlit dashboard and notebooks.
"""

import pandas as pd
import numpy as np


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute key performance indicators from the dataset.

    Returns dict with KPI values.
    """
    kpis = {
        "total_trips": len(df),
        "total_distance_km": float(df["distance_km"].sum()),
        "avg_distance_km": float(df["distance_km"].mean()),
        "total_cost_inr": float(df["actual_trip_cost_inr"].sum()),
        "avg_cost_inr": float(df["actual_trip_cost_inr"].mean()),
        "median_cost_inr": float(df["actual_trip_cost_inr"].median()),
        "avg_cost_per_km": float(
            df["actual_trip_cost_inr"].sum() / df["distance_km"].sum()
        ) if df["distance_km"].sum() > 0 else 0,
        "unique_routes": int(
            df.groupby(["origin_city", "destination_city"]).ngroups
        ),
        "truck_types": int(df["truck_type"].nunique()),
        "cities": int(df["origin_city"].nunique()),
    }

    if "load_utilization_pct" in df.columns:
        kpis["avg_load_utilization_pct"] = float(
            df["load_utilization_pct"].mean()
        )

    if "freight_revenue_inr" in df.columns:
        kpis["total_revenue_inr"] = float(df["freight_revenue_inr"].sum())
    if "trip_profit_inr" in df.columns:
        kpis["total_profit_inr"] = float(df["trip_profit_inr"].sum())
        kpis["profitable_trip_pct"] = float(
            (df["trip_profit_inr"] > 0).mean() * 100
        )

    return kpis


def cost_by_truck_type(df: pd.DataFrame) -> pd.DataFrame:
    """Average cost metrics grouped by truck type."""
    agg = df.groupby("truck_type").agg(
        trip_count=("actual_trip_cost_inr", "count"),
        avg_cost=("actual_trip_cost_inr", "mean"),
        avg_distance=("distance_km", "mean"),
        avg_load=("load_weight_tons", "mean"),
    ).round(2)

    agg["avg_cost_per_km"] = (agg["avg_cost"] / agg["avg_distance"]).round(2)
    return agg.sort_values("avg_cost", ascending=False)


def top_routes(df: pd.DataFrame, n: int = 10,
               sort_by: str = "avg_cost") -> pd.DataFrame:
    """Top N routes by average cost or trip count."""
    routes = df.groupby(["origin_city", "destination_city"]).agg(
        trip_count=("actual_trip_cost_inr", "count"),
        avg_cost=("actual_trip_cost_inr", "mean"),
        avg_distance=("distance_km", "mean"),
    ).round(2)

    routes["route"] = (
        routes.index.get_level_values(0) + " → " +
        routes.index.get_level_values(1)
    )
    return routes.sort_values(sort_by, ascending=False).head(n)


def expense_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown of expense components across all trips."""
    expense_cols = {
        "toll_cost_inr": "Toll",
        "driver_allowance_inr": "Driver Allowance",
        "maintenance_cost_inr": "Maintenance",
        "loading_unloading_cost_inr": "Loading/Unloading",
        "other_cost_inr": "Other",
    }

    if "estimated_fuel_cost_inr" in df.columns:
        expense_cols["estimated_fuel_cost_inr"] = "Estimated Fuel"

    present = {k: v for k, v in expense_cols.items() if k in df.columns}
    totals = {v: float(df[k].sum()) for k, v in present.items()}
    grand_total = sum(totals.values())

    breakdown = pd.DataFrame([
        {
            "Category": cat,
            "Total (₹)": total,
            "Percentage": (total / grand_total * 100) if grand_total > 0 else 0,
        }
        for cat, total in totals.items()
    ]).sort_values("Total (₹)", ascending=False)

    return breakdown.reset_index(drop=True)


def load_utilization_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of load utilization bands."""
    if "load_utilization_pct" not in df.columns:
        return pd.DataFrame()

    bins = [0, 50, 75, 90, 100, float("inf")]
    labels = ["Under-loaded (<50%)", "Moderate (50-75%)",
              "Well-loaded (75-90%)", "Full (90-100%)", "Overloaded (>100%)"]
    df = df.copy()
    df["utilization_band"] = pd.cut(
        df["load_utilization_pct"], bins=bins, labels=labels, right=True
    )
    dist = df["utilization_band"].value_counts().reset_index()
    dist.columns = ["Band", "Count"]
    dist["Percentage"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)
    return dist


def monthly_cost_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregated cost metrics."""
    monthly = df.groupby("month").agg(
        trip_count=("actual_trip_cost_inr", "count"),
        avg_cost=("actual_trip_cost_inr", "mean"),
        total_cost=("actual_trip_cost_inr", "sum"),
        avg_distance=("distance_km", "mean"),
    ).round(2)
    return monthly
