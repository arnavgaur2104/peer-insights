# compare_merchants.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Define metrics for comparison and clustering ---
# Ensure these match columns in generate_data.py
NUMERIC_METRICS = ['avg_txn_value', 'daily_txn_count', 'refund_rate', 'rent_pct_revenue', 'foot_traffic', 'income_level', 'store_size_sqft']
CATEGORICAL_METRICS = ['store_type'] # Add more if needed
ALL_METRICS = NUMERIC_METRICS + CATEGORICAL_METRICS

def get_merchant_profile(merchant_id, merchants_df):
    """Fetch the merchant's basic details."""
    merchant = merchants_df[merchants_df['merchant_id'] == merchant_id]
    if merchant.empty:
        return None
    # Ensure all expected metrics are present, fill with NaN if not
    merchant_data = merchant.iloc[0].to_dict()
    for col in ALL_METRICS + ['pincode', 'industry', 'city']:
         if col not in merchant_data:
             merchant_data[col] = np.nan
    return merchant_data


def cluster_merchants(merchants_df, n_clusters=4):
    """Applies preprocessing and KMeans clustering to merchants."""
    if merchants_df.empty or not all(col in merchants_df.columns for col in ALL_METRICS):
         print("Warning: Missing required columns for clustering or empty DataFrame.")
         return merchants_df, None, None # Return original df, no cluster info

    # Separate features
    features = merchants_df[ALL_METRICS].copy()

    # Handle potential NaNs (e.g., fill with median/mode or drop rows)
    for col in NUMERIC_METRICS:
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())
    for col in CATEGORICAL_METRICS:
         if features[col].isnull().any():
             features[col] = features[col].fillna(features[col].mode()[0])


    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ignore categories not seen during fit
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_METRICS),
            ('cat', categorical_transformer, CATEGORICAL_METRICS)
        ],
        remainder='passthrough' # Keep other columns if any, though we selected specific ones
        )

    # Create the full pipeline including preprocessing and clustering
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clusterer', KMeans(n_clusters=n_clusters, random_state=42, n_init=10)) # Set n_init explicitly
    ])

    # Fit the pipeline and predict clusters
    try:
        clusters = pipeline.fit_predict(features)
        merchants_df['cluster'] = clusters
        # Store the fitted pipeline (including scaler, encoder, clusterer) if needed later
        # For now, just return the dataframe with clusters and cluster centers
        kmeans_model = pipeline.named_steps['clusterer']
        return merchants_df, kmeans_model, pipeline.named_steps['preprocessor']
    except Exception as e:
         print(f"Error during clustering: {e}")
         # Assign a default cluster or handle error appropriately
         merchants_df['cluster'] = -1 # Indicate clustering failed
         return merchants_df, None, None


def get_comparison_data(merchant_id, merchants_df, competitors_df):
    """
    Performs clustering and generates comparison dataframes.
    Returns:
        - merchant_row (dict): Profile of the selected merchant.
        - comparison_df (pd.DataFrame): Comparison vs local competitors avg.
        - cluster_comparison_df (pd.DataFrame): Comparison vs cluster avg.
        - local_competitors (pd.DataFrame): DataFrame of local competitors.
        - cluster_peers (pd.DataFrame): DataFrame of merchants in the same cluster.
        - cluster_averages (pd.Series): Average metrics for the merchant's cluster.
    """
    # --- 1. Cluster ALL merchants first ---
    merchants_clustered_df, kmeans_model, preprocessor = cluster_merchants(merchants_df.copy())
    if kmeans_model is None: # Handle clustering failure
         print("Clustering failed, cannot provide cluster comparison.")
         merchants_clustered_df['cluster'] = -1 # Ensure column exists


    # --- 2. Get Selected Merchant's Profile & Cluster ---
    merchant_row = get_merchant_profile(merchant_id, merchants_clustered_df)
    if merchant_row is None:
        return None, None, None, None, None, None
    merchant_cluster = merchant_row.get('cluster', -1) # Get cluster, default to -1 if missing


    # --- 3. Find Local Competitors ---
    local_competitors = competitors_df[
        (competitors_df['pincode'] == merchant_row['pincode']) &
        (competitors_df['industry'] == merchant_row['industry']) &
        (competitors_df['merchant_id'] != merchant_id) # Exclude self if present
    ].copy()


    # --- 4. Calculate Local Competitor Averages ---
    local_comp_avg = None
    if not local_competitors.empty:
        local_comp_avg = local_competitors[NUMERIC_METRICS].mean()


    # --- 5. Find Cluster Peers & Calculate Cluster Averages ---
    cluster_peers = pd.DataFrame()
    cluster_averages = None
    if merchant_cluster != -1: # Check if clustering was successful and merchant has a cluster
         cluster_peers = merchants_clustered_df[
             (merchants_clustered_df['cluster'] == merchant_cluster) &
             (merchants_clustered_df['merchant_id'] != merchant_id) # Exclude self
         ].copy()
         if not cluster_peers.empty:
             cluster_averages = cluster_peers[NUMERIC_METRICS].mean()


    # --- 6. Build Comparison DataFrames ---
    comparison_dfs = {'local': None, 'cluster': None}

    for comp_type, avg_metrics_series in [('local', local_comp_avg), ('cluster', cluster_averages)]:
        if avg_metrics_series is None:
            continue # Skip if no competitors/peers or averages couldn't be calculated

        comparison = {
            'Metric': [], 'Merchant Value': [], f'{comp_type.capitalize()} Avg': [], 'Performance': []
        }
        for metric in NUMERIC_METRICS:
             if metric not in merchant_row or pd.isna(merchant_row[metric]):
                  continue # Skip if merchant doesn't have this metric

             merchant_value = merchant_row[metric]
             competitor_value = avg_metrics_series.get(metric, np.nan) # Use .get for safety

             if pd.isna(competitor_value):
                  performance = 'N/A' # Cannot compare
             # Define which metrics are 'higher is better' vs 'lower is better'
             elif metric in ['avg_txn_value', 'daily_txn_count', 'foot_traffic', 'income_level', 'store_size_sqft']:
                 performance = '✅ Above Avg' if merchant_value >= competitor_value else '❌ Below Avg'
             elif metric in ['refund_rate', 'rent_pct_revenue']:
                 performance = '✅ Below Avg' if merchant_value <= competitor_value else '❌ Above Avg'
             else:
                 performance = 'N/A' # Metric type not defined for comparison

             comparison['Metric'].append(metric.replace('_', ' ').title())
             comparison['Merchant Value'].append(round(merchant_value, 2))
             comparison[f'{comp_type.capitalize()} Avg'].append(round(competitor_value, 2) if not pd.isna(competitor_value) else 'N/A')
             comparison['Performance'].append(performance)

        comparison_dfs[comp_type] = pd.DataFrame(comparison)

    return (
        merchant_row,
        comparison_dfs['local'],
        comparison_dfs['cluster'],
        local_competitors,
        cluster_peers,
        cluster_averages # Pass cluster averages for potential use in insights
    )