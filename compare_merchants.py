# compare_merchants.py

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import multiprocessing

# --- Define metrics for comparison and clustering ---
# Ensure these match columns in generate_data.py
NUMERIC_METRICS = ['avg_txn_value', 'daily_txn_count', 'refund_rate', 'income_level']
CATEGORICAL_METRICS = ['store_type'] # Add more if needed
ALL_METRICS = NUMERIC_METRICS + CATEGORICAL_METRICS

# Get number of CPU cores for parallel processing
N_CORES = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free

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

def preprocess_chunk(chunk, preprocessor):
    """Preprocess a chunk of data in parallel."""
    return preprocessor.transform(chunk)

def cluster_merchants(merchants_df, n_clusters=4, batch_size=1000):
    """Applies preprocessing and MiniBatchKMeans clustering to merchants with parallel processing."""
    if merchants_df.empty or not all(col in merchants_df.columns for col in ALL_METRICS):
         print("Warning: Missing required columns for clustering or empty DataFrame.")
         return merchants_df, None, None

    # Separate features
    features = merchants_df[ALL_METRICS].copy()

    # Handle potential NaNs
    for col in NUMERIC_METRICS:
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())
    for col in CATEGORICAL_METRICS:
         if features[col].isnull().any():
             features[col] = features[col].fillna(features[col].mode()[0])

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_METRICS),
            ('cat', categorical_transformer, CATEGORICAL_METRICS)
        ],
        remainder='passthrough'
    )

    try:
        # Fit preprocessor
        preprocessor.fit(features)
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(features) // N_CORES)
        chunks = [features[i:i + chunk_size] for i in range(0, len(features), chunk_size)]
        
        # Process chunks in parallel
        processed_chunks = Parallel(n_jobs=N_CORES)(
            delayed(preprocess_chunk)(chunk, preprocessor) for chunk in chunks
        )
        
        # Combine processed chunks
        processed_features = np.vstack(processed_chunks)
        
        # Initialize and fit MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init=3,  # Reduced from 10 since we're using mini-batches
            max_iter=100
        )
        
        # Fit and predict clusters
        clusters = kmeans.fit_predict(processed_features)
        merchants_df['cluster'] = clusters
        
        return merchants_df, kmeans, preprocessor
        
    except Exception as e:
        print(f"Error during clustering: {e}")
        merchants_df['cluster'] = -1
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
             elif metric in ['avg_txn_value', 'daily_txn_count', 'income_level']:
                 performance = '✅ Above Avg' if merchant_value >= competitor_value else '❌ Below Avg'
             elif metric in ['refund_rate']:
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