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
NUMERIC_METRICS = ['avg_txn_value', 'daily_txn_count', 'refund_rate', 'repeat_customer_rate', 'income_level']
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

def cluster_merchants_by_industry(merchants_df, n_clusters=3, batch_size=1000):
    """
    Applies preprocessing and MiniBatchKMeans clustering to merchants within each industry separately.
    This provides more meaningful comparisons as businesses in the same industry have similar patterns.
    """
    if merchants_df.empty or not all(col in merchants_df.columns for col in ALL_METRICS):
        print("Warning: Missing required columns for clustering or empty DataFrame.")
        return merchants_df, None, None

    # Initialize cluster column
    merchants_df['cluster'] = -1
    
    # Get unique industries
    industries = merchants_df['industry'].unique()
    print(f"Clustering merchants within {len(industries)} industries: {industries}")
    
    models_dict = {}
    preprocessors_dict = {}
    
    for industry in industries:
        print(f"Clustering {industry} merchants...")
        
        # Filter merchants by industry
        industry_merchants = merchants_df[merchants_df['industry'] == industry].copy()
        
        if len(industry_merchants) < 2:
            print(f"Skipping {industry}: insufficient data (only {len(industry_merchants)} merchants)")
            continue
            
        # Adjust number of clusters based on data size
        industry_n_clusters = min(n_clusters, len(industry_merchants) // 2)
        if industry_n_clusters < 2:
            print(f"Skipping {industry}: insufficient merchants for clustering")
            continue
            
        # Separate features (excluding industry since all are the same)
        features = industry_merchants[ALL_METRICS].copy()

        # Handle potential NaNs
        for col in NUMERIC_METRICS:
            if features[col].isnull().any():
                features[col] = features[col].fillna(features[col].median())
        for col in CATEGORICAL_METRICS:
            if features[col].isnull().any():
                features[col] = features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else 'Unknown')

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
            # Fit preprocessor for this industry
            preprocessor.fit(features)
            
            # Process features
            processed_features = preprocessor.transform(features)
            
            # Initialize and fit MiniBatchKMeans for this industry
            kmeans = MiniBatchKMeans(
                n_clusters=industry_n_clusters,
                batch_size=min(batch_size, len(industry_merchants)),
                random_state=42,
                n_init=3,
                max_iter=100
            )
            
            # Fit and predict clusters for this industry
            industry_clusters = kmeans.fit_predict(processed_features)
            
            # Create industry-specific cluster IDs (e.g., "Retail_0", "Restaurant_1")
            industry_cluster_ids = [f"{industry}_{cluster}" for cluster in industry_clusters]
            
            # Update the main dataframe with cluster assignments
            merchants_df.loc[merchants_df['industry'] == industry, 'cluster'] = industry_cluster_ids
            
            # Store models and preprocessors for this industry
            models_dict[industry] = kmeans
            preprocessors_dict[industry] = preprocessor
            
            print(f"Successfully clustered {len(industry_merchants)} {industry} merchants into {industry_n_clusters} clusters")
            
        except Exception as e:
            print(f"Error clustering {industry} merchants: {e}")
            continue
    
    return merchants_df, models_dict, preprocessors_dict

# Keep the old function name for backward compatibility
def cluster_merchants(merchants_df, n_clusters=3, batch_size=1000):
    """Wrapper function that calls the new industry-specific clustering."""
    return cluster_merchants_by_industry(merchants_df, n_clusters, batch_size)

def get_comparison_data(merchant_id, merchants_df):
    """
    Performs industry-specific clustering and generates comparison dataframes.
    Returns:
        - merchant_row (dict): Profile of the selected merchant.
        - comparison_df_local (pd.DataFrame): Comparison vs local competitors (same pincode + industry).
        - comparison_df_cluster (pd.DataFrame): Comparison vs cluster avg (ML clustering within industry).
        - local_competitors (pd.DataFrame): Merchants in same pincode + industry.
        - cluster_peers (pd.DataFrame): Merchants in the same ML cluster.
        - cluster_averages (pd.Series): Average metrics for the merchant's cluster.
    """
    # --- 1. Get Selected Merchant's Profile ---
    merchant_row = get_merchant_profile(merchant_id, merchants_df)
    if merchant_row is None:
        return None, None, None, None, None, None
    
    merchant_pincode = merchant_row.get('pincode', None)
    merchant_industry = merchant_row.get('industry', None)

    # --- 2. Find Local Competitors (same pincode + same industry) ---
    local_competitors = pd.DataFrame()
    local_averages = None
    
    if merchant_pincode is not None and merchant_industry is not None:
        local_competitors = merchants_df[
            (merchants_df['pincode'] == merchant_pincode) &
            (merchants_df['industry'] == merchant_industry) &
            (merchants_df['merchant_id'] != merchant_id)  # Exclude self
        ].copy()
        
        if not local_competitors.empty:
            local_averages = local_competitors[NUMERIC_METRICS].mean()
            print(f"Found {len(local_competitors)} local competitors in {merchant_industry} industry, pincode {merchant_pincode}")
        else:
            print(f"No local competitors found for {merchant_industry} industry in pincode {merchant_pincode}")

    # --- 3. Cluster merchants by industry for ML-based clustering ---
    merchants_clustered_df, kmeans_models, preprocessors = cluster_merchants(merchants_df.copy())
    if kmeans_models is None or not kmeans_models:
        print("Clustering failed, cannot provide cluster comparison.")
        merchants_clustered_df['cluster'] = -1

    # --- 4. Update merchant row with cluster info ---
    merchant_row_clustered = get_merchant_profile(merchant_id, merchants_clustered_df)
    if merchant_row_clustered:
        merchant_cluster = merchant_row_clustered.get('cluster', -1)
        merchant_row['cluster'] = merchant_cluster  # Add cluster info to merchant_row
    else:
        merchant_cluster = -1

    # --- 5. Find Cluster Peers & Calculate Cluster Averages (same industry + same ML cluster) ---
    cluster_peers = pd.DataFrame()
    cluster_averages = None
    
    if merchant_cluster != -1 and merchant_industry is not None:
        cluster_peers = merchants_clustered_df[
            (merchants_clustered_df['cluster'] == merchant_cluster) &
            (merchants_clustered_df['industry'] == merchant_industry) &
            (merchants_clustered_df['merchant_id'] != merchant_id)  # Exclude self
        ].copy()
        
        if not cluster_peers.empty:
            cluster_averages = cluster_peers[NUMERIC_METRICS].mean()
            print(f"Found {len(cluster_peers)} cluster peers in {merchant_industry} industry, cluster {merchant_cluster}")
        else:
            print(f"No cluster peers found for {merchant_industry} industry, cluster {merchant_cluster}")

    # --- 6. Build Comparison DataFrames ---
    comparison_dfs = {'local': None, 'cluster': None}

    # Create both local and cluster comparisons
    for comp_type, avg_metrics_series in [('local', local_averages), ('cluster', cluster_averages)]:
        if avg_metrics_series is None:
            print(f"Skipping {comp_type} comparison - no data available")
            continue

        comparison = {
            'Metric': [], 'Merchant Value': [], f'{comp_type.capitalize()} Avg': [], 'Performance': [],
            'Merchant Raw': [], f'{comp_type.capitalize()} Raw': []
        }
        
        for metric in NUMERIC_METRICS:
            if metric not in merchant_row or pd.isna(merchant_row[metric]):
                continue

            merchant_value = merchant_row[metric]
            competitor_value = avg_metrics_series.get(metric, np.nan)

            if pd.isna(competitor_value):
                performance = 'N/A'
            elif metric in ['avg_txn_value', 'daily_txn_count', 'repeat_customer_rate', 'income_level']:
                performance = '✅ Above Avg' if merchant_value >= competitor_value else '❌ Below Avg'
            elif metric in ['refund_rate']:
                performance = '✅ Below Avg' if merchant_value <= competitor_value else '❌ Above Avg'
            else:
                performance = 'N/A'

            comparison['Metric'].append(metric.replace('_', ' ').title())
            
            # Store raw values for calculations
            comparison['Merchant Raw'].append(merchant_value)
            comparison[f'{comp_type.capitalize()} Raw'].append(competitor_value if not pd.isna(competitor_value) else 0)
            
            # Format display values based on metric type
            if metric in ['repeat_customer_rate', 'refund_rate']:
                display_merchant_value = f"{merchant_value*100:.1f}%"
                display_competitor_value = f"{competitor_value*100:.1f}%" if not pd.isna(competitor_value) else 'N/A'
            elif metric == 'avg_txn_value':
                display_merchant_value = f"₹{merchant_value:.2f}"
                display_competitor_value = f"₹{competitor_value:.2f}" if not pd.isna(competitor_value) else 'N/A'
            elif metric == 'income_level':
                display_merchant_value = f"₹{merchant_value:.2f}"
                display_competitor_value = f"₹{competitor_value:.2f}" if not pd.isna(competitor_value) else 'N/A'
            else:
                display_merchant_value = round(merchant_value, 2)
                display_competitor_value = round(competitor_value, 2) if not pd.isna(competitor_value) else 'N/A'
            
            comparison['Merchant Value'].append(display_merchant_value)
            comparison[f'{comp_type.capitalize()} Avg'].append(display_competitor_value)
            comparison['Performance'].append(performance)

        comparison_dfs[comp_type] = pd.DataFrame(comparison)

    return (
        merchant_row,
        comparison_dfs['local'],    # Local comparison (same pincode + industry)
        comparison_dfs['cluster'],  # Cluster comparison (ML clustering within industry)
        local_competitors,          # Local competitors DataFrame
        cluster_peers,              # Cluster peers DataFrame
        cluster_averages           # Cluster averages
    )