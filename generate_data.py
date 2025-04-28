# generate_data.py

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
store_types = ['Mall', 'Street Front', 'Standalone']
industries = ['Retail', 'Restaurant', 'Fashion']

def generate_merchants(n_merchants=100):
    merchants = []
    for i in range(n_merchants): # Use index for unique ID
        city = random.choice(cities)
        pincode = random.randint(400001, 400100)
        store_type = random.choice(store_types)
        industry = random.choice(industries)
        income_level = abs(np.random.normal(50000, 15000)) # Ensure non-negative
        foot_traffic = abs(np.random.normal(200, 70))     # Ensure non-negative
        rent_pct_revenue = np.random.uniform(0.05, 0.25)
        weather_preference = random.choice(['Rain Boost', 'Sunny Boost', 'None'])

        # --- NEW: Add Store Size ---
        if store_type == 'Mall':
            store_size_sqft = np.random.randint(800, 5000)
        elif store_type == 'Street Front':
            store_size_sqft = np.random.randint(400, 2500)
        else: # Standalone
            store_size_sqft = np.random.randint(1000, 8000)
        # --- END NEW ---

        # Industry-specific metrics
        if industry == 'Retail':
            avg_txn_value = np.random.uniform(100, 500)
            daily_txn_count = np.random.randint(30, 100)
            refund_rate = np.random.uniform(0.01, 0.1)
        elif industry == 'Restaurant':
            avg_txn_value = np.random.uniform(300, 1000)
            daily_txn_count = np.random.randint(20, 80)
            refund_rate = np.random.uniform(0.01, 0.05)
        else:  # Fashion
            avg_txn_value = np.random.uniform(800, 3000)
            daily_txn_count = np.random.randint(10, 40)
            refund_rate = np.random.uniform(0.02, 0.15)

        merchants.append({
            # --- MODIFIED: Use index for predictable ID ---
            'merchant_id': f'M{1000+i}',
            # --- END MODIFIED ---
            'city': city,
            'pincode': pincode,
            'store_type': store_type,
            'industry': industry,
            'income_level': round(income_level, 2),
            'foot_traffic': round(foot_traffic, 2),
            'rent_pct_revenue': round(rent_pct_revenue, 4),
            'weather_preference': weather_preference,
            'avg_txn_value': round(avg_txn_value, 2),
            'daily_txn_count': daily_txn_count,
            'refund_rate': round(refund_rate, 4),
            # --- NEW ---
            'store_size_sqft': store_size_sqft
             # --- END NEW ---
        })

    df = pd.DataFrame(merchants)
    # Ensure data directory exists
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/merchants.csv', index=False)
    print("Merchants data generated and saved.")
    return df

def generate_competitors(merchants_df):
    competitors = []
    comp_counter = 10000
    for idx, row in merchants_df.iterrows():
        n_competitors = np.random.randint(3, 8)
        for _ in range(n_competitors):
            comp = row.copy()
            # --- MODIFIED: Use counter for unique ID ---
            comp['merchant_id'] = f'C{comp_counter}'
            comp_counter += 1
            # --- END MODIFIED ---
            # Slightly vary competitor metrics
            comp['avg_txn_value'] *= np.random.uniform(0.9, 1.1)
            comp['daily_txn_count'] *= np.random.uniform(0.8, 1.2)
            comp['refund_rate'] *= np.random.uniform(0.8, 1.2)
            comp['income_level'] *= np.random.uniform(0.95, 1.05)
            comp['foot_traffic'] *= np.random.uniform(0.9, 1.1)
            comp['rent_pct_revenue'] *= np.random.uniform(0.9, 1.1)
            comp['store_size_sqft'] *= np.random.uniform(0.85, 1.15)

            # Ensure non-negative values and round
            comp['avg_txn_value'] = round(max(0, comp['avg_txn_value']), 2)
            comp['daily_txn_count'] = max(0, int(comp['daily_txn_count']))
            comp['refund_rate'] = round(max(0, min(1, comp['refund_rate'])), 4)
            comp['income_level'] = round(max(0, comp['income_level']), 2)
            comp['foot_traffic'] = round(max(0, comp['foot_traffic']), 2)
            comp['rent_pct_revenue'] = round(max(0, comp['rent_pct_revenue']), 4)
            comp['store_size_sqft'] = max(100, int(comp['store_size_sqft'])) # Min size 100

            competitors.append(comp)

    comp_df = pd.DataFrame(competitors)
    # Ensure data directory exists
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    comp_df.to_csv('data/competitors.csv', index=False)
    print("Competitors data generated and saved.")
    return comp_df

if __name__ == "__main__":
    # Ensure the data directory exists before generating
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    merchants_df = generate_merchants()
    competitors_df = generate_competitors(merchants_df)