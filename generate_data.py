# generate_data.py

import pandas as pd
import numpy as np
import random
from pathlib import Path
import os
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Constants
CHUNK_SIZE = 10000  # Number of merchants to generate at once
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai']
store_types = ['Mall', 'Street Front', 'Standalone']
industries = ['Retail', 'Restaurant', 'Fashion']

def generate_merchant_chunk(start_idx, n_merchants):
    """Generate a chunk of merchant data."""
    merchants = []
    for i in range(n_merchants):
        idx = start_idx + i
        city = random.choice(cities)
        pincode = random.randint(400001, 400100)
        store_type = random.choice(store_types)
        industry = random.choice(industries)
        income_level = abs(np.random.normal(50000, 15000))
        weather_preference = random.choice(['Rain Boost', 'Sunny Boost', 'None'])

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
            'merchant_id': f'M{1000+idx}',
            'city': city,
            'pincode': pincode,
            'store_type': store_type,
            'industry': industry,
            'income_level': round(income_level, 2),
            'weather_preference': weather_preference,
            'avg_txn_value': round(avg_txn_value, 2),
            'daily_txn_count': daily_txn_count,
            'refund_rate': round(refund_rate, 4)
        })
    return pd.DataFrame(merchants)

def generate_competitor_chunk(merchant_chunk, comp_counter):
    """Generate competitors for a chunk of merchants."""
    competitors = []
    for _, row in merchant_chunk.iterrows():
        n_competitors = np.random.randint(3, 8)
        for _ in range(n_competitors):
            comp = row.copy()
            comp['merchant_id'] = f'C{comp_counter}'
            comp_counter += 1
            
            # Vary competitor metrics
            comp['avg_txn_value'] *= np.random.uniform(0.9, 1.1)
            comp['daily_txn_count'] *= np.random.uniform(0.8, 1.2)
            comp['refund_rate'] *= np.random.uniform(0.8, 1.2)
            comp['income_level'] *= np.random.uniform(0.95, 1.05)

            # Ensure non-negative values and round
            comp['avg_txn_value'] = round(max(0, comp['avg_txn_value']), 2)
            comp['daily_txn_count'] = max(0, int(comp['daily_txn_count']))
            comp['refund_rate'] = round(max(0, min(1, comp['refund_rate'])), 4)
            comp['income_level'] = round(max(0, comp['income_level']), 2)

            competitors.append(comp)
    return pd.DataFrame(competitors), comp_counter

def generate_data(n_merchants=100000):
    """Generate merchant and competitor data in chunks."""
    print(f"Generating {n_merchants} merchants and their competitors...")
    
    # Initialize counters and file paths
    comp_counter = 10000
    merchants_file = DATA_DIR / 'merchants.csv'
    competitors_file = DATA_DIR / 'competitors.csv'
    
    # Clear existing files
    for file in [merchants_file, competitors_file]:
        if file.exists():
            file.unlink()
    
    # Generate data in chunks
    for start_idx in tqdm(range(0, n_merchants, CHUNK_SIZE)):
        # Calculate chunk size
        chunk_size = min(CHUNK_SIZE, n_merchants - start_idx)
        
        # Generate merchant chunk
        merchant_chunk = generate_merchant_chunk(start_idx, chunk_size)
        
        # Generate competitor chunk
        competitor_chunk, comp_counter = generate_competitor_chunk(merchant_chunk, comp_counter)
        
        # Save chunks to CSV
        merchant_chunk.to_csv(merchants_file, mode='a', header=not merchants_file.exists(), index=False)
        competitor_chunk.to_csv(competitors_file, mode='a', header=not competitors_file.exists(), index=False)
        
        # Clear memory
        del merchant_chunk
        del competitor_chunk
    
    print("Data generation complete!")
    print(f"Merchants saved to: {merchants_file}")
    print(f"Competitors saved to: {competitors_file}")

if __name__ == "__main__":
    generate_data()