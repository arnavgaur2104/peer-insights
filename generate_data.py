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
            repeat_customer_rate = np.random.uniform(0.25, 0.60)  # 25-60% for retail
        elif industry == 'Restaurant':
            avg_txn_value = np.random.uniform(300, 1000)
            daily_txn_count = np.random.randint(20, 80)
            refund_rate = np.random.uniform(0.01, 0.05)
            repeat_customer_rate = np.random.uniform(0.35, 0.75)  # 35-75% for restaurants
        else:  # Fashion
            avg_txn_value = np.random.uniform(800, 3000)
            daily_txn_count = np.random.randint(10, 40)
            refund_rate = np.random.uniform(0.02, 0.15)
            repeat_customer_rate = np.random.uniform(0.15, 0.45)  # 15-45% for fashion

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
            'refund_rate': round(refund_rate, 4),
            'repeat_customer_rate': round(repeat_customer_rate, 4)
        })
    return pd.DataFrame(merchants)

def generate_data(n_merchants=100000):
    """Generate merchant data in chunks."""
    print(f"Generating {n_merchants} merchants...")
    
    # Initialize file path
    merchants_file = DATA_DIR / 'merchants.csv'
    
    # Clear existing file
    if merchants_file.exists():
        merchants_file.unlink()
    
    # Generate data in chunks
    for start_idx in tqdm(range(0, n_merchants, CHUNK_SIZE)):
        # Calculate chunk size
        chunk_size = min(CHUNK_SIZE, n_merchants - start_idx)
        
        # Generate merchant chunk
        merchant_chunk = generate_merchant_chunk(start_idx, chunk_size)
        
        # Save chunk to CSV
        merchant_chunk.to_csv(merchants_file, mode='a', header=not merchants_file.exists(), index=False)
        
        # Clear memory
        del merchant_chunk
    
    print("Data generation complete!")
    print(f"Merchants saved to: {merchants_file}")
    print(f"Total merchants generated: {n_merchants}")

if __name__ == "__main__":
    generate_data()