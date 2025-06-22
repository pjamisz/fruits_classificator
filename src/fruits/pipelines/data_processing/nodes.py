
# src/fruit_classifier/pipelines/data_processing/nodes.py

import pandas as pd
from typing import Dict, Tuple
import os
from pathlib import Path
from .download_dataset import get_existing_dataset_paths, prepare_subset_dataframe
from sklearn.model_selection import train_test_split

def download_and_prepare_data(params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Prepare train/test DataFrames from existing fruit dataset.
    
    Args:
        params: Dictionary containing configuration parameters
        
    Returns:
        Tuple of (train_df, test_df, label_map)
    """
    # Get paths to existing dataset
    data_path, _ = get_existing_dataset_paths(params.get("raw_data_path", "data/01_raw"))
    
    # Prepare DataFrame for selected fruit classes
    selected_fruits = params.get("selected_fruits", 
                                ['Banana', 'Strawberry', 'Watermelon'])
                             
    selected_fruits = ['Banana', 'Strawberry', 'Watermelon']
    samples_per_class = params.get("samples_per_class", None)  # Use all available samples
    
    # Get all data first
    all_df, label_map = prepare_subset_dataframe(
        data_path, 
        selected_fruits, 
        samples_per_class
    )
    
    # Check if we have data
    if len(all_df) == 0:
        raise ValueError(f"No data found in {data_path} for fruits: {selected_fruits}")
    
    print(f"Total samples found: {len(all_df)}")
    for fruit in selected_fruits:
        count = len(all_df[all_df['fruit_name'] == fruit])
        print(f"  {fruit}: {count} samples")
    
    # Split into train and test
    train_split = params.get("train_split", 0.8)
    
    # Keep only image and label columns
    all_df = all_df[['image', 'label']]
    
    # Split the data
    train_df, test_df = train_test_split(
        all_df, 
        test_size=1-train_split, 
        random_state=42, 
        stratify=all_df['label']  # Ensure balanced split
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df, label_map