# src/fruit_classifier/pipelines/data_processing/download_dataset.py

import os
import zipfile
from pathlib import Path
import pandas as pd
from typing import Tuple

def get_existing_dataset_paths(data_dir: str = "data/01_raw") -> Tuple[str, str]:
    """
    Get paths to existing dataset directories.
    Since you already have the data, we'll just return the paths.
    
    Args:
        data_dir: Directory where the dataset is located
        
    Returns:
        Tuple of (train_path, test_path)
    """
    # Since your data is already in data/01_raw
    # We'll use the same directory for both train and test
    # You can modify this if you have separate train/test folders
    data_path = data_dir
    
    return data_path, data_path

def prepare_subset_dataframe(
    data_path: str, 
    selected_fruits: list = None,
    samples_per_class: int = 100,
    train_split: float = 0.8
) -> pd.DataFrame:
    """
    Prepare a DataFrame with image paths and labels for selected fruit classes.
    
    Args:
        data_path: Path to the dataset directory
        selected_fruits: List of fruit classes to include
        samples_per_class: Maximum number of samples per class
        train_split: Proportion of data to use for training (vs testing)
        
    Returns:
        DataFrame with 'image' and 'label' columns
    """
    
        # Updated to match your folder names exactly
    selected_fruits = ['Banana', 'Strawberry', 'Watermelon']
    
    data_rows = []
    label_map = {}
    
    for idx, fruit in enumerate(selected_fruits):
        fruit_dir = os.path.join(data_path, fruit)
        if os.path.exists(fruit_dir):
            label_map[fruit] = idx
            
            # Get all image files in the directory
            images = []
            for file in os.listdir(fruit_dir):
                if file.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    images.append(file)
            
            # Limit to samples_per_class if specified
            if samples_per_class and len(images) > samples_per_class:
                images = images[:samples_per_class]
            
            for img in images:
                data_rows.append({
                    'image': os.path.join(fruit_dir, img),
                    'label': idx,
                    'fruit_name': fruit
                })
        else:
            print(f"Warning: Directory not found: {fruit_dir}")
    
    df = pd.DataFrame(data_rows)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df, label_map