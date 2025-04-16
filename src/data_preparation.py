import os
import shutil
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile
from tqdm import tqdm


def download_dataset():
    """Check for dataset files."""

    # Create data directories if they don't exist
    os.makedirs('../data/raw', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)

    print("Using manually downloaded dataset.")

    # Verify the files are found
    if not os.path.exists('../data/raw/HAM10000_metadata.csv'):
        print("Error: HAM10000_metadata.csv not found in data/raw directory.")
        sys.exit(1)

    if not os.path.exists('../data/raw/HAM10000_images_part_1'):
        print("Error: HAM10000_images_part_1 folder not found in data/raw directory.")
        sys.exit(1)

    if not os.path.exists('../data/raw/HAM10000_images_part_2'):
        print("Error: HAM10000_images_part_2 folder not found in data/raw directory.")
        sys.exit(1)

    print("All required files found. Ready to prepare dataset.")


def prepare_dataset():
    """Prepare the dataset for training."""

    # Read metadata - use HAM10000_metadata.csv which is the main metadata file
    df = pd.read_csv('../data/raw/HAM10000_metadata.csv')

    # Include all seven skin conditions
    conditions = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

    # Create processed data directories
    for condition in conditions:
        os.makedirs(f'../data/processed/train/{condition}', exist_ok=True)
        os.makedirs(f'../data/processed/val/{condition}', exist_ok=True)
        os.makedirs(f'../data/processed/test/{condition}', exist_ok=True)

    # Split data: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx'], random_state=42)

    # Copy images to their respective directories
    # Images are in two folders: HAM10000_images_part_1 and HAM10000_images_part_2

    # Helper function to copy images
    def copy_images(dataframe, split_name):
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Copying {split_name} images"):
            image_id = row['image_id']
            condition = row['dx']

            # Check part_1 first
            src_path = os.path.join('../data/raw/HAM10000_images_part_1', f"{image_id}.jpg")

            # If not in part_1, check part_2
            if not os.path.exists(src_path):
                src_path = os.path.join('../data/raw/HAM10000_images_part_2', f"{image_id}.jpg")

            dst_path = f"../data/processed/{split_name}/{condition}/{image_id}.jpg"

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Could not find image {image_id}.jpg")

    # Copy images to train, validation, and test directories
    copy_images(train_df, "train")
    copy_images(val_df, "val")
    copy_images(test_df, "test")

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images: {len(test_df)}")

    # Class distribution
    print("\nClass Distribution:")
    for condition in conditions:
        print(f"{condition}: {len(df[df['dx'] == condition])}")


if __name__ == "__main__":
    download_dataset()
    prepare_dataset()
