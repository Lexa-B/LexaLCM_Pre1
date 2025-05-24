import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import argparse

def add_train_val_split(parquet_dir: str | Path, validation_ratio: float = 0.1, seed: int = 42):
    """
    Add a train/validation split to the existing parquet file.
    
    Args:
        parquet_dir: Directory containing the parquet files (can be string or Path)
        validation_ratio: Ratio of data to use for validation (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    """
    # Convert string to Path if needed
    parquet_dir = Path(parquet_dir)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Get all parquet files in the directory
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file}")
        
        # Read the parquet file
        table = pq.read_table(parquet_file)
        
        # Check if split column already exists
        if "split" in table.column_names:
            print(f"Split column already exists in {parquet_file.name}, skipping...")
            continue
        
        # Get number of rows
        num_rows = len(table)
        
        # Generate random split assignments
        split_assignments = np.random.choice(
            ["train", "validation"],
            size=num_rows,
            p=[1 - validation_ratio, validation_ratio]
        )
        
        # Create split column
        split_array = pa.array(split_assignments)
        
        # Add split column to table
        table = table.append_column("split", split_array)
        
        # Write back to the same parquet file
        pq.write_table(table, parquet_file)
        
        # Print statistics
        train_count = sum(split_assignments == "train")
        val_count = sum(split_assignments == "validation")
        print(f"Split statistics for {parquet_file.name}:")
        print(f"  Total rows: {num_rows}")
        print(f"  Train rows: {train_count} ({train_count/num_rows:.1%})")
        print(f"  Validation rows: {val_count} ({val_count/num_rows:.1%})")
        print(f"  Updated file: {parquet_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add train/validation split to parquet files")
    parser.add_argument("-d", "--parquet_dir", type=str, required=True, help="Directory containing the parquet files")
    parser.add_argument("-v", "--validation_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    add_train_val_split(parquet_dir=args.parquet_dir, validation_ratio=args.validation_ratio, seed=args.seed)