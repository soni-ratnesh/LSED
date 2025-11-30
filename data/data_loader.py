"""
Data Loader Module for LSED
Handles loading CSV data and splitting into offline/online message blocks.

Based on: "Text is All You Need: LLM-enhanced Incremental Social Event Detection"
ACL 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MessageBlock:
    """Represents a message block in the social event detection pipeline."""
    block_id: int
    block_name: str  # e.g., "M0", "M1", etc.
    data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    num_messages: int
    num_events: int
    is_offline: bool
    
    def __repr__(self):
        return (f"MessageBlock({self.block_name}, "
                f"messages={self.num_messages}, "
                f"events={self.num_events}, "
                f"period={self.start_date.date()} to {self.end_date.date()})")


class DataLoader:
    """
    Data loader for Social Event Detection.
    
    Handles:
    - Loading CSV data with text, timestamp, and event columns
    - Splitting data into offline (M0) and online (M1, M2, ..., Mn) blocks
    - Data preprocessing and validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.data_config = config.get('data', {})
        
        # Column names
        self.text_column = self.data_config.get('text_column', 'text')
        self.timestamp_column = self.data_config.get('timestamp_column', 'created_at')
        self.event_column = self.data_config.get('event_column', 'event')
        
        # Split settings
        self.offline_days = self.data_config.get('offline_days', 7)
        self.train_ratio = self.data_config.get('train_ratio', 0.7)
        self.val_ratio = self.data_config.get('val_ratio', 0.1)
        self.test_ratio = self.data_config.get('test_ratio', 0.2)
        
        # Storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.offline_block: Optional[MessageBlock] = None
        self.online_blocks: List[MessageBlock] = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = [self.text_column, self.timestamp_column, self.event_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Parse timestamps
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(by=self.timestamp_column).reset_index(drop=True)
        
        # Remove rows with missing values
        initial_len = len(df)
        df = df.dropna(subset=required_columns)
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with missing values")
        
        # Store raw data
        self.raw_data = df
        
        logger.info(f"Loaded {len(df)} messages with {df[self.event_column].nunique()} unique events")
        logger.info(f"Date range: {df[self.timestamp_column].min()} to {df[self.timestamp_column].max()}")
        
        return df
    
    def split_into_blocks(self, df: Optional[pd.DataFrame] = None) -> Tuple[MessageBlock, List[MessageBlock]]:
        """
        Split data into offline (M0) and online (M1, M2, ..., Mn) message blocks.
        
        The first `offline_days` days form the offline block (M0).
        Remaining days are split into daily online blocks (M1, M2, ..., Mn).
        
        Args:
            df: DataFrame to split (uses self.raw_data if None)
            
        Returns:
            Tuple of (offline_block, list of online_blocks)
        """
        if df is None:
            df = self.raw_data
        
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info(f"Splitting data into message blocks (offline_days={self.offline_days})")
        
        # Get date range
        min_date = df[self.timestamp_column].min()
        max_date = df[self.timestamp_column].max()
        
        # Calculate offline period end date
        offline_end = min_date + timedelta(days=self.offline_days)
        
        # Split into offline and online data
        offline_mask = df[self.timestamp_column] < offline_end
        offline_data = df[offline_mask].copy()
        online_data = df[~offline_mask].copy()
        
        # Create offline block (M0)
        self.offline_block = MessageBlock(
            block_id=0,
            block_name="M0",
            data=offline_data,
            start_date=offline_data[self.timestamp_column].min(),
            end_date=offline_data[self.timestamp_column].max(),
            num_messages=len(offline_data),
            num_events=offline_data[self.event_column].nunique(),
            is_offline=True
        )
        
        logger.info(f"Offline block: {self.offline_block}")
        
        # Split online data into daily blocks
        self.online_blocks = []
        
        if len(online_data) > 0:
            # Group by date
            online_data['date'] = online_data[self.timestamp_column].dt.date
            unique_dates = sorted(online_data['date'].unique())
            
            for i, date in enumerate(unique_dates, start=1):
                block_data = online_data[online_data['date'] == date].copy()
                block_data = block_data.drop(columns=['date'])
                
                block = MessageBlock(
                    block_id=i,
                    block_name=f"M{i}",
                    data=block_data,
                    start_date=block_data[self.timestamp_column].min(),
                    end_date=block_data[self.timestamp_column].max(),
                    num_messages=len(block_data),
                    num_events=block_data[self.event_column].nunique(),
                    is_offline=False
                )
                self.online_blocks.append(block)
                
            logger.info(f"Created {len(self.online_blocks)} online blocks")
        else:
            logger.warning("No online data available after splitting")
        
        return self.offline_block, self.online_blocks
    
    def get_train_val_test_split(
        self, 
        block: MessageBlock
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a message block into train, validation, and test sets.
        
        Args:
            block: MessageBlock to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data = block.data.copy()
        n = len(data)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # Split data
        train_df = data.iloc[:train_end]
        val_df = data.iloc[train_end:val_end]
        test_df = data.iloc[val_end:]
        
        logger.debug(f"Split {block.block_name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_all_blocks(self) -> List[MessageBlock]:
        """Get all message blocks (offline + online)."""
        blocks = []
        if self.offline_block is not None:
            blocks.append(self.offline_block)
        blocks.extend(self.online_blocks)
        return blocks
    
    def get_block_by_name(self, name: str) -> Optional[MessageBlock]:
        """Get a specific message block by name (e.g., 'M0', 'M1')."""
        for block in self.get_all_blocks():
            if block.block_name == name:
                return block
        return None
    
    def print_statistics(self):
        """Print statistics about the loaded data and blocks."""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        if self.raw_data is not None:
            print(f"\nTotal Messages: {len(self.raw_data)}")
            print(f"Total Events: {self.raw_data[self.event_column].nunique()}")
            print(f"Date Range: {self.raw_data[self.timestamp_column].min()} to {self.raw_data[self.timestamp_column].max()}")
        
        print("\nMessage Blocks:")
        print("-"*60)
        print(f"{'Block':<8} {'Messages':>10} {'Events':>8} {'Period':<30}")
        print("-"*60)
        
        for block in self.get_all_blocks():
            period = f"{block.start_date.date()} to {block.end_date.date()}"
            print(f"{block.block_name:<8} {block.num_messages:>10} {block.num_events:>8} {period:<30}")
        
        print("="*60 + "\n")


def create_sample_data(output_path: str, n_messages: int = 1000, n_events: int = 50):
    """
    Create sample data for testing.
    
    Args:
        output_path: Path to save the sample CSV
        n_messages: Number of messages to generate
        n_events: Number of unique events
    """
    np.random.seed(42)
    
    # Generate sample data
    base_date = datetime(2024, 1, 1)
    
    data = {
        'text': [f"Sample message {i} about event topic" for i in range(n_messages)],
        'created_at': [base_date + timedelta(hours=np.random.randint(0, 24*30)) for _ in range(n_messages)],
        'event': np.random.randint(0, n_events, n_messages)
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('created_at').reset_index(drop=True)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Created sample data at {output_path}")
    print(f"Messages: {n_messages}, Events: {n_events}")


if __name__ == "__main__":
    # Test the data loader
    import yaml
    
    # Create sample data
    sample_path = "data/sample_data.csv"
    create_sample_data(sample_path)
    
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Test data loader
    loader = DataLoader(config)
    loader.load_data(sample_path)
    loader.split_into_blocks()
    loader.print_statistics()
