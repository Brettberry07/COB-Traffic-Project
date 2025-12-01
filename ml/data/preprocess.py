"""
Data Preprocessing Module - Handle missing values and extract features.

This module processes ingested data to:
- Impute missing values (* markers)
- Extract temporal features (hour, day, weekend)
- Extract rolling volume statistics
- Extract phase timing features
- Prepare data for model training
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


class VolumePreprocessor:
    """Preprocessor for volume (turning movement count) data."""
    
    def __init__(
        self,
        forward_fill_window: int = 4,  # 4 intervals = 1 hour
        rolling_windows: List[int] = None
    ):
        """
        Initialize the volume preprocessor.
        
        Args:
            forward_fill_window: Number of intervals for forward fill imputation
            rolling_windows: List of rolling window sizes for features (default: [4, 8, 16])
        """
        self.forward_fill_window = forward_fill_window
        self.rolling_windows = rolling_windows or [4, 8, 16]  # 1hr, 2hr, 4hr
    
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse DATE and TIME columns into proper datetime.
        
        Args:
            df: DataFrame with DATE and TIME columns
            
        Returns:
            DataFrame with datetime column added
        """
        df = df.copy()
        
        # Ensure TIME is 4-digit string
        df['TIME'] = df['TIME'].fillna('0000').astype(str).str.zfill(4)
        df['DATE'] = df['DATE'].fillna('').astype(str)
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(
            df['DATE'] + ' ' + df['TIME'],
            format='%m/%d/%Y %H%M',
            errors='coerce'
        )
        
        return df
    
    def impute_missing_values(
        self,
        df: pd.DataFrame,
        movement_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Impute missing values in movement columns.
        
        Strategy:
        1. Forward fill within same hour where possible
        2. Fallback to median of similar intervals (same time of day)
        3. Last resort: use 0
        
        Args:
            df: DataFrame with movement columns
            movement_cols: List of movement column names
            
        Returns:
            Tuple of (imputed DataFrame, dict of imputation counts per column)
        """
        df = df.copy()
        imputation_counts = {}
        
        # Ensure datetime is parsed
        if 'datetime' not in df.columns:
            df = self.parse_datetime(df)
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Extract hour for similar-interval matching
        df['hour'] = df['datetime'].dt.hour
        
        for col in movement_cols:
            if col not in df.columns:
                continue
            
            original_na = df[col].isna().sum()
            
            if original_na == 0:
                imputation_counts[col] = 0
                continue
            
            # Strategy 1: Forward fill within window
            df[col] = df[col].ffill(limit=self.forward_fill_window)
            
            # Strategy 2: Fill with median of same hour
            after_ffill_na = df[col].isna().sum()
            if after_ffill_na > 0:
                hour_medians = df.groupby('hour')[col].transform('median')
                df[col] = df[col].fillna(hour_medians)
            
            # Strategy 3: Fill with global median
            after_hour_na = df[col].isna().sum()
            if after_hour_na > 0:
                global_median = df[col].median()
                df[col] = df[col].fillna(global_median if pd.notna(global_median) else 0)
            
            # Strategy 4: Fill remaining with 0
            df[col] = df[col].fillna(0)
            
            imputation_counts[col] = original_na
        
        # Clean up
        if 'hour' in df.columns:
            df = df.drop(columns=['hour'])
        
        return df, imputation_counts
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from datetime.
        
        Features:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - is_weekend: Boolean for Saturday/Sunday
        - minute_of_day: Minutes since midnight
        - interval_of_day: 15-minute interval index (0-95)
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        if 'datetime' not in df.columns:
            df = self.parse_datetime(df)
        
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        df['interval_of_day'] = df['minute_of_day'] // 15
        
        return df
    
    def extract_rolling_features(
        self,
        df: pd.DataFrame,
        movement_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract rolling volume statistics.
        
        Features for each movement column:
        - Rolling mean over each window
        - Rolling std over each window
        
        Args:
            df: DataFrame with movement columns
            movement_cols: List of movement column names
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        # Sort by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        for col in movement_cols:
            if col not in df.columns:
                continue
            
            for window in self.rolling_windows:
                # Rolling mean
                df[f'{col}_mean_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                # Rolling std
                df[f'{col}_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)
        
        # Calculate total volume
        if movement_cols:
            available = [c for c in movement_cols if c in df.columns]
            df['total_volume'] = df[available].sum(axis=1)
            
            for window in self.rolling_windows:
                df[f'total_volume_mean_{window}'] = df['total_volume'].rolling(
                    window=window, min_periods=1
                ).mean()
        
        return df
    
    def process(
        self,
        df: pd.DataFrame,
        movement_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run full preprocessing pipeline on volume data.
        
        Args:
            df: Raw DataFrame from ingestion
            movement_cols: List of movement column names
            
        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        metadata = {
            'original_rows': len(df),
            'imputation_counts': {},
            'features_added': []
        }
        
        # Parse datetime
        df = self.parse_datetime(df)
        
        # Remove rows with invalid datetime
        valid_mask = df['datetime'].notna()
        df = df[valid_mask].copy()
        metadata['valid_rows'] = len(df)
        
        # Impute missing values
        df, imputation_counts = self.impute_missing_values(df, movement_cols)
        metadata['imputation_counts'] = imputation_counts
        
        # Extract temporal features
        df = self.extract_temporal_features(df)
        metadata['features_added'].extend(['hour', 'day_of_week', 'is_weekend', 
                                           'minute_of_day', 'interval_of_day'])
        
        # Extract rolling features
        df = self.extract_rolling_features(df, movement_cols)
        rolling_features = [c for c in df.columns if 'mean_' in c or 'std_' in c]
        metadata['features_added'].extend(rolling_features)
        
        return df, metadata


class TimingPreprocessor:
    """Preprocessor for phase timing data."""
    
    def __init__(self):
        """Initialize timing preprocessor."""
        pass
    
    def extract_timing_features(
        self,
        timing_data: Dict[str, Any],
        plan_number: int = 25
    ) -> Dict[str, Any]:
        """
        Extract features from phase timing data.
        
        Features:
        - Cycle length (computed from phases)
        - Green split ratios for each phase
        - Average yellow time
        - Average red clearance
        
        Args:
            timing_data: Dict from ingest_phase_timing_file
            plan_number: Plan number to use
            
        Returns:
            Dict with timing features
        """
        plans = timing_data.get('plans', {})
        yellow_times = timing_data.get('yellow_times', {})
        red_clearance = timing_data.get('red_clearance', {})
        
        # Get the specified plan or fallback to first available
        if plan_number in plans:
            plan = plans[plan_number]
        elif plans:
            plan_number = list(plans.keys())[0]
            plan = plans[plan_number]
        else:
            return {
                'error': 'No plans available',
                'plan_used': None
            }
        
        phase_greens = plan.get('phase_greens', {})
        offset = plan.get('offset')
        
        # Calculate total green time
        total_green = sum(phase_greens.values())
        
        # Average yellow and red clearance
        avg_yellow = np.mean(list(yellow_times.values())) if yellow_times else 4.0
        avg_red = np.mean(list(red_clearance.values())) if red_clearance else 1.0
        
        # Estimate cycle length (8-phase dual ring assumption)
        # For dual-ring, cycle = max(ring1, ring2) + clearances
        greens = list(phase_greens.values())
        num_phases = len(greens)
        
        if num_phases >= 8:
            ring1 = sum(greens[:4])
            ring2 = sum(greens[4:8])
            cycle_length = max(ring1, ring2) + 4 * (avg_yellow + avg_red)
        elif num_phases >= 4:
            cycle_length = sum(greens[:4]) + 2 * (avg_yellow + avg_red)
        else:
            cycle_length = sum(greens) + 2 * (avg_yellow + avg_red)
        
        # Compute green split ratios
        green_splits = {}
        if cycle_length > 0:
            for phase, green in phase_greens.items():
                green_splits[phase] = green / cycle_length
        
        return {
            'plan_used': plan_number,
            'cycle_length': cycle_length,
            'phase_greens': phase_greens,
            'green_splits': green_splits,
            'yellow_times': yellow_times,
            'red_clearance': red_clearance,
            'offset': offset,
            'avg_yellow': avg_yellow,
            'avg_red_clearance': avg_red,
            'total_green': total_green,
            'num_phases': num_phases
        }
    
    def get_all_plan_features(
        self,
        timing_data: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract features for all plans in timing data.
        
        Args:
            timing_data: Dict from ingest_phase_timing_file
            
        Returns:
            Dict mapping plan numbers to their features
        """
        plans = timing_data.get('plans', {})
        results = {}
        
        for plan_num in plans.keys():
            results[plan_num] = self.extract_timing_features(timing_data, plan_num)
        
        return results


def preprocess_intersection(
    volume_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    plan_number: int = 25
) -> Dict[str, Any]:
    """
    Preprocess data for a single intersection.
    
    Args:
        volume_data: Dict from ingest_volume_file
        timing_data: Dict from ingest_phase_timing_file
        plan_number: Signal timing plan to use
        
    Returns:
        Dict with preprocessed data and features
    """
    intersection_id = volume_data.get('intersection_id')
    raw_df = volume_data.get('raw_df')
    
    if raw_df is None:
        return {
            'intersection_id': intersection_id,
            'error': 'No volume data available',
            'volume_df': None,
            'timing_features': None
        }
    
    # Get available movement columns
    movement_cols = volume_data['metadata'].get('available_movements', [])
    
    # Process volume data
    volume_processor = VolumePreprocessor()
    processed_df, vol_metadata = volume_processor.process(raw_df, movement_cols)
    
    # Process timing data
    timing_processor = TimingPreprocessor()
    timing_features = timing_processor.extract_timing_features(timing_data, plan_number)
    
    return {
        'intersection_id': intersection_id,
        'volume_df': processed_df,
        'volume_metadata': vol_metadata,
        'timing_features': timing_features,
        'movement_cols': movement_cols
    }


def preprocess_all(
    ingested_data: Dict[str, Any],
    plan_number: int = 25
) -> Dict[str, Any]:
    """
    Preprocess all ingested data.
    
    Args:
        ingested_data: Dict from ingest.ingest_all()
        plan_number: Signal timing plan to use
        
    Returns:
        Dict mapping intersection_id to preprocessed data
    """
    volumes = ingested_data.get('volumes', {})
    timings = ingested_data.get('timings', {})
    matched = ingested_data.get('matched_intersections', [])
    
    results = {}
    
    for int_id in matched:
        volume_data = volumes.get(int_id, {})
        timing_data = timings.get(int_id, {})
        
        results[int_id] = preprocess_intersection(volume_data, timing_data, plan_number)
    
    return results


if __name__ == '__main__':
    from ingest import ingest_all
    import json
    
    # Test preprocessing
    print("Loading data...")
    ingested = ingest_all()
    
    print("Preprocessing data...")
    processed = preprocess_all(ingested)
    
    print("\nPreprocessing Results:")
    for int_id, data in processed.items():
        if data.get('error'):
            print(f"  {int_id}: Error - {data['error']}")
            continue
        
        vol_meta = data.get('volume_metadata', {})
        timing = data.get('timing_features', {})
        
        print(f"  {int_id}:")
        print(f"    Valid rows: {vol_meta.get('valid_rows', 'N/A')}")
        print(f"    Features added: {len(vol_meta.get('features_added', []))}")
        print(f"    Cycle length: {timing.get('cycle_length', 'N/A'):.1f}s")
        print(f"    Plan used: {timing.get('plan_used', 'N/A')}")
