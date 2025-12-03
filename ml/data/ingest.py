"""
Data Ingestion Module - Parse volume and timing CSV files.

This module reads the raw CSV files from volume/ and times/ directories
and provides structured data for preprocessing and model training.
"""

import os
import re
import glob
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np


# Constants
VOLUME_HEADER_ROWS = 2  # Skip first 2 header rows in volume files
MOVEMENT_COLS = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR', 
                 'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']


def parse_time_value(time_str: str) -> Optional[str]:
    """
    Parse TIME column value from volume files.
    
    Handles formats like:
    - ="0015" (Excel escaped format)
    - "0015"
    - 0015
    
    Args:
        time_str: Raw time string from CSV
        
    Returns:
        4-digit time string like "0015" or None if invalid
    """
    if pd.isna(time_str):
        return None
    
    time_str = str(time_str).strip()
    
    # Extract 4-digit time from various formats
    match = re.search(r'(\d{4})', time_str)
    if match:
        return match.group(1)
    
    return None


def ingest_volume_file(file_path: str) -> Dict[str, Any]:
    """
    Ingest a single volume (turning movement count) CSV file.
    
    Expected format:
    - Row 1-2: Headers
    - Row 3: Column names (DATE, TIME, INTID, NBL, NBT, ...)
    - Row 4+: Data rows with 15-minute interval counts
    
    Args:
        file_path: Path to volume CSV file
        
    Returns:
        Dict containing:
        - intersection_id: Derived from filename
        - raw_df: Raw DataFrame with parsed values
        - metadata: Dict with file info and parsing notes
    """
    filename = os.path.basename(file_path)
    intersection_id = os.path.splitext(filename)[0]
    
    metadata = {
        'source_file': file_path,
        'intersection_id': intersection_id,
        'missing_values': {},
        'notes': [],
        'warnings': []
    }
    
    try:
        # Read CSV skipping header rows
        df = pd.read_csv(file_path, skiprows=VOLUME_HEADER_ROWS, index_col=False)
        
        # Remove unnamed columns (from trailing commas)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
    except Exception as e:
        return {
            'intersection_id': intersection_id,
            'raw_df': None,
            'metadata': {
                **metadata,
                'error': f'Failed to read CSV: {str(e)}'
            }
        }
    
    # Validate required columns
    if 'DATE' not in df.columns or 'TIME' not in df.columns:
        metadata['error'] = 'Missing required DATE or TIME column'
        return {
            'intersection_id': intersection_id,
            'raw_df': None,
            'metadata': metadata
        }
    
    # Parse TIME column
    df['TIME'] = df['TIME'].apply(parse_time_value)
    
    # Track missing values in movement columns
    available_movements = [col for col in MOVEMENT_COLS if col in df.columns]
    
    for col in available_movements:
        # Count and track * values
        star_count = (df[col] == '*').sum()
        if star_count > 0:
            metadata['missing_values'][col] = star_count
        
        # Convert to numeric, marking * as NaN
        # Use pd.NA for replacement to avoid deprecation warning
        df[col] = df[col].astype(str).replace('*', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Store metadata
    metadata['row_count'] = len(df)
    metadata['available_movements'] = available_movements
    metadata['date_range'] = {
        'start': df['DATE'].iloc[0] if len(df) > 0 else None,
        'end': df['DATE'].iloc[-1] if len(df) > 0 else None
    }
    
    # Get INTID if available
    if 'INTID' in df.columns and len(df) > 0:
        metadata['intid'] = df['INTID'].iloc[0]
    
    total_missing = sum(metadata['missing_values'].values())
    if total_missing > 0:
        metadata['notes'].append(f"Found {total_missing} missing (*) values")
    
    return {
        'intersection_id': intersection_id,
        'raw_df': df,
        'metadata': metadata
    }


def ingest_phase_timing_file(file_path: str) -> Dict[str, Any]:
    """
    Ingest a single phase timing CSV file.
    
    Expected format:
    - Row 1-2: Headers (Plan, empty)
    - Row 3: Phase labels (Phase, empty, 1 EBLT, 2WB, ...)
    - Row 4+: Plan rows with green times
    - Yellow Change row
    - Red Clearence row
    
    Args:
        file_path: Path to phase timing CSV file
        
    Returns:
        Dict containing:
        - intersection_id: Derived from filename
        - plans: Dict mapping plan numbers to timing data
        - phase_labels: List of phase labels
        - yellow_times: Dict mapping phase labels to yellow times
        - red_clearance: Dict mapping phase labels to red clearance times
        - metadata: Dict with file info and parsing notes
    """
    filename = os.path.basename(file_path)
    intersection_id = os.path.splitext(filename)[0]
    
    metadata = {
        'source_file': file_path,
        'intersection_id': intersection_id,
        'notes': [],
        'warnings': []
    }
    
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        return {
            'intersection_id': intersection_id,
            'plans': {},
            'phase_labels': [],
            'yellow_times': {},
            'red_clearance': {},
            'metadata': {
                **metadata,
                'error': f'Failed to read CSV: {str(e)}'
            }
        }
    
    # Drop completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Find Phase header row
    phase_header_idx = None
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        if first_val == 'phase':
            phase_header_idx = int(idx)
            break
    
    if phase_header_idx is None:
        metadata['error'] = 'Phase header row not found'
        return {
            'intersection_id': intersection_id,
            'plans': {},
            'phase_labels': [],
            'yellow_times': {},
            'red_clearance': {},
            'metadata': metadata
        }
    
    # Extract phase labels and offset column
    phase_row = df.iloc[phase_header_idx]
    phase_labels = []
    offset_col_idx = None
    
    for col_idx in range(2, len(phase_row)):
        label = str(phase_row.iloc[col_idx]).strip() if pd.notna(phase_row.iloc[col_idx]) else ''
        if not label:
            continue
        if label.lower() == 'offset':
            offset_col_idx = col_idx
            break
        phase_labels.append((col_idx, label))
    
    # Parse plan rows, yellow change, and red clearance
    plans = {}
    yellow_times = {}
    red_clearance = {}
    
    for idx in range(phase_header_idx + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        
        if 'yellow' in first_cell:
            # Parse yellow times
            for col_idx, label in phase_labels:
                if col_idx < len(row):
                    try:
                        yt = float(row.iloc[col_idx])
                        if yt > 0:
                            yellow_times[label] = yt
                    except (ValueError, TypeError):
                        pass
            continue
        
        if 'red' in first_cell:
            # Parse red clearance times
            for col_idx, label in phase_labels:
                if col_idx < len(row):
                    try:
                        rc = float(row.iloc[col_idx])
                        if rc >= 0:
                            red_clearance[label] = rc
                    except (ValueError, TypeError):
                        pass
            continue
        
        # Check if this is a plan row
        plan_id = row.iloc[1]
        if pd.notna(plan_id):
            try:
                plan_num = int(float(plan_id))
                
                # Extract phase greens
                phase_greens = {}
                for col_idx, label in phase_labels:
                    if col_idx < len(row):
                        try:
                            green = float(row.iloc[col_idx])
                            if green > 0:
                                phase_greens[label] = green
                        except (ValueError, TypeError):
                            pass
                
                # Extract offset if available
                offset = None
                if offset_col_idx is not None and offset_col_idx < len(row):
                    try:
                        offset = float(row.iloc[offset_col_idx])
                    except (ValueError, TypeError):
                        pass
                
                plans[plan_num] = {
                    'phase_greens': phase_greens,
                    'offset': offset
                }
                
            except (ValueError, TypeError):
                continue
    
    metadata['plan_count'] = len(plans)
    metadata['phase_count'] = len(phase_labels)
    metadata['available_plans'] = list(plans.keys())
    
    return {
        'intersection_id': intersection_id,
        'plans': plans,
        'phase_labels': [label for _, label in phase_labels],
        'yellow_times': yellow_times,
        'red_clearance': red_clearance,
        'metadata': metadata
    }


def ingest_volume_directory(directory: str) -> Dict[str, Any]:
    """
    Ingest all volume files from a directory.
    
    Args:
        directory: Path to volume directory
        
    Returns:
        Dict mapping intersection_id to ingested data
    """
    result = {}
    files = glob.glob(os.path.join(directory, '*.csv'))
    
    for file_path in sorted(files):
        data = ingest_volume_file(file_path)
        intersection_id = data['intersection_id']
        result[intersection_id] = data
    
    return result


def ingest_timing_directory(directory: str) -> Dict[str, Any]:
    """
    Ingest all timing files from a directory.
    
    Args:
        directory: Path to times directory
        
    Returns:
        Dict mapping intersection_id to ingested data
    """
    result = {}
    files = glob.glob(os.path.join(directory, '*.csv'))
    
    for file_path in sorted(files):
        data = ingest_phase_timing_file(file_path)
        intersection_id = data['intersection_id']
        result[intersection_id] = data
    
    return result


def ingest_all(
    volume_dir: str = None,
    times_dir: str = None
) -> Dict[str, Any]:
    """
    Ingest all data from volume and times directories.
    
    Args:
        volume_dir: Path to volume directory (default: data/volume)
        times_dir: Path to times directory (default: data/times)
        
    Returns:
        Dict with:
        - volumes: Dict mapping intersection_id to volume data
        - timings: Dict mapping intersection_id to timing data
        - matched_intersections: List of intersection IDs with both volume and timing data
        - summary: Summary statistics
    """
    # Default to project root data directories
    if volume_dir is None:
        # Find project root (parent of ml directory)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        volume_dir = os.path.join(project_root, 'data', 'volume')
    if times_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        times_dir = os.path.join(project_root, 'data', 'times')
    
    volumes = ingest_volume_directory(volume_dir)
    timings = ingest_timing_directory(times_dir)
    
    # Find matched intersections
    volume_ids = set(volumes.keys())
    timing_ids = set(timings.keys())
    matched = sorted(volume_ids & timing_ids)
    
    # Compile summary
    total_rows = sum(
        v['metadata'].get('row_count', 0) 
        for v in volumes.values() 
        if v.get('raw_df') is not None
    )
    
    summary = {
        'volume_files': len(volumes),
        'timing_files': len(timings),
        'matched_intersections': len(matched),
        'total_volume_rows': total_rows,
        'unmatched_volumes': sorted(volume_ids - timing_ids),
        'unmatched_timings': sorted(timing_ids - volume_ids)
    }
    
    return {
        'volumes': volumes,
        'timings': timings,
        'matched_intersections': matched,
        'summary': summary
    }


if __name__ == '__main__':
    import json
    
    # Test ingestion
    result = ingest_all()
    
    print("Ingestion Summary:")
    print(json.dumps(result['summary'], indent=2))
    
    print("\nMatched Intersections:")
    for int_id in result['matched_intersections']:
        vol_meta = result['volumes'][int_id]['metadata']
        tim_meta = result['timings'][int_id]['metadata']
        print(f"  {int_id}:")
        print(f"    Volume rows: {vol_meta.get('row_count', 'N/A')}")
        print(f"    Date range: {vol_meta.get('date_range', {})}")
        print(f"    Available plans: {tim_meta.get('available_plans', [])}")
