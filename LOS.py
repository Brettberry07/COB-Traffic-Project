"""
LOS.py - Level of Service Calculator for Signalized Intersections

This module computes Level of Service (LOS) for signalized intersections using
FHWA/HCM (Highway Capacity Manual) methodology based on average control delay
in seconds per vehicle.

Reference: HCM 2010 (Highway Capacity Manual), Chapter 18 - Signalized Intersections
FHWA Signal Timing Manual, 2nd Edition

Example JSON Output Structure:
'''
{
  "intersection_id": "102_A",
  "source_files": {
    "phase_file": "102_A.csv",
    "volume_file": "102_A.csv"
  },
  "cycle_length_s": 295.0,
  "plan_used": 25,
  "per_lane": [
    {
      "movement": "NBT",
      "volume_vph": 450,
      "saturation_flow_vphpl": 1900,
      "g_s": 31.0,
      "g_over_C": 0.105,
      "degree_of_saturation": 1.25,
      "delay_s_per_veh": 85.2,
      "LOS": "F"
    },
    {
      "movement": "EBT",
      "volume_vph": 1220,
      "saturation_flow_vphpl": 1900,
      "g_s": 77.0,
      "g_over_C": 0.261,
      "degree_of_saturation": 2.46,
      "delay_s_per_veh": 150.0,
      "LOS": "F"
    }
  ],
  "intersection": {
    "average_delay_s_per_veh": 45.3,
    "LOS": "D",
    "total_volume_vph": 3250
  },
  "parameters": {
    "saturation_flow_vphpl": 1900,
    "aggregation": "peak_hour"
  },
  "notes": [
    "Missing values (*) treated as 0",
    "Used default saturation flow 1900 vph/ln",
    "Peak hour detected: 07:00-08:00"
  ],
  "warnings": []
}
'''

LOS Thresholds (HCM 2010, Signalized Intersections):
  LOS A: delay <= 10 seconds
  LOS B: 10 < delay <= 20 seconds
  LOS C: 20 < delay <= 35 seconds
  LOS D: 35 < delay <= 55 seconds
  LOS E: 55 < delay <= 80 seconds
  LOS F: delay > 80 seconds

Control Delay Calculation (Simplified HCM Approach):
  The HCM composite delay equation is:
    d = d1*PF + d2 + d3

  Where:
    d1 = uniform delay = (0.5 * C * (1 - g/C)^2) / (1 - min(1, X) * g/C)
    d2 = incremental delay = 900 * T * [(X - 1) + sqrt((X-1)^2 + 8*k*I*X/(c*T))]
    d3 = initial queue delay (assumed 0 for this implementation)
    PF = progression factor (assumed 1.0 for isolated signals)
    X = degree of saturation = v / c = v / (s * g/C)
    T = analysis period (typically 0.25 hours for 15-min analysis)
    k = incremental delay factor (0.5 for typical pre-timed)
    I = upstream filtering/metering adjustment (1.0 for isolated)
    c = capacity = s * g/C

  APPROXIMATION NOTE:
    This implementation uses the simplified HCM delay equation. For X >= 1.0,
    the delay is capped at a high value (150 seconds) to indicate severe
    oversaturation. The full HCM incremental delay term with queue accumulation
    over the analysis period would produce different results for X > 1.0.
    
    For X < 1.0, the approximation is quite accurate compared to the full HCM method.
"""

import os
import re
import json
import sys
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np


# =============================================================================
# Constants and Thresholds
# =============================================================================

# HCM 2010 LOS thresholds for signalized intersections (seconds per vehicle)
LOS_THRESHOLDS = {
    'A': (0, 10),
    'B': (10, 20),
    'C': (20, 35),
    'D': (35, 55),
    'E': (55, 80),
    'F': (80, float('inf'))
}

# Standard movement column names in volume files
MOVEMENT_COLS = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR', 
                 'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']

# Default parameters
DEFAULT_SATURATION_FLOW = 1900  # veh/hr/lane
DEFAULT_ANALYSIS_PERIOD = 0.25  # hours (15 minutes)
DEFAULT_INCREMENTAL_DELAY_FACTOR = 0.5  # k factor for pre-timed signals
DEFAULT_UPSTREAM_FACTOR = 1.0  # I factor for isolated signals
DEFAULT_PROGRESSION_FACTOR = 1.0  # PF for isolated signals

# Maximum delay cap for oversaturated conditions
MAX_DELAY_CAP = 150.0  # seconds


# =============================================================================
# Phase-to-Movement Mapping
# =============================================================================

def parse_phase_label(phase_label: str) -> List[str]:
    """
    Parse a phase label like '1 EBLT' or '2WB' to extract movement codes.
    
    Returns a list of movements served by this phase.
    Phase labels may contain:
      - Direction prefix: NB, SB, EB, WB
      - Movement suffix: L (left), T (through), R (right), LT (left-through)
    
    Examples:
      '1 EBLT' -> ['EBL', 'EBT']
      '2WB' -> ['WBL', 'WBT', 'WBR']
      '3NBLT' -> ['NBL', 'NBT']
      '4SB' -> ['SBL', 'SBT', 'SBR']
      '8SBRT' -> ['SBR', 'SBT']
    """
    # Remove numeric prefix and spaces
    cleaned = re.sub(r'^[\d\s]+', '', phase_label.strip())
    cleaned = cleaned.replace(' ', '').upper()
    
    if not cleaned or cleaned.lower() in ('not used', 'notused', 'offset'):
        return []
    
    movements = []
    
    # Extract direction (NB, SB, EB, WB)
    direction_match = re.match(r'^(NB|SB|EB|WB)', cleaned)
    if not direction_match:
        return []
    
    direction = direction_match.group(1)
    suffix = cleaned[len(direction):]
    
    # Parse movement types from suffix
    if not suffix:
        # Just direction means all movements for that approach
        movements = [f'{direction}L', f'{direction}T', f'{direction}R']
    elif suffix in ('L', 'LT', 'TL'):
        if 'L' in suffix:
            movements.append(f'{direction}L')
        if 'T' in suffix:
            movements.append(f'{direction}T')
    elif suffix == 'T':
        movements.append(f'{direction}T')
    elif suffix == 'R':
        movements.append(f'{direction}R')
    elif suffix in ('RT', 'TR'):
        movements.append(f'{direction}R')
        movements.append(f'{direction}T')
    elif suffix == 'LR':
        movements.append(f'{direction}L')
        movements.append(f'{direction}R')
    elif suffix == 'LTR':
        movements = [f'{direction}L', f'{direction}T', f'{direction}R']
    else:
        # Try to parse individual characters
        if 'L' in suffix:
            movements.append(f'{direction}L')
        if 'T' in suffix:
            movements.append(f'{direction}T')
        if 'R' in suffix:
            movements.append(f'{direction}R')
    
    return movements


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_phase_timing_csv(file_path: str, plan_number: int = 25) -> Dict[str, Any]:
    """
    Parse a phase timing CSV file.
    
    Expected format:
      Row with 'Phase' in first column contains phase labels
      Subsequent rows contain plan number and phase green times
      'Yellow Change' row contains yellow times per phase
      'Red Clearence' (note spelling) row contains red clearance times
    
    Args:
        file_path: Path to the phase timing CSV file
        plan_number: Signal timing plan number to use (default 25)
    
    Returns:
        Dict containing:
          - intersection_id: Derived from filename
          - plan_used: The plan number actually used
          - cycle_length_s: Total cycle length in seconds
          - phases: Dict mapping phase labels to green times
          - yellow_times: Dict mapping phase labels to yellow times
          - red_clearance_times: Dict mapping phase labels to red clearance times
          - movement_green_times: Dict mapping movements to their green times
          - notes: List of parsing notes
          - warnings: List of parsing warnings
    """
    notes = []
    warnings = []
    
    # Derive intersection ID from filename
    filename = os.path.basename(file_path)
    intersection_id = os.path.splitext(filename)[0]
    
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        return {
            'intersection_id': intersection_id,
            'error': f'Failed to read CSV: {str(e)}',
            'notes': notes,
            'warnings': [f'CSV read error: {str(e)}']
        }
    
    # Drop completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)
    
    # Find the Phase header row
    phase_header_idx = None
    for idx, row in df.iterrows():
        first_val = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        if first_val == 'phase':
            phase_header_idx = int(idx)
            break
    
    if phase_header_idx is None:
        return {
            'intersection_id': intersection_id,
            'error': 'Phase header row not found',
            'notes': notes,
            'warnings': ['Could not locate Phase header row in timing file']
        }
    
    # Extract phase labels from header row
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
    
    if offset_col_idx is None:
        offset_col_idx = len(phase_row)
    
    # Find plan rows, yellow change row, and red clearance row
    plan_rows = []
    yellow_row = None
    red_row = None
    
    for idx in range(phase_header_idx + 1, len(df)):
        row = df.iloc[idx]
        first_cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ''
        
        if 'yellow' in first_cell:
            yellow_row = row
            continue
        if 'red' in first_cell:
            red_row = row
            continue
        
        # Check if this is a plan row (has a numeric value in column 1)
        plan_id = row.iloc[1]
        if pd.notna(plan_id):
            try:
                plan_val = int(float(plan_id))
                plan_rows.append((plan_val, row))
            except (ValueError, TypeError):
                continue
    
    if not plan_rows:
        return {
            'intersection_id': intersection_id,
            'error': 'No plan rows found in timing file',
            'notes': notes,
            'warnings': ['No signal timing plans found']
        }
    
    # Select the appropriate plan
    selected_plan = None
    selected_row = None
    
    for plan_val, row in plan_rows:
        if plan_val == plan_number:
            selected_plan = plan_val
            selected_row = row
            break
    
    if selected_row is None:
        # Use first available plan as fallback
        selected_plan, selected_row = plan_rows[0]
        warnings.append(f'Plan {plan_number} not found, using plan {selected_plan}')
    
    notes.append(f'Using signal timing plan {selected_plan}')
    
    # Extract phase green times
    phases = {}
    for col_idx, label in phase_labels:
        if col_idx < len(selected_row):
            try:
                green_time = float(selected_row.iloc[col_idx])
                if green_time > 0:
                    phases[label] = green_time
            except (ValueError, TypeError):
                warnings.append(f'Invalid green time for phase {label}')
    
    # Extract yellow times
    yellow_times = {}
    if yellow_row is not None:
        for col_idx, label in phase_labels:
            if col_idx < len(yellow_row):
                try:
                    yt = float(yellow_row.iloc[col_idx])
                    if yt > 0:
                        yellow_times[label] = yt
                except (ValueError, TypeError):
                    pass
    
    # Extract red clearance times
    red_clearance_times = {}
    if red_row is not None:
        for col_idx, label in phase_labels:
            if col_idx < len(red_row):
                try:
                    rc = float(red_row.iloc[col_idx])
                    if rc >= 0:
                        red_clearance_times[label] = rc
                except (ValueError, TypeError):
                    pass
    
    # Calculate cycle length
    # Signal phases typically run in dual-ring configuration where:
    # Ring 1: odd phases (1,3,5,7) and Ring 2: even phases (2,4,6,8)
    # OR Ring 1: phases 1-4 and Ring 2: phases 5-8
    # The cycle length is the sum of concurrent phase pairs plus clearance.
    #
    # For typical NEMA phasing:
    # - Phases 1&2 and 5&6 are barrier 1 (usually main street)
    # - Phases 3&4 and 7&8 are barrier 2 (usually cross street)
    #
    # Cycle = max phase in pair 1-2 + max in pair 3-4 + max in pair 5-6 + max in pair 7-8
    #       + clearance intervals
    # 
    # Simplified approach: Use the offset value if available (often equals cycle length)
    # or calculate based on ring structure
    
    phase_greens = list(phases.values())
    num_phases = len(phase_greens)
    
    # Get offset from the row if it was stored
    offset_value = None
    if offset_col_idx is not None and offset_col_idx < len(selected_row):
        try:
            offset_value = float(selected_row.iloc[offset_col_idx])
        except (ValueError, TypeError):
            pass
    
    # Calculate total clearance (yellow + red for each phase change)
    avg_yellow = np.mean(list(yellow_times.values())) if yellow_times else 4.0
    avg_red_clearance = np.mean(list(red_clearance_times.values())) if red_clearance_times else 1.0
    
    # Estimate cycle length based on ring structure
    if num_phases >= 8:
        # Dual-ring 8-phase: concurrent phases run together
        # Ring 1 phases: 0,2,4,6 (even indices = 1,3,5,7 NEMA)
        # Ring 2 phases: 1,3,5,7 (odd indices = 2,4,6,8 NEMA)
        # Barrier 1: max(phase 1,2) + max(phase 5,6)
        # Barrier 2: max(phase 3,4) + max(phase 7,8)
        # But actually the greens in CSV are sequential, so let's use a simpler approach:
        # Cycle = (sum of first 4 phases OR sum of last 4 phases - whichever is greater)
        #       + number of phase transitions * clearance time
        ring1_sum = sum(phase_greens[:4])
        ring2_sum = sum(phase_greens[4:8])
        max_ring = max(ring1_sum, ring2_sum)
        num_transitions = 4  # typical 4 phase changes per cycle
        cycle_length = max_ring + num_transitions * (avg_yellow + avg_red_clearance)
    elif num_phases >= 4:
        # 4-phase signal
        ring_sum = sum(phase_greens[:4])
        num_transitions = 2
        cycle_length = ring_sum + num_transitions * (avg_yellow + avg_red_clearance)
    else:
        # Simple 2-phase
        cycle_length = sum(phase_greens) + 2 * (avg_yellow + avg_red_clearance)
    
    # If offset value is available and seems reasonable, use it as a sanity check
    # Offset often represents the cycle length in coordinated signals
    if offset_value and offset_value > 30 and offset_value < 300:
        # Use offset as cycle length if it's reasonable
        # This is common in coordinated signal timing files
        if abs(cycle_length - offset_value) > 50:
            notes.append(f'Using offset value {offset_value}s as cycle length (computed was {cycle_length:.1f}s)')
            cycle_length = offset_value
    
    if cycle_length <= 0:
        return {
            'intersection_id': intersection_id,
            'error': 'Computed cycle length is zero or negative',
            'notes': notes,
            'warnings': warnings + ['Invalid cycle length computed']
        }
    
    # Map phases to movements
    movement_green_times = {}
    for phase_label, green_time in phases.items():
        movements = parse_phase_label(phase_label)
        for mov in movements:
            if mov in movement_green_times:
                # Take maximum green time if movement served by multiple phases
                movement_green_times[mov] = max(movement_green_times[mov], green_time)
            else:
                movement_green_times[mov] = green_time
    
    notes.append(f'Mapped {len(phases)} phases to {len(movement_green_times)} movements')
    
    return {
        'intersection_id': intersection_id,
        'plan_used': selected_plan,
        'cycle_length_s': cycle_length,
        'phases': phases,
        'yellow_times': yellow_times,
        'red_clearance_times': red_clearance_times,
        'movement_green_times': movement_green_times,
        'notes': notes,
        'warnings': warnings
    }


def parse_volume_csv(file_path: str, aggregation: str = 'peak_hour') -> Dict[str, Any]:
    """
    Parse a volume (turning movement count) CSV file.
    
    Expected format:
      - Header rows with metadata
      - Data row with columns: DATE, TIME, INTID, NBL, NBT, NBR, SBL, SBT, SBR, EBL, EBT, EBR, WBL, WBT, WBR
      - TIME formatted as ="0000" or "0000"
      - Values may include '*' or empty cells (treated as 0)
    
    Args:
        file_path: Path to the volume CSV file
        aggregation: 'peak_hour' to find highest 1-hour sum, 'total' for full file sum
    
    Returns:
        Dict containing:
          - intersection_id: From INTID column or filename
          - volumes_vph: Dict mapping movements to hourly volumes
          - peak_hour_start: Start time of peak hour (if peak_hour aggregation)
          - total_intervals: Number of 15-min intervals in data
          - notes: List of parsing notes
          - warnings: List of parsing warnings
    """
    notes = []
    warnings = []
    missing_value_count = 0
    
    # Derive intersection ID from filename
    filename = os.path.basename(file_path)
    intersection_id = os.path.splitext(filename)[0]
    
    try:
        # Read the CSV, skipping the first 2 header lines
        # Use index_col=False to prevent first column from being used as index
        df = pd.read_csv(file_path, skiprows=2, index_col=False)
        
        # Remove any unnamed columns (from trailing commas)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    except Exception as e:
        return {
            'intersection_id': intersection_id,
            'error': f'Failed to read CSV: {str(e)}',
            'notes': notes,
            'warnings': [f'CSV read error: {str(e)}']
        }
    
    # Verify required columns exist
    required_cols = ['DATE', 'TIME']
    for col in required_cols:
        if col not in df.columns:
            return {
                'intersection_id': intersection_id,
                'error': f'Required column {col} not found',
                'notes': notes,
                'warnings': [f'Missing column: {col}']
            }
    
    # Find movement columns that exist in this file
    available_movements = [col for col in MOVEMENT_COLS if col in df.columns]
    if not available_movements:
        return {
            'intersection_id': intersection_id,
            'error': 'No movement columns found',
            'notes': notes,
            'warnings': ['No movement columns (NBL, NBT, etc.) found in volume file']
        }
    
    # Parse TIME column (handles ="0000" format)
    df['TIME'] = df['TIME'].astype(str).str.extract(r'(\d{4})', expand=False)
    
    # Convert movement columns to numeric, treating '*' and invalid values as NaN
    for col in available_movements:
        df[col] = pd.to_numeric(df[col].replace('*', np.nan), errors='coerce')
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_value_count += missing_count
    
    if missing_value_count > 0:
        notes.append(f'Missing values (*) treated as 0: {missing_value_count} instances')
    
    # Fill NaN with 0
    df[available_movements] = df[available_movements].fillna(0)
    
    # Check for lanes that are ALL zeros or NaN (should be excluded from LOS calc)
    lanes_with_data = []
    lanes_excluded = []
    for col in available_movements:
        if df[col].sum() > 0:
            lanes_with_data.append(col)
        else:
            lanes_excluded.append(col)
    
    if lanes_excluded:
        notes.append(f'Lanes with no valid data excluded: {lanes_excluded}')
    
    # Create datetime for time-based aggregation
    try:
        # Ensure TIME is properly formatted
        df['TIME'] = df['TIME'].fillna('0000').astype(str).str.zfill(4)
        df['DATE'] = df['DATE'].fillna('').astype(str)
        
        # Filter out rows with invalid date/time
        valid_rows = (df['DATE'].str.len() > 0) & (df['TIME'].str.len() == 4)
        df_valid = df[valid_rows].copy()
        
        if len(df_valid) > 0:
            df_valid['DateTime'] = pd.to_datetime(
                df_valid['DATE'] + ' ' + df_valid['TIME'],
                format='%m/%d/%Y %H%M',
                errors='coerce'
            )
            df = df_valid.dropna(subset=['DateTime']).copy()
            df = df.sort_values('DateTime')
        else:
            warnings.append('No valid date/time rows found')
            df['DateTime'] = pd.RangeIndex(len(df))
    except Exception as e:
        warnings.append(f'Could not parse datetime: {str(e)}')
        df['DateTime'] = pd.RangeIndex(len(df))
    
    total_intervals = len(df)
    notes.append(f'Total 15-minute intervals: {total_intervals}')
    
    volumes_vph = {}
    peak_hour_start = None
    
    if aggregation == 'peak_hour' and total_intervals >= 4:
        # Find the 4 consecutive intervals with highest total volume
        df['TotalVol'] = df[lanes_with_data].sum(axis=1)
        
        best_sum = 0
        best_start_idx = 0
        
        for i in range(len(df) - 3):
            window_sum = df['TotalVol'].iloc[i:i+4].sum()
            if window_sum > best_sum:
                best_sum = window_sum
                best_start_idx = i
        
        peak_df = df.iloc[best_start_idx:best_start_idx+4]
        
        # Get peak hour start time
        if 'DateTime' in df.columns:
            peak_dt = peak_df['DateTime'].iloc[0]
            if pd.notna(peak_dt):
                try:
                    peak_hour_start = peak_dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    peak_hour_start = str(peak_dt)
            else:
                peak_hour_start = f"Interval {best_start_idx}"
        else:
            peak_hour_start = f"Interval {best_start_idx}"
        
        notes.append(f'Peak hour detected starting at: {peak_hour_start}')
        
        # Sum volumes for peak hour (already hourly since 4 x 15-min = 1 hour)
        for col in lanes_with_data:
            volumes_vph[col] = int(peak_df[col].sum())
    
    else:
        # Use total file sum converted to hourly rate
        total_hours = total_intervals * 0.25  # 15-min intervals to hours
        if total_hours <= 0:
            total_hours = 1
        
        for col in lanes_with_data:
            total_count = df[col].sum()
            volumes_vph[col] = int(total_count / total_hours)
        
        notes.append(f'Using total volume over {total_hours:.1f} hours as hourly rate')
    
    # Get INTID if available
    if 'INTID' in df.columns and len(df) > 0:
        intid = df['INTID'].iloc[0]
        notes.append(f'INTID from file: {intid}')
    
    return {
        'intersection_id': intersection_id,
        'volumes_vph': volumes_vph,
        'peak_hour_start': peak_hour_start,
        'total_intervals': total_intervals,
        'lanes_with_data': lanes_with_data,
        'notes': notes,
        'warnings': warnings
    }


# =============================================================================
# LOS Calculation Functions
# =============================================================================

def determine_los(delay: float) -> str:
    """
    Determine Level of Service based on control delay.
    
    Uses HCM 2010 thresholds for signalized intersections.
    
    Args:
        delay: Average control delay in seconds per vehicle
    
    Returns:
        LOS grade as single character string ('A' through 'F')
    """
    if delay is None or np.isnan(delay):
        return 'F'
    
    for los, (min_delay, max_delay) in LOS_THRESHOLDS.items():
        if min_delay <= delay < max_delay:
            return los
    
    return 'F'


def compute_control_delay(
    volume_vph: float,
    saturation_flow: float,
    green_time: float,
    cycle_length: float,
    analysis_period: float = DEFAULT_ANALYSIS_PERIOD,
    k_factor: float = DEFAULT_INCREMENTAL_DELAY_FACTOR,
    i_factor: float = DEFAULT_UPSTREAM_FACTOR,
    pf: float = DEFAULT_PROGRESSION_FACTOR
) -> Tuple[float, float, float]:
    """
    Compute control delay using HCM methodology.
    
    HCM 2010 Control Delay Equation:
      d = d1*PF + d2 + d3
    
    Where:
      d1 = uniform delay component
      d2 = incremental delay component (accounts for random arrivals and oversaturation)
      d3 = initial queue delay (assumed 0)
    
    APPROXIMATION NOTE:
      This implementation uses the standard HCM equations but makes the following
      simplifications:
      1. PF (progression factor) is assumed to be 1.0 (isolated/random arrivals)
      2. Initial queue delay (d3) is assumed to be 0
      3. For X >= 1.0 (oversaturation), delay is capped at MAX_DELAY_CAP
      
      The d2 incremental delay uses the HCM equation:
        d2 = 900 * T * [(X-1) + sqrt((X-1)^2 + 8*k*I*X/(c*T))]
      
      where c = capacity = s * (g/C)
    
    Args:
        volume_vph: Volume in vehicles per hour
        saturation_flow: Saturation flow rate in veh/hr/lane
        green_time: Effective green time in seconds
        cycle_length: Cycle length in seconds
        analysis_period: Analysis period in hours (default 0.25 = 15 min)
        k_factor: Incremental delay factor (0.5 for pre-timed, 0.4 for actuated)
        i_factor: Upstream filtering adjustment (1.0 for isolated)
        pf: Progression factor (1.0 for random arrivals)
    
    Returns:
        Tuple of (total_delay, degree_of_saturation, g_over_C)
    """
    if cycle_length <= 0 or green_time <= 0 or saturation_flow <= 0:
        return (MAX_DELAY_CAP, float('inf'), 0.0)
    
    # Effective green ratio
    g_over_C = green_time / cycle_length
    
    if g_over_C <= 0:
        return (MAX_DELAY_CAP, float('inf'), g_over_C)
    
    # Capacity (veh/hr)
    capacity = saturation_flow * g_over_C
    
    if capacity <= 0:
        return (MAX_DELAY_CAP, float('inf'), g_over_C)
    
    # Degree of saturation (v/c ratio)
    X = volume_vph / capacity
    
    # Uniform delay (d1)
    # d1 = 0.5 * C * (1 - g/C)^2 / (1 - min(1, X) * g/C)
    numerator = 0.5 * cycle_length * (1 - g_over_C) ** 2
    denominator = 1 - min(1.0, X) * g_over_C
    
    if denominator <= 0.001:
        d1 = MAX_DELAY_CAP
    else:
        d1 = numerator / denominator
    
    # Incremental delay (d2)
    # d2 = 900 * T * [(X-1) + sqrt((X-1)^2 + 8*k*I*X/(c*T))]
    T = analysis_period
    c_T = capacity * T  # capacity in veh for analysis period
    
    if c_T <= 0:
        d2 = MAX_DELAY_CAP
    else:
        term_inside_sqrt = (X - 1) ** 2 + (8 * k_factor * i_factor * X) / c_T
        d2 = 900 * T * ((X - 1) + np.sqrt(max(0, term_inside_sqrt)))
    
    # Total delay (d3 assumed 0)
    total_delay = d1 * pf + d2
    
    # Cap at maximum delay
    total_delay = min(total_delay, MAX_DELAY_CAP)
    total_delay = max(0, total_delay)
    
    return (total_delay, X, g_over_C)


def compute_intersection_los(
    per_lane_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute intersection-level LOS from per-lane results.
    
    HCM methodology: Intersection LOS is based on the volume-weighted average
    delay across all lane groups/movements.
    
    Args:
        per_lane_results: List of per-lane result dicts with 'volume_vph' and 'delay_s_per_veh'
    
    Returns:
        Dict with intersection-level average delay and LOS
    """
    total_volume = 0
    weighted_delay_sum = 0
    
    for lane in per_lane_results:
        vol = lane.get('volume_vph', 0) or 0
        delay = lane.get('delay_s_per_veh')
        
        if delay is not None and vol > 0:
            total_volume += vol
            weighted_delay_sum += vol * delay
    
    if total_volume > 0:
        avg_delay = weighted_delay_sum / total_volume
    else:
        avg_delay = 0
    
    return {
        'average_delay_s_per_veh': round(avg_delay, 2),
        'LOS': determine_los(avg_delay),
        'total_volume_vph': total_volume
    }


# =============================================================================
# Main API Functions
# =============================================================================

def compute_los_for_intersection(
    phase_csv: str,
    volume_csv: str,
    *,
    saturation_flow: int = DEFAULT_SATURATION_FLOW,
    peak_hour_window: int = 60,
    plan_number: int = 25,
    aggregation: str = 'peak_hour'
) -> Dict[str, Any]:
    """
    Compute LOS for a single intersection.
    
    Args:
        phase_csv: Path to phase timing CSV file
        volume_csv: Path to volume (turning movement) CSV file
        saturation_flow: Saturation flow rate in veh/hr/lane (default 1900)
        peak_hour_window: Peak hour window in minutes (default 60)
        plan_number: Signal timing plan number to use (default 25)
        aggregation: 'peak_hour' or 'total' for volume aggregation
    
    Returns:
        JSON-serializable dict with LOS results
    """
    notes = []
    warnings = []
    
    # Parse phase timing file
    timing_result = parse_phase_timing_csv(phase_csv, plan_number=plan_number)
    
    if 'error' in timing_result:
        return {
            'intersection_id': timing_result.get('intersection_id', 'unknown'),
            'error': timing_result['error'],
            'source_files': {
                'phase_file': os.path.basename(phase_csv),
                'volume_file': os.path.basename(volume_csv)
            },
            'notes': timing_result.get('notes', []),
            'warnings': timing_result.get('warnings', [])
        }
    
    notes.extend(timing_result.get('notes', []))
    warnings.extend(timing_result.get('warnings', []))
    
    # Parse volume file
    volume_result = parse_volume_csv(volume_csv, aggregation=aggregation)
    
    if 'error' in volume_result:
        return {
            'intersection_id': timing_result.get('intersection_id', 'unknown'),
            'error': volume_result['error'],
            'source_files': {
                'phase_file': os.path.basename(phase_csv),
                'volume_file': os.path.basename(volume_csv)
            },
            'notes': notes + volume_result.get('notes', []),
            'warnings': warnings + volume_result.get('warnings', [])
        }
    
    notes.extend(volume_result.get('notes', []))
    warnings.extend(volume_result.get('warnings', []))
    
    # Get parameters
    cycle_length = timing_result['cycle_length_s']
    movement_green_times = timing_result.get('movement_green_times', {})
    volumes = volume_result.get('volumes_vph', {})
    
    # Add parameter note
    notes.append(f'Used saturation flow {saturation_flow} vph/ln')
    
    # Compute LOS for each movement
    per_lane_results = []
    
    for movement, volume_vph in volumes.items():
        if volume_vph == 0:
            # Skip movements with zero volume
            continue
        
        # Get green time for this movement
        green_time = movement_green_times.get(movement)
        
        if green_time is None:
            # Try to find a matching phase
            # Look for partial matches (e.g., 'EB' for 'EBT')
            direction = movement[:2]
            for phase_label, g in movement_green_times.items():
                if direction in phase_label.upper():
                    green_time = g
                    break
        
        if green_time is None or green_time <= 0:
            warnings.append(f'No green time found for movement {movement}')
            per_lane_results.append({
                'movement': movement,
                'volume_vph': volume_vph,
                'saturation_flow_vphpl': saturation_flow,
                'g_s': None,
                'g_over_C': None,
                'degree_of_saturation': None,
                'delay_s_per_veh': None,
                'LOS': None,
                'warning': 'No green time found'
            })
            continue
        
        # Compute delay
        delay, X, g_over_C = compute_control_delay(
            volume_vph=volume_vph,
            saturation_flow=saturation_flow,
            green_time=green_time,
            cycle_length=cycle_length
        )
        
        per_lane_results.append({
            'movement': movement,
            'volume_vph': volume_vph,
            'saturation_flow_vphpl': saturation_flow,
            'g_s': round(green_time, 1),
            'g_over_C': round(g_over_C, 3),
            'degree_of_saturation': round(X, 3),
            'delay_s_per_veh': round(delay, 1),
            'LOS': determine_los(delay)
        })
    
    # Compute intersection-level LOS
    intersection_summary = compute_intersection_los(per_lane_results)
    
    return {
        'intersection_id': timing_result['intersection_id'],
        'source_files': {
            'phase_file': os.path.basename(phase_csv),
            'volume_file': os.path.basename(volume_csv)
        },
        'cycle_length_s': round(cycle_length, 1),
        'plan_used': timing_result['plan_used'],
        'per_lane': per_lane_results,
        'intersection': intersection_summary,
        'parameters': {
            'saturation_flow_vphpl': saturation_flow,
            'aggregation': aggregation
        },
        'notes': notes,
        'warnings': warnings
    }


def compute_los_for_files(
    phase_file_paths: List[str],
    volume_file_paths: List[str],
    *,
    saturation_flow: int = DEFAULT_SATURATION_FLOW,
    aggregation: str = 'peak_hour',
    plan_number: int = 25
) -> Dict[str, Any]:
    """
    Compute LOS for multiple intersections from lists of phase and volume files.
    
    Files are matched by their intersection identifier (filename prefix before underscore).
    
    Args:
        phase_file_paths: List of paths to phase timing CSV files
        volume_file_paths: List of paths to volume CSV files
        saturation_flow: Saturation flow rate in veh/hr/lane (default 1900)
        aggregation: 'peak_hour' or 'total' for volume aggregation
        plan_number: Signal timing plan number to use (default 25)
    
    Returns:
        Dict keyed by intersection_id with per-intersection results
    """
    results = {}
    notes = []
    warnings = []
    
    # Build lookup tables by filename
    phase_by_name = {os.path.basename(p): p for p in phase_file_paths}
    volume_by_name = {os.path.basename(p): p for p in volume_file_paths}
    
    # Also build lookup by intersection ID
    def extract_intersection_id(filename):
        """Extract intersection ID from filename like '102_A.csv'"""
        name = os.path.splitext(filename)[0]
        return name
    
    phase_by_id = {extract_intersection_id(os.path.basename(p)): p for p in phase_file_paths}
    volume_by_id = {extract_intersection_id(os.path.basename(p)): p for p in volume_file_paths}
    
    # Match files by intersection ID
    all_ids = set(phase_by_id.keys()) | set(volume_by_id.keys())
    
    for int_id in sorted(all_ids):
        phase_file = phase_by_id.get(int_id)
        volume_file = volume_by_id.get(int_id)
        
        if phase_file is None:
            warnings.append(f'No phase timing file found for intersection {int_id}')
            results[int_id] = {
                'intersection_id': int_id,
                'error': 'No phase timing file found',
                'warnings': [f'No phase timing file for {int_id}']
            }
            continue
        
        if volume_file is None:
            warnings.append(f'No volume file found for intersection {int_id}')
            results[int_id] = {
                'intersection_id': int_id,
                'error': 'No volume file found',
                'warnings': [f'No volume file for {int_id}']
            }
            continue
        
        # Compute LOS for this intersection
        result = compute_los_for_intersection(
            phase_csv=phase_file,
            volume_csv=volume_file,
            saturation_flow=saturation_flow,
            plan_number=plan_number,
            aggregation=aggregation
        )
        
        results[int_id] = result
    
    return {
        'intersections': results,
        'summary': {
            'total_intersections': len(results),
            'successful': sum(1 for r in results.values() if 'error' not in r),
            'failed': sum(1 for r in results.values() if 'error' in r)
        },
        'parameters': {
            'saturation_flow_vphpl': saturation_flow,
            'aggregation': aggregation,
            'plan_number': plan_number
        },
        'notes': notes,
        'warnings': warnings
    }


def compute_los_for_directories(
    phase_dir: str,
    volume_dir: str,
    *,
    saturation_flow: int = DEFAULT_SATURATION_FLOW,
    aggregation: str = 'peak_hour',
    plan_number: int = 25
) -> Dict[str, Any]:
    """
    Convenience function to compute LOS for all intersections in directories.
    
    Args:
        phase_dir: Directory containing phase timing CSV files
        volume_dir: Directory containing volume CSV files
        saturation_flow: Saturation flow rate in veh/hr/lane
        aggregation: 'peak_hour' or 'total'
        plan_number: Signal timing plan number to use
    
    Returns:
        Dict with results for all matched intersections
    """
    import glob
    
    phase_files = glob.glob(os.path.join(phase_dir, '*.csv'))
    volume_files = glob.glob(os.path.join(volume_dir, '*.csv'))
    
    return compute_los_for_files(
        phase_file_paths=phase_files,
        volume_file_paths=volume_files,
        saturation_flow=saturation_flow,
        aggregation=aggregation,
        plan_number=plan_number
    )


# =============================================================================
# Test Functions (pytest style)
# =============================================================================

def test_parse_phase_label():
    """Test phase label parsing."""
    assert parse_phase_label('1 EBLT') == ['EBL', 'EBT']
    assert parse_phase_label('2WB') == ['WBL', 'WBT', 'WBR']
    assert parse_phase_label('3NBLT') == ['NBL', 'NBT']
    assert parse_phase_label('4SB') == ['SBL', 'SBT', 'SBR']
    assert parse_phase_label('8SBRT') == ['SBR', 'SBT']
    assert parse_phase_label('6 EB') == ['EBL', 'EBT', 'EBR']
    assert parse_phase_label('5WBLT') == ['WBL', 'WBT']
    assert parse_phase_label('4(Not used)') == []
    assert parse_phase_label('Offset') == []
    print("test_parse_phase_label PASSED")


def test_determine_los():
    """Test LOS determination from delay values."""
    assert determine_los(5) == 'A'
    assert determine_los(10) == 'B'
    assert determine_los(15) == 'B'
    assert determine_los(20) == 'C'
    assert determine_los(30) == 'C'
    assert determine_los(35) == 'D'
    assert determine_los(50) == 'D'
    assert determine_los(55) == 'E'
    assert determine_los(75) == 'E'
    assert determine_los(80) == 'F'
    assert determine_los(100) == 'F'
    assert determine_los(0) == 'A'
    print("test_determine_los PASSED")


def test_compute_control_delay():
    """Test control delay calculation with known inputs."""
    # Test case: Moderate volume, adequate green
    # v=500, s=1900, g=30, C=100 -> g/C=0.3, X=500/(1900*0.3)=0.877
    # At X=0.877, expect moderate delay in LOS C-D range
    delay, X, g_over_C = compute_control_delay(
        volume_vph=500,
        saturation_flow=1900,
        green_time=30,
        cycle_length=100
    )
    assert 0.2 < g_over_C < 0.4
    assert 0.7 < X < 1.0
    assert 20 < delay < 80  # Should be LOS C-D range for X near 0.9
    print(f"  Moderate volume case: delay={delay:.1f}s, X={X:.2f}, LOS={determine_los(delay)}")
    
    # Test case: High volume, constrained green (oversaturated)
    # v=1500, s=1900, g=20, C=100 -> g/C=0.2, X=1500/(1900*0.2)=3.95
    delay2, X2, g_over_C2 = compute_control_delay(
        volume_vph=1500,
        saturation_flow=1900,
        green_time=20,
        cycle_length=100
    )
    assert X2 > 1.0  # Oversaturated
    assert delay2 >= 80  # Should be LOS F
    print(f"  High volume case: delay={delay2:.1f}s, X={X2:.2f}, LOS={determine_los(delay2)}")
    
    # Test case: Very low volume
    delay3, X3, g_over_C3 = compute_control_delay(
        volume_vph=100,
        saturation_flow=1900,
        green_time=40,
        cycle_length=90
    )
    assert X3 < 0.2
    assert delay3 < 20  # Should be LOS A-B
    print(f"  Very low volume: delay={delay3:.1f}s, X={X3:.2f}, LOS={determine_los(delay3)}")
    
    print("test_compute_control_delay PASSED")


def test_intersection_los():
    """Test intersection-level LOS calculation."""
    per_lane = [
        {'movement': 'NBT', 'volume_vph': 400, 'delay_s_per_veh': 20},
        {'movement': 'SBT', 'volume_vph': 600, 'delay_s_per_veh': 30},
        {'movement': 'EBT', 'volume_vph': 500, 'delay_s_per_veh': 25},
    ]
    
    result = compute_intersection_los(per_lane)
    
    # Weighted average: (400*20 + 600*30 + 500*25) / (400+600+500) = 38500/1500 = 25.67
    expected_avg = (400*20 + 600*30 + 500*25) / 1500
    
    assert abs(result['average_delay_s_per_veh'] - expected_avg) < 0.1
    assert result['LOS'] == 'C'  # 25.67 is in C range (20-35)
    assert result['total_volume_vph'] == 1500
    print(f"  Intersection avg delay: {result['average_delay_s_per_veh']:.1f}s, LOS: {result['LOS']}")
    print("test_intersection_los PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    test_parse_phase_label()
    test_determine_los()
    test_compute_control_delay()
    test_intersection_los()
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """
    Command-line interface for LOS computation.
    
    Usage:
        python LOS.py <phase_csv> <volume_csv>
        python LOS.py --test
        python LOS.py --dir <phase_dir> <volume_dir>
    
    Examples:
        python LOS.py ./data/times/102_A.csv ./data/volume/102_A.csv
        python LOS.py --test
        python LOS.py --dir ./data/times ./data/volume
    """
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python LOS.py <phase_csv> <volume_csv>")
        print("  python LOS.py --test")
        print("  python LOS.py --dir <phase_dir> <volume_dir>")
        sys.exit(1)
    
    if sys.argv[1] == '--test':
        run_all_tests()
        sys.exit(0)
    
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 4:
            print("Usage: python LOS.py --dir <phase_dir> <volume_dir>")
            sys.exit(1)
        
        phase_dir = sys.argv[2]
        volume_dir = sys.argv[3]
        
        result = compute_los_for_directories(
            phase_dir=phase_dir,
            volume_dir=volume_dir
        )
        
        print(json.dumps(result, indent=2))
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("Usage: python LOS.py <phase_csv> <volume_csv>")
        sys.exit(1)
    
    phase_csv = sys.argv[1]
    volume_csv = sys.argv[2]
    
    result = compute_los_for_intersection(
        phase_csv=phase_csv,
        volume_csv=volume_csv
    )
    
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
