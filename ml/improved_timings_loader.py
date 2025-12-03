"""
Improved Timings Loader - Load and use optimized timing plans.

This module provides utilities to:
1. Load improved timing plans from the data/improved_timings directory
2. Calculate LOS using improved timings
3. Compare improved vs current timing performance
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np


# Default directory for improved timings
IMPROVED_TIMINGS_DIR = Path("data/improved_timings")


def get_available_intersections() -> List[str]:
    """
    Get list of intersections that have improved timing files.
    
    Returns:
        List of intersection IDs
    """
    if not IMPROVED_TIMINGS_DIR.exists():
        return []
    
    intersections = []
    for item in IMPROVED_TIMINGS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            intersections.append(item.name)
    
    return sorted(intersections)


def get_available_plans(intersection_id: str) -> List[int]:
    """
    Get list of plan numbers that have improved timings for an intersection.
    
    Args:
        intersection_id: Intersection identifier
        
    Returns:
        List of plan numbers (e.g., [61, 64])
    """
    int_dir = IMPROVED_TIMINGS_DIR / intersection_id
    if not int_dir.exists():
        return []
    
    plans = []
    for item in int_dir.iterdir():
        if item.name.startswith('plan_') and item.name.endswith('_improved.json'):
            try:
                plan_num = int(item.name.split('_')[1])
                plans.append(plan_num)
            except (ValueError, IndexError):
                continue
    
    return sorted(plans)


def load_improved_timing(
    intersection_id: str,
    plan_number: int
) -> Optional[Dict[str, Any]]:
    """
    Load improved timing for a specific intersection and plan.
    
    Args:
        intersection_id: Intersection identifier
        plan_number: Plan number (e.g., 61, 64)
        
    Returns:
        Dict with improved timing data or None if not found
    """
    filepath = IMPROVED_TIMINGS_DIR / intersection_id / f"plan_{plan_number}_improved.json"
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_improved_timings(intersection_id: str) -> Dict[int, Dict[str, Any]]:
    """
    Load all improved timings for an intersection.
    
    Args:
        intersection_id: Intersection identifier
        
    Returns:
        Dict mapping plan numbers to timing data
    """
    plans = get_available_plans(intersection_id)
    result = {}
    
    for plan_num in plans:
        timing = load_improved_timing(intersection_id, plan_num)
        if timing:
            result[plan_num] = timing
    
    return result


def get_improved_phase_greens(
    intersection_id: str,
    plan_number: int
) -> Optional[Dict[str, float]]:
    """
    Get just the improved phase green times for quick access.
    
    Args:
        intersection_id: Intersection identifier
        plan_number: Plan number
        
    Returns:
        Dict mapping phase names to green times in seconds
    """
    timing = load_improved_timing(intersection_id, plan_number)
    
    if timing is None:
        return None
    
    improved = timing.get('improved_timing', {})
    return improved.get('phase_greens', {})


def get_improved_cycle_length(
    intersection_id: str,
    plan_number: int
) -> Optional[float]:
    """
    Get just the improved cycle length for quick access.
    
    Args:
        intersection_id: Intersection identifier
        plan_number: Plan number
        
    Returns:
        Cycle length in seconds or None
    """
    timing = load_improved_timing(intersection_id, plan_number)
    
    if timing is None:
        return None
    
    improved = timing.get('improved_timing', {})
    return improved.get('cycle_length')


def get_improvement_summary(
    intersection_id: str,
    plan_number: int
) -> Optional[Dict[str, Any]]:
    """
    Get a summary of improvements for a specific plan.
    
    Args:
        intersection_id: Intersection identifier
        plan_number: Plan number
        
    Returns:
        Dict with improvement metrics
    """
    timing = load_improved_timing(intersection_id, plan_number)
    
    if timing is None:
        return None
    
    current = timing.get('current_timing', {})
    improved = timing.get('improved_timing', {})
    improvement = timing.get('improvement', {})
    
    return {
        'intersection_id': intersection_id,
        'plan_number': plan_number,
        'plan_name': timing.get('plan_name'),
        'time_range': timing.get('time_range'),
        'current_delay': current.get('delay'),
        'current_los': current.get('los'),
        'improved_delay': improved.get('delay'),
        'improved_los': improved.get('los'),
        'delay_reduction_s': improvement.get('delay_reduction_s'),
        'delay_reduction_pct': improvement.get('delay_reduction_pct')
    }


def load_pm_optimization_summary() -> Optional[Dict[str, Any]]:
    """
    Load the overall PM optimization summary report.
    
    Returns:
        Summary dict or None if not found
    """
    summary_path = IMPROVED_TIMINGS_DIR / 'pm_optimization_summary.json'
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def create_comparison_dataframe() -> pd.DataFrame:
    """
    Create a DataFrame comparing current vs improved timings for all intersections.
    
    Returns:
        DataFrame with comparison metrics
    """
    records = []
    
    for int_id in get_available_intersections():
        for plan_num in get_available_plans(int_id):
            summary = get_improvement_summary(int_id, plan_num)
            if summary:
                records.append(summary)
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Sort by plan number then intersection
    df = df.sort_values(['plan_number', 'intersection_id']).reset_index(drop=True)
    
    return df


def print_improvement_report():
    """Print a formatted report of all improvements."""
    summary = load_pm_optimization_summary()
    
    if summary is None:
        print("No optimization summary found. Run train_pm_plans.py first.")
        return
    
    print("="*70)
    print("PM TIMING PLAN IMPROVEMENTS REPORT")
    print("="*70)
    print(f"Generated: {summary.get('generated_at', 'Unknown')}")
    print(f"Intersections: {summary.get('total_intersections', 0)}")
    print()
    
    # Aggregate stats
    agg = summary.get('aggregate_stats', {})
    
    for plan_num in [61, 64]:
        plan_key = f'plan_{plan_num}'
        stats = agg.get(plan_key, {})
        
        if stats.get('count', 0) > 0:
            print(f"Plan {plan_num}:")
            print(f"  Optimized: {stats['count']} intersections")
            print(f"  Avg improvement: {stats.get('avg_improvement_s', 0):.1f}s")
            print(f"  Total reduction: {stats.get('total_improvement_s', 0):.1f}s")
            print()
    
    # Per-intersection details
    print("-"*70)
    print("Intersection Details:")
    print("-"*70)
    print(f"{'Intersection':<16} {'Plan':<8} {'Current':<15} {'Improved':<15} {'Change':<12}")
    print(f"{'':16} {'':8} {'Delay   LOS':<15} {'Delay   LOS':<15} {'(seconds)':<12}")
    print("-"*70)
    
    for int_id, int_data in summary.get('intersections', {}).items():
        if 'error' in int_data:
            print(f"{int_id:<16} Error: {int_data['error']}")
            continue
        
        for plan_num, plan_data in int_data.get('plans', {}).items():
            if 'error' in plan_data:
                continue
            
            curr_delay = plan_data.get('current_delay', 0)
            curr_los = plan_data.get('current_los', '?')
            imp_delay = plan_data.get('improved_delay', 0)
            imp_los = plan_data.get('improved_los', '?')
            change = plan_data.get('improvement_s', 0)
            
            print(f"{int_id:<16} {plan_num:<8} "
                  f"{curr_delay:>5.1f}s   {curr_los:<4} "
                  f"{imp_delay:>5.1f}s   {imp_los:<4} "
                  f"{change:>+6.1f}s")


# Example usage
if __name__ == '__main__':
    print("Improved Timings Loader")
    print("="*50)
    
    intersections = get_available_intersections()
    
    if not intersections:
        print("No improved timings found.")
        print("Run 'python ml/models/train_pm_plans.py' to generate them.")
    else:
        print(f"Found improved timings for {len(intersections)} intersections:")
        for int_id in intersections:
            plans = get_available_plans(int_id)
            print(f"  {int_id}: Plans {plans}")
        
        print()
        print_improvement_report()
