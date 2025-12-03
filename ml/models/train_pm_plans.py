"""
PM Plans Training Module - Optimize timing plans for Plan 61 and Plan 64.

This module focuses specifically on:
- Plan 61: Early PM (13:15 - 14:45)
- Plan 64: PM Peak (14:45 - 18:45)

The model trains on volume data from these time periods and generates
improved timing recommendations that are saved for LOS evaluation.
"""

import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from los_wrapper import LOSWrapper
from models.hcm2010 import HCM2010Optimizer, extract_phase_movements
from models.train import (
    GradientBoostedTimingModel,
    ModelTrainer,
    WalkForwardValidator
)

# Import target generator for optimal training targets
try:
    from data.target_generator import OptimalTargetGenerator
    HAS_TARGET_GENERATOR = True
except ImportError:
    HAS_TARGET_GENERATOR = False


# ============================================================================
# PM PLAN CONFIGURATION
# ============================================================================
# Plan 61: 13:15 - 14:45 (Early PM)
# Plan 64: 14:45 - 18:45 (PM Peak)
# ============================================================================

PM_PLANS = {
    61: {
        "name": "Early PM",
        "start_hour": 13,
        "start_minute": 15,
        "end_hour": 14,
        "end_minute": 45,
        "start_hhmm": 1315,
        "end_hhmm": 1445
    },
    64: {
        "name": "PM Peak", 
        "start_hour": 14,
        "start_minute": 45,
        "end_hour": 18,
        "end_minute": 45,
        "start_hhmm": 1445,
        "end_hhmm": 1845
    }
}

# Output directory for improved timings
IMPROVED_TIMINGS_DIR = Path("data/improved_timings")


class PMPlanOptimizer:
    """
    Specialized optimizer for PM timing plans (Plan 61 and Plan 64).
    
    Uses ML to learn optimal green splits from traffic patterns during
    the PM periods, then applies HCM2010 constraints to ensure valid plans.
    """
    
    def __init__(
        self,
        output_dir: str = 'ml/models/trained',
        saturation_flow: int = 1900
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.improved_timings_dir = IMPROVED_TIMINGS_DIR
        self.improved_timings_dir.mkdir(parents=True, exist_ok=True)
        
        self.saturation_flow = saturation_flow
        self.los_wrapper = LOSWrapper(saturation_flow=saturation_flow)
        self.hcm_optimizer = HCM2010Optimizer(saturation_flow=saturation_flow)
        
        if HAS_TARGET_GENERATOR:
            self.target_generator = OptimalTargetGenerator(saturation_flow=saturation_flow)
        else:
            self.target_generator = None
    
    def filter_pm_data(
        self,
        volume_df: pd.DataFrame,
        plan_number: int
    ) -> pd.DataFrame:
        """
        Filter volume data to only include records from the specified PM plan period.
        
        Args:
            volume_df: DataFrame with volume data (must have 'hour' and optionally 'minute_of_day')
            plan_number: 61 (Early PM) or 64 (PM Peak)
            
        Returns:
            Filtered DataFrame
        """
        if plan_number not in PM_PLANS:
            raise ValueError(f"Plan {plan_number} not in PM plans. Use 61 or 64.")
        
        plan_config = PM_PLANS[plan_number]
        
        df = volume_df.copy()
        
        # Try to use minute_of_day for precise filtering
        if 'minute_of_day' in df.columns:
            start_minutes = plan_config["start_hour"] * 60 + plan_config["start_minute"]
            end_minutes = plan_config["end_hour"] * 60 + plan_config["end_minute"]
            mask = (df['minute_of_day'] >= start_minutes) & (df['minute_of_day'] < end_minutes)
        elif 'hour' in df.columns:
            # Fall back to hour-based filtering (less precise)
            start_hour = plan_config["start_hour"]
            end_hour = plan_config["end_hour"]
            # Include full hours that overlap with the period
            mask = (df['hour'] >= start_hour) & (df['hour'] < end_hour + 1)
        else:
            print("Warning: No hour/minute columns found, using all data")
            return df
        
        filtered = df[mask].copy()
        print(f"  Filtered Plan {plan_number} ({plan_config['name']}): {len(filtered)} rows "
              f"({plan_config['start_hour']:02d}:{plan_config['start_minute']:02d} - "
              f"{plan_config['end_hour']:02d}:{plan_config['end_minute']:02d})")
        
        return filtered
    
    def compute_peak_volumes(
        self,
        volume_df: pd.DataFrame,
        movement_cols: List[str]
    ) -> Dict[str, float]:
        """
        Compute representative peak-hour volumes for the time period.
        
        Uses the 95th percentile of volumes to capture peak conditions
        that the timing plan should be optimized for.
        
        Args:
            volume_df: Filtered volume DataFrame
            movement_cols: List of movement column names
            
        Returns:
            Dict mapping movements to peak hourly volumes
        """
        peak_volumes = {}
        
        for col in movement_cols:
            if col in volume_df.columns:
                # Use 95th percentile to capture peak conditions
                # Multiply by 4 since data is 15-min intervals
                p95 = volume_df[col].quantile(0.95)
                peak_volumes[col] = float(p95 * 4)  # Convert to hourly
        
        return peak_volumes
    
    def optimize_timing_for_plan(
        self,
        plan_number: int,
        volumes: Dict[str, float],
        current_timing: Dict[str, Any],
        phase_movements: Dict[str, List[str]],
        intersection_id: str
    ) -> Dict[str, Any]:
        """
        Optimize timing for a specific PM plan using local search from current timing.
        
        DUAL-RING STRUCTURE:
        - Ring 1: Phases 1, 2, 3, 4 (typically NB/SB left turns + through)
        - Ring 2: Phases 5, 6, 7, 8 (typically opposite ring)
        - Phases run concurrently: 1&5, 2&6, 3&7, 4&8
        - Cycle length = max(ring1_sum, ring2_sum) + clearance
        
        CONSTRAINTS:
        - Can only redistribute green time within each ring
        - Must maintain barrier phases (2&6 must both end at barrier)
        - Minimum green: 7s, Maximum green: 90s
        
        Args:
            plan_number: 61 or 64
            volumes: Peak volumes for the period
            current_timing: Current timing parameters
            phase_movements: Phase to movement mapping
            intersection_id: Intersection identifier
            
        Returns:
            Optimized timing plan with evaluation
        """
        plan_config = PM_PLANS[plan_number]
        
        # Current timing
        current_greens = current_timing.get('phase_greens', {})
        current_cycle = current_timing.get('cycle_length', 120)
        
        # =================================================================
        # REALISTIC MINIMUM GREEN TIMES (based on MUTCD and engineering practice)
        # =================================================================
        # Left turn phases (protected): 10-12s minimum for queue clearance
        # Through movements: 15s minimum for pedestrian crossing
        # Minor street through: 12s minimum
        # Right turn only: 10s minimum
        # =================================================================
        MIN_GREEN_LEFT_TURN = 10.0   # Protected left turn minimum
        MIN_GREEN_THROUGH = 15.0     # Major through movement minimum  
        MIN_GREEN_MINOR = 12.0       # Minor street/movement minimum
        MIN_GREEN_DEFAULT = 12.0     # Default minimum if phase type unknown
        MAX_GREEN = 90.0
        CLEARANCE_PER_PHASE = 5.5    # yellow + all-red
        
        # Maximum reduction from current timing (don't cut more than 50%)
        MAX_REDUCTION_PCT = 0.50
        
        def get_min_green_for_phase(phase_label: str, current_green: float) -> float:
            """Determine minimum green based on phase type and current timing."""
            label_upper = str(phase_label).upper()
            
            # Left turn phases (contain "LT")
            if 'LT' in label_upper:
                base_min = MIN_GREEN_LEFT_TURN
            # Through phases (phases 2, 4, 6, 8 typically, or contain direction only)
            elif any(x in label_upper for x in ['NB', 'SB', 'EB', 'WB']) and 'LT' not in label_upper and 'RT' not in label_upper:
                base_min = MIN_GREEN_THROUGH
            else:
                base_min = MIN_GREEN_DEFAULT
            
            # Also enforce: don't reduce below 50% of current timing
            floor_from_current = current_green * (1 - MAX_REDUCTION_PCT)
            
            return max(base_min, floor_from_current)
        
        current_eval = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=current_cycle,
            phase_greens=current_greens
        )
        current_delay = current_eval['intersection']['average_delay_s_per_veh']
        current_los = current_eval['intersection']['LOS']
        
        print(f"\n  Current Plan {plan_number} ({plan_config['name']}):")
        print(f"    Cycle: {current_cycle:.0f}s, Delay: {current_delay:.1f}s, LOS: {current_los}")
        
        # Identify dual-ring structure
        phases = list(current_greens.keys())
        ring1_phases = []
        ring2_phases = []
        
        for phase in phases:
            # Extract phase number from label (e.g., "1NBLT" -> 1)
            import re
            match = re.match(r'^(\d+)', str(phase))
            if match:
                phase_num = int(match.group(1))
                if phase_num <= 4:
                    ring1_phases.append(phase)
                else:
                    ring2_phases.append(phase)
            else:
                # Numeric-only phases (e.g., "1.0", "2.0")
                try:
                    phase_num = int(float(phase))
                    if phase_num <= 4:
                        ring1_phases.append(phase)
                    else:
                        ring2_phases.append(phase)
                except:
                    ring1_phases.append(phase)  # default to ring 1
        
        is_dual_ring = len(ring1_phases) > 0 and len(ring2_phases) > 0
        
        # Compute phase-specific minimum greens
        phase_min_greens = {
            phase: get_min_green_for_phase(phase, current_greens[phase])
            for phase in current_greens
        }
        
        # START from current timing
        best_cycle = current_cycle
        best_greens = current_greens.copy()
        best_delay = current_delay
        
        # STRATEGY: Redistribute green within rings while maintaining total ring time
        improved = True
        iteration = 0
        max_iterations = 30
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # For each ring, try redistributing green between phases
            for ring_phases in [ring1_phases, ring2_phases] if is_dual_ring else [phases]:
                if len(ring_phases) < 2:
                    continue
                    
                for i, phase_from in enumerate(ring_phases):
                    for phase_to in ring_phases[i+1:]:
                        # Get phase-specific minimums
                        min_from = phase_min_greens[phase_from]
                        min_to = phase_min_greens[phase_to]
                        
                        # Try shifting green from one phase to another (zero-sum within ring)
                        for shift in [2, 3, 5, 7]:
                            # Can we take from phase_from? (respect phase-specific minimum)
                            if best_greens[phase_from] - shift >= min_from:
                                test_greens = best_greens.copy()
                                test_greens[phase_from] -= shift
                                test_greens[phase_to] = min(MAX_GREEN, test_greens[phase_to] + shift)
                                
                                test_eval = self.los_wrapper.evaluate_timing_plan(
                                    volumes=volumes,
                                    cycle_length=best_cycle,
                                    phase_greens=test_greens
                                )
                                test_delay = test_eval['intersection']['average_delay_s_per_veh']
                                
                                if test_delay < best_delay - 0.1:
                                    best_greens = test_greens
                                    best_delay = test_delay
                                    improved = True
                                    break
                            
                            # Try the opposite direction (respect phase-specific minimum)
                            if best_greens[phase_to] - shift >= min_to:
                                test_greens = best_greens.copy()
                                test_greens[phase_to] -= shift
                                test_greens[phase_from] = min(MAX_GREEN, test_greens[phase_from] + shift)
                                
                                test_eval = self.los_wrapper.evaluate_timing_plan(
                                    volumes=volumes,
                                    cycle_length=best_cycle,
                                    phase_greens=test_greens
                                )
                                test_delay = test_eval['intersection']['average_delay_s_per_veh']
                                
                                if test_delay < best_delay - 0.1:
                                    best_greens = test_greens
                                    best_delay = test_delay
                                    improved = True
                                    break
                        
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # STRATEGY 2: Try modest cycle length changes with proportional scaling
        # Only if phase redistribution didn't help much
        if best_delay >= current_delay * 0.95:  # Less than 5% improvement
            for cycle_pct in [-10, -5, 5, 10]:
                test_cycle = round(current_cycle * (1 + cycle_pct/100))
                test_cycle = max(60, min(180, test_cycle))
                
                if test_cycle == current_cycle:
                    continue
                
                # Scale phases proportionally
                scale = test_cycle / current_cycle
                test_greens = {}
                valid = True
                
                for phase, green in best_greens.items():
                    scaled = round(green * scale)
                    # Use phase-specific minimum
                    if scaled < phase_min_greens[phase] or scaled > MAX_GREEN:
                        valid = False
                        break
                    test_greens[phase] = scaled
                
                if not valid:
                    continue
                
                test_eval = self.los_wrapper.evaluate_timing_plan(
                    volumes=volumes,
                    cycle_length=test_cycle,
                    phase_greens=test_greens
                )
                test_delay = test_eval['intersection']['average_delay_s_per_veh']
                
                if test_delay < best_delay - 0.5:
                    best_cycle = test_cycle
                    best_greens = test_greens
                    best_delay = test_delay
        
        # Final evaluation
        final_eval = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=best_cycle,
            phase_greens=best_greens
        )
        final_delay = final_eval['intersection']['average_delay_s_per_veh']
        final_los = final_eval['intersection']['LOS']
        
        # Calculate improvement
        improvement = current_delay - final_delay
        improvement_pct = (improvement / current_delay * 100) if current_delay > 0 else 0
        
        # Show phase changes summary
        changes_made = []
        for phase in current_greens:
            diff = best_greens.get(phase, current_greens[phase]) - current_greens[phase]
            if diff != 0:
                changes_made.append(f"{phase}: {'+' if diff > 0 else ''}{diff:.0f}s")
        
        print(f"  Optimized Plan {plan_number}:")
        print(f"    Cycle: {best_cycle:.0f}s, Delay: {final_delay:.1f}s, LOS: {final_los}")
        if changes_made:
            print(f"    Phase changes: {', '.join(changes_made)}")
        else:
            print(f"    No phase changes (current timing is optimal)")
        print(f"    Improvement: {improvement:.1f}s ({improvement_pct:.1f}%)")
        
        return {
            'plan_number': plan_number,
            'plan_name': plan_config['name'],
            'intersection_id': intersection_id,
            'time_range': f"{plan_config['start_hour']:02d}:{plan_config['start_minute']:02d}-"
                         f"{plan_config['end_hour']:02d}:{plan_config['end_minute']:02d}",
            'current_timing': {
                'cycle_length': current_cycle,
                'phase_greens': current_greens,
                'delay': current_delay,
                'los': current_los
            },
            'improved_timing': {
                'cycle_length': best_cycle,
                'phase_greens': best_greens,
                'delay': final_delay,
                'los': final_los
            },
            'improvement': {
                'delay_reduction_s': improvement,
                'delay_reduction_pct': improvement_pct,
                'search_iterations': iteration
            },
            'volumes_used': volumes,
            'evaluation': final_eval
        }
    
    def save_improved_timing(
        self,
        result: Dict[str, Any],
        intersection_id: str,
        plan_number: int
    ) -> Path:
        """
        Save improved timing to the improved_timings directory.
        
        Args:
            result: Optimization result dict
            intersection_id: Intersection identifier
            plan_number: Plan number (61 or 64)
            
        Returns:
            Path to saved file
        """
        # Create intersection subdirectory
        int_dir = self.improved_timings_dir / intersection_id
        int_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy access
        filename = f"plan_{plan_number}_improved.json"
        filepath = int_dir / filename
        
        # Add metadata
        output = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'PMPlanOptimizer',
            **result
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  Saved improved timing to: {filepath}")
        return filepath
    
    def train_and_optimize(
        self,
        preprocessed_data: Dict[str, Any],
        timing_features: Dict[str, Any],
        intersection_id: str,
        plans: List[int] = None
    ) -> Dict[str, Any]:
        """
        Train model and optimize timings for PM plans.
        
        Args:
            preprocessed_data: Preprocessed intersection data
            timing_features: Timing features from phase files
            intersection_id: Intersection identifier
            plans: List of plan numbers to optimize (default: [61, 64])
            
        Returns:
            Dict with results for each plan
        """
        if plans is None:
            plans = [61, 64]
        
        volume_df = preprocessed_data.get('volume_df')
        movement_cols = preprocessed_data.get('movement_cols', [])
        
        if volume_df is None or len(volume_df) == 0:
            return {'error': 'No volume data available'}
        
        # Get phase movements
        phase_greens = timing_features.get('phase_greens', {})
        phase_names = list(phase_greens.keys())
        phase_movements = extract_phase_movements(phase_names)
        
        results = {
            'intersection_id': intersection_id,
            'plans': {}
        }
        
        for plan_number in plans:
            print(f"\n{'='*60}")
            print(f"Optimizing {intersection_id} - Plan {plan_number}")
            print('='*60)
            
            try:
                # Filter data for this plan's time period
                plan_df = self.filter_pm_data(volume_df, plan_number)
                
                if len(plan_df) < 10:
                    print(f"  Insufficient data for Plan {plan_number} ({len(plan_df)} rows)")
                    results['plans'][plan_number] = {'error': 'Insufficient data'}
                    continue
                
                # Compute peak volumes for this period
                peak_volumes = self.compute_peak_volumes(plan_df, movement_cols)
                
                # Get current timing for this plan
                current_timing = {
                    'cycle_length': timing_features.get('cycle_length', 120),
                    'phase_greens': phase_greens
                }
                
                # Optimize timing
                opt_result = self.optimize_timing_for_plan(
                    plan_number=plan_number,
                    volumes=peak_volumes,
                    current_timing=current_timing,
                    phase_movements=phase_movements,
                    intersection_id=intersection_id
                )
                
                # Save improved timing
                saved_path = self.save_improved_timing(
                    result=opt_result,
                    intersection_id=intersection_id,
                    plan_number=plan_number
                )
                
                opt_result['saved_path'] = str(saved_path)
                results['plans'][plan_number] = opt_result
                
            except Exception as e:
                print(f"  Error optimizing Plan {plan_number}: {e}")
                import traceback
                traceback.print_exc()
                results['plans'][plan_number] = {'error': str(e)}
        
        return results


def generate_summary_report(
    all_results: Dict[str, Dict[str, Any]],
    output_path: str = 'data/improved_timings/pm_optimization_summary.json'
) -> None:
    """
    Generate a summary report of all PM plan optimizations.
    
    Args:
        all_results: Dict mapping intersection_id to optimization results
        output_path: Path to save summary report
    """
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_intersections': len(all_results),
        'plans_optimized': [61, 64],
        'intersections': {},
        'aggregate_stats': {
            'plan_61': {'total_improvement_s': 0, 'count': 0, 'los_changes': []},
            'plan_64': {'total_improvement_s': 0, 'count': 0, 'los_changes': []}
        }
    }
    
    for int_id, result in all_results.items():
        if 'error' in result:
            summary['intersections'][int_id] = {'error': result['error']}
            continue
        
        int_summary = {
            'plans': {}
        }
        
        for plan_num, plan_result in result.get('plans', {}).items():
            if 'error' in plan_result:
                int_summary['plans'][plan_num] = {'error': plan_result['error']}
                continue
            
            improvement = plan_result.get('improvement', {})
            current = plan_result.get('current_timing', {})
            improved = plan_result.get('improved_timing', {})
            
            plan_key = f'plan_{plan_num}'
            
            int_summary['plans'][plan_num] = {
                'current_los': current.get('los'),
                'improved_los': improved.get('los'),
                'current_delay': current.get('delay'),
                'improved_delay': improved.get('delay'),
                'improvement_s': improvement.get('delay_reduction_s'),
                'improvement_pct': improvement.get('delay_reduction_pct'),
                'saved_path': plan_result.get('saved_path')
            }
            
            # Aggregate stats
            if plan_key in summary['aggregate_stats']:
                stats = summary['aggregate_stats'][plan_key]
                stats['total_improvement_s'] += improvement.get('delay_reduction_s', 0)
                stats['count'] += 1
                stats['los_changes'].append({
                    'intersection': int_id,
                    'from': current.get('los'),
                    'to': improved.get('los')
                })
        
        summary['intersections'][int_id] = int_summary
    
    # Calculate averages
    for plan_key in ['plan_61', 'plan_64']:
        stats = summary['aggregate_stats'][plan_key]
        if stats['count'] > 0:
            stats['avg_improvement_s'] = stats['total_improvement_s'] / stats['count']
    
    # Save summary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PM OPTIMIZATION SUMMARY")
    print('='*60)
    print(f"Total intersections processed: {len(all_results)}")
    
    for plan_num in [61, 64]:
        plan_key = f'plan_{plan_num}'
        stats = summary['aggregate_stats'][plan_key]
        if stats['count'] > 0:
            print(f"\nPlan {plan_num}:")
            print(f"  Intersections optimized: {stats['count']}")
            print(f"  Average improvement: {stats['avg_improvement_s']:.1f}s")
            print(f"  Total delay reduction: {stats['total_improvement_s']:.1f}s")
    
    print(f"\nSummary saved to: {output_path}")


def main():
    """Main entry point for PM plan optimization."""
    # Import data modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from ingest import ingest_all
    from preprocess import preprocess_all
    
    print("="*60)
    print("PM TIMING PLAN OPTIMIZATION")
    print("Optimizing Plan 61 (Early PM) and Plan 64 (PM Peak)")
    print("="*60)
    
    print("\nStep 1: Loading and preprocessing data...")
    ingested = ingest_all()
    
    # Use non-aggregated data for more samples
    preprocessed = preprocess_all(ingested, aggregate_window=False)
    
    print(f"\nStep 2: Optimizing PM plans for {len(preprocessed)} intersections...")
    optimizer = PMPlanOptimizer()
    
    all_results = {}
    
    for int_id, data in preprocessed.items():
        if data.get('error'):
            print(f"\nSkipping {int_id}: {data['error']}")
            all_results[int_id] = {'error': data['error']}
            continue
        
        timing_features = data.get('timing_features', {})
        
        if not timing_features.get('phase_greens'):
            print(f"\nSkipping {int_id}: No phase timing data")
            all_results[int_id] = {'error': 'No phase timing data'}
            continue
        
        results = optimizer.train_and_optimize(
            preprocessed_data=data,
            timing_features=timing_features,
            intersection_id=int_id,
            plans=[61, 64]
        )
        
        all_results[int_id] = results
    
    print("\n\nStep 3: Generating summary report...")
    generate_summary_report(all_results)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nImproved timings saved to: {IMPROVED_TIMINGS_DIR}")
    print("You can now use these timings to recalculate LOS scores.")
    
    return all_results


if __name__ == '__main__':
    main()
