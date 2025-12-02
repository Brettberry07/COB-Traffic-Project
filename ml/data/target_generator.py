"""
Target Generator Module - Generate optimal timing targets for ML training.

This module addresses the critical issue of training on static targets by:
1. Using HCM2010 optimization to find theoretical optimal timings
2. Exploring timing variations via grid search
3. Creating optimal targets that vary based on traffic conditions

Without this, the model learns to predict current (suboptimal) timings
rather than learning to improve them.
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from los_wrapper import LOSWrapper
from models.hcm2010 import HCM2010Optimizer, extract_phase_movements


class OptimalTargetGenerator:
    """
    Generate optimal timing targets for ML training.
    
    Instead of training on static current timings, this generates
    optimal timings for each traffic condition using:
    1. HCM2010 Webster optimization
    2. Local search around optimal
    3. Constraint validation
    """
    
    def __init__(
        self,
        saturation_flow: int = 1900,
        search_increments: List[int] = None,
        max_search_iterations: int = 20
    ):
        """
        Initialize the target generator.
        
        Args:
            saturation_flow: Saturation flow rate (veh/hr/lane)
            search_increments: Green time adjustment increments to try
            max_search_iterations: Maximum iterations for local search
        """
        self.optimizer = HCM2010Optimizer(saturation_flow=saturation_flow)
        self.los_wrapper = LOSWrapper(saturation_flow=saturation_flow)
        self.search_increments = search_increments or [1, 2, 3, 5]
        self.max_search_iterations = max_search_iterations
    
    def compute_optimal_for_volumes(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_timing: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute optimal timing for given traffic volumes.
        
        Uses current timing as base, then performs local search to minimize delay.
        This is more reliable than pure Webster optimization which doesn't
        account for dual-ring coordination constraints.
        
        Args:
            volumes: Dict mapping movements to volumes (vph)
            phase_movements: Dict mapping phases to movements
            current_timing: Optional current timing for comparison and base
            
        Returns:
            Dict with optimal timing and metrics
        """
        # Start from current timing if available
        if current_timing:
            best_plan = {
                'cycle_length': current_timing.get('cycle_length', 120),
                'phase_greens': current_timing.get('phase_greens', {}).copy()
            }
        else:
            # Get HCM2010 as starting point
            hcm_result = self.optimizer.optimize_timing_plan(
                volumes=volumes,
                phase_movements=phase_movements
            )
            best_plan = {
                'cycle_length': hcm_result['cycle_length'],
                'phase_greens': hcm_result['phase_greens'].copy()
            }
        
        # Evaluate starting point
        start_eval = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=best_plan['cycle_length'],
            phase_greens=best_plan['phase_greens']
        )
        best_delay = start_eval['intersection']['average_delay_s_per_veh']
        
        # Local search to improve
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_search_iterations:
            improved = False
            iteration += 1
            
            for phase in best_plan['phase_greens'].keys():
                for increment in self.search_increments:
                    # Try increasing green
                    test_greens = best_plan['phase_greens'].copy()
                    new_green = min(90, test_greens[phase] + increment)
                    test_greens[phase] = new_green
                    
                    eval_result = self.los_wrapper.evaluate_timing_plan(
                        volumes=volumes,
                        cycle_length=best_plan['cycle_length'],
                        phase_greens=test_greens
                    )
                    
                    new_delay = eval_result['intersection']['average_delay_s_per_veh']
                    
                    if new_delay < best_delay - 0.1:  # Require meaningful improvement
                        best_plan['phase_greens'] = test_greens
                        best_delay = new_delay
                        improved = True
                        break
                    
                    # Try decreasing green
                    test_greens = best_plan['phase_greens'].copy()
                    new_green = max(7, test_greens[phase] - increment)
                    test_greens[phase] = new_green
                    
                    eval_result = self.los_wrapper.evaluate_timing_plan(
                        volumes=volumes,
                        cycle_length=best_plan['cycle_length'],
                        phase_greens=test_greens
                    )
                    
                    new_delay = eval_result['intersection']['average_delay_s_per_veh']
                    
                    if new_delay < best_delay - 0.1:
                        best_plan['phase_greens'] = test_greens
                        best_delay = new_delay
                        improved = True
                        break
                
                if improved:
                    break
        
        # Also try cycle length adjustments
        for cycle_adj in [-10, -5, 5, 10]:
            test_cycle = max(60, min(180, best_plan['cycle_length'] + cycle_adj))
            
            eval_result = self.los_wrapper.evaluate_timing_plan(
                volumes=volumes,
                cycle_length=test_cycle,
                phase_greens=best_plan['phase_greens']
            )
            
            new_delay = eval_result['intersection']['average_delay_s_per_veh']
            
            if new_delay < best_delay - 0.1:
                best_plan['cycle_length'] = test_cycle
                best_delay = new_delay
        
        # Final evaluation
        final_eval = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=best_plan['cycle_length'],
            phase_greens=best_plan['phase_greens']
        )
        
        # Compare to current if provided
        improvement_vs_current = 0
        if current_timing:
            current_eval = self.los_wrapper.evaluate_timing_plan(
                volumes=volumes,
                cycle_length=current_timing.get('cycle_length', 120),
                phase_greens=current_timing.get('phase_greens', {})
            )
            current_delay = current_eval['intersection']['average_delay_s_per_veh']
            if current_delay > 0:
                improvement_vs_current = (current_delay - best_delay) / current_delay * 100
        
        return {
            'cycle_length': best_plan['cycle_length'],
            'phase_greens': best_plan['phase_greens'],
            'delay': best_delay,
            'los': final_eval['intersection']['LOS'],
            'evaluation': final_eval,
            'improvement_vs_current': improvement_vs_current,
            'search_iterations': iteration
        }
    
    def generate_training_targets(
        self,
        volume_df: pd.DataFrame,
        movement_cols: List[str],
        phase_movements: Dict[str, List[str]],
        current_timing: Optional[Dict[str, Any]] = None,
        sample_rate: float = 1.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate optimal timing targets for each row of volume data.
        
        Args:
            volume_df: DataFrame with traffic volumes
            movement_cols: List of movement column names
            phase_movements: Dict mapping phases to movements
            current_timing: Current timing for comparison
            sample_rate: Fraction of rows to compute (1.0 = all)
            
        Returns:
            Tuple of (targets_df, metadata)
        """
        phase_names = list(phase_movements.keys())
        n_samples = len(volume_df)
        
        # Optionally sample for faster computation
        if sample_rate < 1.0:
            sample_indices = np.random.choice(
                n_samples, 
                size=int(n_samples * sample_rate),
                replace=False
            )
            sample_indices = np.sort(sample_indices)
        else:
            sample_indices = np.arange(n_samples)
        
        # Initialize target arrays
        targets = {phase: np.zeros(n_samples) for phase in phase_names}
        targets['optimal_cycle'] = np.zeros(n_samples)
        targets['optimal_delay'] = np.zeros(n_samples)
        
        # Track statistics
        total_improvement = 0
        computed_count = 0
        
        print(f"Generating optimal targets for {len(sample_indices)} samples...")
        
        for idx in sample_indices:
            row = volume_df.iloc[idx]
            
            # Extract volumes for this interval
            volumes = {col: row.get(col, 0) for col in movement_cols if col in row.index}
            
            # Skip if no significant volume
            total_vol = sum(volumes.values())
            if total_vol < 10:
                # Use minimum greens for very low volume
                for phase in phase_names:
                    targets[phase][idx] = 7.0
                targets['optimal_cycle'][idx] = 60.0
                targets['optimal_delay'][idx] = 5.0
                continue
            
            # Compute optimal timing
            optimal = self.compute_optimal_for_volumes(
                volumes=volumes,
                phase_movements=phase_movements,
                current_timing=current_timing
            )
            
            # Store targets
            for phase in phase_names:
                targets[phase][idx] = optimal['phase_greens'].get(phase, 15.0)
            
            targets['optimal_cycle'][idx] = optimal['cycle_length']
            targets['optimal_delay'][idx] = optimal['delay']
            
            total_improvement += optimal['improvement_vs_current']
            computed_count += 1
            
            if computed_count % 50 == 0:
                print(f"  Computed {computed_count}/{len(sample_indices)} optimal timings...")
        
        # Fill unsampled rows with interpolation or nearest neighbor
        if sample_rate < 1.0:
            for col_name in list(targets.keys()):
                col = targets[col_name]
                mask = np.isin(np.arange(n_samples), sample_indices)
                
                # Simple forward/backward fill for non-computed rows
                for i in range(n_samples):
                    if not mask[i]:
                        # Find nearest computed value
                        dists = np.abs(sample_indices - i)
                        nearest_idx = sample_indices[np.argmin(dists)]
                        col[i] = col[nearest_idx]
        
        targets_df = pd.DataFrame(targets)
        
        metadata = {
            'n_samples': n_samples,
            'n_computed': computed_count,
            'sample_rate': sample_rate,
            'avg_improvement_vs_current': total_improvement / max(1, computed_count),
            'phase_names': phase_names
        }
        
        return targets_df, metadata
    
    def generate_augmented_targets(
        self,
        volume_df: pd.DataFrame,
        movement_cols: List[str],
        phase_movements: Dict[str, List[str]],
        current_timing: Optional[Dict[str, Any]] = None,
        volume_scaling_factors: List[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generate targets with data augmentation via volume scaling.
        
        Creates additional training samples by scaling volumes to simulate
        different traffic conditions.
        
        Args:
            volume_df: Original volume DataFrame
            movement_cols: Movement column names
            phase_movements: Phase to movement mapping
            current_timing: Current timing for comparison
            volume_scaling_factors: List of scaling factors (e.g., [0.7, 0.85, 1.0, 1.15, 1.3])
            
        Returns:
            Tuple of (augmented_features_df, augmented_targets_df, metadata)
        """
        scaling_factors = volume_scaling_factors or [0.8, 0.9, 1.0, 1.1, 1.2]
        
        augmented_features = []
        augmented_targets = []
        
        for scale in scaling_factors:
            # Scale volume columns
            scaled_df = volume_df.copy()
            for col in movement_cols:
                if col in scaled_df.columns:
                    scaled_df[col] = scaled_df[col] * scale
            
            # Add scale factor as feature
            scaled_df['volume_scale_factor'] = scale
            
            # Generate optimal targets for scaled volumes
            targets, meta = self.generate_training_targets(
                volume_df=scaled_df,
                movement_cols=movement_cols,
                phase_movements=phase_movements,
                current_timing=current_timing,
                sample_rate=0.5 if scale != 1.0 else 1.0  # Full sample for original, partial for augmented
            )
            
            augmented_features.append(scaled_df)
            augmented_targets.append(targets)
        
        combined_features = pd.concat(augmented_features, ignore_index=True)
        combined_targets = pd.concat(augmented_targets, ignore_index=True)
        
        metadata = {
            'original_samples': len(volume_df),
            'augmented_samples': len(combined_features),
            'scaling_factors': scaling_factors,
            'augmentation_ratio': len(combined_features) / len(volume_df)
        }
        
        return combined_features, combined_targets, metadata


class VolumeConditionClassifier:
    """
    Classify traffic volume conditions to help model specialize.
    
    Instead of one model for all conditions, this allows training
    condition-specific models or using condition as a feature.
    """
    
    CONDITION_LABELS = {
        'very_low': 0,
        'low': 1,
        'moderate': 2,
        'high': 3,
        'very_high': 4,
        'saturated': 5
    }
    
    def __init__(
        self,
        saturation_flow: int = 1900,
        lanes_per_approach: int = 2
    ):
        """
        Initialize classifier.
        
        Args:
            saturation_flow: Saturation flow per lane
            lanes_per_approach: Average lanes per approach
        """
        self.saturation_flow = saturation_flow
        self.lanes_per_approach = lanes_per_approach
        self.capacity_per_approach = saturation_flow * lanes_per_approach * 0.5  # Assume 50% green
    
    def classify_volume(self, total_volume: float) -> str:
        """
        Classify total intersection volume into condition categories.
        
        Args:
            total_volume: Total vehicles per hour across all movements
            
        Returns:
            Condition label string
        """
        # Approximate intersection capacity (4 approaches)
        intersection_capacity = self.capacity_per_approach * 4
        
        v_c_ratio = total_volume / intersection_capacity
        
        if v_c_ratio < 0.3:
            return 'very_low'
        elif v_c_ratio < 0.5:
            return 'low'
        elif v_c_ratio < 0.7:
            return 'moderate'
        elif v_c_ratio < 0.85:
            return 'high'
        elif v_c_ratio < 1.0:
            return 'very_high'
        else:
            return 'saturated'
    
    def add_condition_features(
        self,
        df: pd.DataFrame,
        movement_cols: List[str]
    ) -> pd.DataFrame:
        """
        Add volume condition features to DataFrame.
        
        Args:
            df: DataFrame with volume columns
            movement_cols: List of movement column names
            
        Returns:
            DataFrame with condition features added
        """
        df = df.copy()
        
        # Calculate total volume if not present
        available_cols = [c for c in movement_cols if c in df.columns]
        if 'total_volume' not in df.columns:
            df['total_volume'] = df[available_cols].sum(axis=1)
        
        # Classify each row
        df['volume_condition'] = df['total_volume'].apply(self.classify_volume)
        df['volume_condition_code'] = df['volume_condition'].map(self.CONDITION_LABELS)
        
        # Calculate v/c ratio feature
        intersection_capacity = self.capacity_per_approach * 4
        df['vc_ratio'] = df['total_volume'] / intersection_capacity
        df['vc_ratio'] = df['vc_ratio'].clip(upper=1.5)  # Cap extreme values
        
        # Calculate directional imbalance features
        if all(c in df.columns for c in ['NBT', 'SBT']):
            df['ns_imbalance'] = abs(df['NBT'] - df['SBT']) / (df['NBT'] + df['SBT'] + 1)
        
        if all(c in df.columns for c in ['EBT', 'WBT']):
            df['ew_imbalance'] = abs(df['EBT'] - df['WBT']) / (df['EBT'] + df['WBT'] + 1)
        
        # Major road dominance (assuming EB/WB is major road)
        if all(c in available_cols for c in ['EBT', 'WBT', 'NBT', 'SBT']):
            major = df[['EBT', 'WBT']].sum(axis=1) if 'EBT' in df.columns else 0
            minor = df[['NBT', 'SBT']].sum(axis=1) if 'NBT' in df.columns else 0
            df['major_road_ratio'] = major / (major + minor + 1)
        
        # Left turn demand relative to through
        left_cols = [c for c in available_cols if c.endswith('L')]
        through_cols = [c for c in available_cols if c.endswith('T')]
        
        if left_cols and through_cols:
            total_left = df[left_cols].sum(axis=1)
            total_through = df[through_cols].sum(axis=1)
            df['left_turn_ratio'] = total_left / (total_through + 1)
        
        return df


if __name__ == '__main__':
    # Test the target generator
    print("Testing OptimalTargetGenerator...")
    
    generator = OptimalTargetGenerator()
    
    # Sample volumes
    volumes = {
        'NBL': 100, 'NBT': 400, 'NBR': 80,
        'SBL': 120, 'SBT': 450, 'SBR': 90,
        'EBL': 80, 'EBT': 800, 'EBR': 60,
        'WBL': 70, 'WBT': 750, 'WBR': 50
    }
    
    phase_labels = ['1 EBLT', '2WB', '3NBLT', '4SB', '5WBLT', '6 EB', '7SBLT', '8NB']
    phase_movements = extract_phase_movements(phase_labels)
    
    current_timing = {
        'cycle_length': 120,
        'phase_greens': {
            '1 EBLT': 15, '2WB': 77, '3NBLT': 15, '4SB': 33,
            '5WBLT': 15, '6 EB': 77, '7SBLT': 17, '8NB': 31
        }
    }
    
    print("\nComputing optimal timing...")
    optimal = generator.compute_optimal_for_volumes(
        volumes=volumes,
        phase_movements=phase_movements,
        current_timing=current_timing
    )
    
    print(f"\nOptimal cycle: {optimal['cycle_length']}s")
    print(f"Optimal LOS: {optimal['los']}")
    print(f"Optimal delay: {optimal['delay']:.1f}s")
    print(f"Improvement vs current: {optimal['improvement_vs_current']:.1f}%")
    
    print("\nOptimal phase greens:")
    for phase, green in optimal['phase_greens'].items():
        print(f"  {phase}: {green}s")
