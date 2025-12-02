"""
Prediction Module - Inference pipeline producing validated timing plans.

This module:
- Loads trained ML models
- Generates timing plan recommendations (single recommendation for 12pm-6pm window)
- Validates all plans using LOS.py
- Uses ML model exclusively for timing optimization
- Outputs results as NDJSON
"""

import sys
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from los_wrapper import LOSWrapper
from models.hcm2010 import extract_phase_movements  # Only for phase parsing
from models.train import (
    GradientBoostedTimingModel,
    SequenceTimingModel,
    HybridTimingModel
)

# Import timing optimizer for improved predictions
try:
    from models.timing_optimizer import TimingOptimizer, OptimizationStrategy
    HAS_OPTIMIZER = True
except ImportError:
    HAS_OPTIMIZER = False


class TimingPlanPredictor:
    """
    Generates validated timing plan recommendations using ML models.
    """
    
    def __init__(
        self,
        model_dir: str = 'ml/models/trained'
    ):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.los_wrapper = LOSWrapper()
        
        self.loaded_models = {}
    
    def load_model(
        self,
        intersection_id: str,
        model_type: str = 'gradient_boosted'
    ) -> Optional[Any]:
        """
        Load a trained model for an intersection.
        
        Args:
            intersection_id: Intersection identifier
            model_type: Type of model ('gradient_boosted', 'sequence', 'hybrid')
            
        Returns:
            Loaded model or None if not found
        """
        model_key = f"{intersection_id}_{model_type}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Map model type to file suffix
        suffix_map = {
            'gradient_boosted': 'gb',
            'sequence': 'seq',
            'hybrid': 'hybrid'
        }
        
        suffix = suffix_map.get(model_type, 'gb')
        model_path = os.path.join(self.model_dir, f'{intersection_id}_{suffix}.pkl')
        
        if not os.path.exists(model_path):
            return None
        
        try:
            if model_type == 'gradient_boosted':
                model = GradientBoostedTimingModel()
            elif model_type == 'sequence':
                model = SequenceTimingModel()
            elif model_type == 'hybrid':
                model = HybridTimingModel()
            else:
                model = GradientBoostedTimingModel()
            
            model.load(model_path)
            self.loaded_models[model_key] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None
    
    def predict_timing_plan(
        self,
        intersection_id: str,
        features: pd.DataFrame,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_timing: Dict[str, Any],
        model_type: str = 'gradient_boosted',
        use_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Predict timing plan using ML model with optional optimization.
        
        IMPROVEMENT: When use_optimizer=True, performs additional search
        optimization after ML prediction to find better solutions that the
        model might miss.
        
        Args:
            intersection_id: Intersection identifier
            features: Feature DataFrame for current interval
            volumes: Current traffic volumes
            phase_movements: Phase to movement mapping
            current_timing: Current timing parameters
            model_type: Type of model to use
            use_optimizer: If True, apply search optimization after ML
            
        Returns:
            Validated timing plan dict
        """
        # Get current LOS as baseline
        current_plan = {
            'cycle_length': current_timing.get('cycle_length', 120),
            'phase_greens': current_timing.get('phase_greens', {})
        }
        
        current_eval = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=current_plan['cycle_length'],
            phase_greens=current_plan['phase_greens']
        )
        
        current_delay = current_eval['intersection']['average_delay_s_per_veh']
        current_los = current_eval['intersection']['LOS']
        
        # Try ML model prediction
        model = self.load_model(intersection_id, model_type)
        
        ml_plan = None
        ml_eval = None
        
        if model is not None:
            try:
                if model_type == 'hybrid':
                    ml_result = model.predict(features, volumes, phase_movements)
                    ml_plan = {
                        'cycle_length': ml_result['cycle_length'],
                        'phase_greens': ml_result['phase_greens']
                    }
                    ml_eval = ml_result['evaluation']
                else:
                    # Get predictions
                    predictions = model.predict(features)
                    
                    # Average if multiple rows
                    avg_greens = predictions.mean().to_dict()
                    
                    # Apply constraints (min/max green times)
                    MIN_GREEN = 7
                    MAX_GREEN = 90
                    constrained_greens = {}
                    for phase, green in avg_greens.items():
                        constrained_greens[phase] = max(MIN_GREEN, min(MAX_GREEN, green))
                    
                    # Compute cycle length as sum of greens plus clearance times
                    total_green = sum(constrained_greens.values())
                    num_phases = len(constrained_greens)
                    avg_clearance = 5.0  # yellow + red clearance per phase
                    cycle_length = total_green + (num_phases * avg_clearance)
                    cycle_length = max(60, min(180, cycle_length))  # Constrain cycle
                    
                    ml_plan = {
                        'cycle_length': cycle_length,
                        'phase_greens': constrained_greens
                    }
                    
                    ml_eval = self.los_wrapper.evaluate_timing_plan(
                        volumes=volumes,
                        cycle_length=cycle_length,
                        phase_greens=constrained_greens
                    )
            except Exception as e:
                print(f"ML prediction error: {e}")
                ml_plan = None
        
        # Apply optimization to find improvements
        optimized_plan = None
        if use_optimizer and HAS_OPTIMIZER:
            optimizer = TimingOptimizer()
            
            # Use ML plan as starting point if available, otherwise current plan
            start_plan = ml_plan if ml_plan else current_plan
            
            opt_result = optimizer.optimize(
                volumes=volumes,
                current_plan=current_plan,
                ml_plan=ml_plan,
                phase_movements=phase_movements,
                strategy=OptimizationStrategy.LOCAL_SEARCH
            )
            
            if opt_result['delay'] < current_delay - 0.5:  # At least 0.5s improvement
                optimized_plan = {
                    'cycle_length': opt_result['cycle_length'],
                    'phase_greens': opt_result['phase_greens']
                }
                opt_eval = self.los_wrapper.evaluate_timing_plan(
                    volumes=volumes,
                    cycle_length=opt_result['cycle_length'],
                    phase_greens=opt_result['phase_greens']
                )
        
        # Decide which plan to use
        best_plan = None
        best_delay = current_delay
        best_source = 'keep_current'
        
        # Check optimized plan first (if available)
        if optimized_plan is not None:
            opt_delay = opt_eval['intersection']['average_delay_s_per_veh']
            if opt_delay < best_delay:
                best_plan = optimized_plan
                best_delay = opt_delay
                best_source = 'optimizer'
        
        # Check ML plan
        if ml_plan is not None and ml_eval is not None:
            ml_delay = ml_eval['intersection']['average_delay_s_per_veh']
            if ml_delay < best_delay:
                best_plan = ml_plan
                best_delay = ml_delay
                best_source = f'{model_type}_ml'
        
        # Select the best plan
        if best_plan is not None and best_delay < current_delay:
            if best_source == 'optimizer':
                selected_plan = {
                    'cycle_length': best_plan['cycle_length'],
                    'phase_greens': best_plan['phase_greens'],
                    'evaluation': opt_eval,
                    'source': 'optimizer',
                    'improvement': (current_delay - best_delay) / current_delay * 100 if current_delay > 0 else 0
                }
            else:
                selected_plan = {
                    'cycle_length': best_plan['cycle_length'],
                    'phase_greens': best_plan['phase_greens'],
                    'evaluation': ml_eval,
                    'source': best_source,
                    'improvement': (current_delay - best_delay) / current_delay * 100 if current_delay > 0 else 0
                }
        else:
            # No improvement found - keep current timing
            selected_plan = {
                'cycle_length': current_plan['cycle_length'],
                'phase_greens': current_plan['phase_greens'],
                'evaluation': current_eval,
                'source': 'keep_current',
                'note': 'No improvement found over current timing'
            }
        
        # Validate final plan
        is_valid, violations = self.los_wrapper.validate_timing_plan(
            cycle_length=selected_plan['cycle_length'],
            phase_greens=selected_plan['phase_greens'],
            yellow_times=current_timing.get('yellow_times'),
            red_clearance=current_timing.get('red_clearance')
        )
        
        if not is_valid:
            # Keep current timing if validation fails
            selected_plan = {
                'cycle_length': current_plan['cycle_length'],
                'phase_greens': current_plan['phase_greens'],
                'evaluation': current_eval,
                'source': 'keep_current_safety',
                'safety_violations': violations
            }
        
        selected_plan['is_valid'] = is_valid
        selected_plan['current_los'] = current_los
        selected_plan['current_delay'] = current_delay
        
        return selected_plan
    
    def generate_recommendation_json(
        self,
        intersection_id: str,
        timestamp: str,
        timing_plan: Dict[str, Any],
        yellow_times: Dict[str, float],
        red_clearance: Dict[str, float],
        time_window: str = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON recommendation object for output.
        
        Args:
            intersection_id: Intersection identifier
            timestamp: ISO timestamp for the interval
            timing_plan: The selected timing plan
            yellow_times: Yellow times for each phase
            red_clearance: Red clearance times for each phase
            time_window: Optional time window description (e.g., "12:00-18:00")
            
        Returns:
            JSON-serializable recommendation dict
        """
        evaluation = timing_plan.get('evaluation', {})
        intersection_eval = evaluation.get('intersection', {})
        
        # Build phases list - only green times are recommendations
        # Yellow and red clearance are preserved from current timing (not changeable)
        phases = []
        for phase, green in timing_plan.get('phase_greens', {}).items():
            phases.append({
                'phase': phase,
                'green_recommended': round(green, 1),
                'yellow_unchanged': yellow_times.get(phase, 4.0),
                'red_clearance_unchanged': red_clearance.get(phase, 1.0)
            })
        
        # Compute recommendation score
        current_delay = timing_plan.get('current_delay', 0)
        new_delay = intersection_eval.get('average_delay_s_per_veh', current_delay)
        
        if current_delay > 0:
            improvement = (current_delay - new_delay) / current_delay * 100
        else:
            improvement = 0
        
        # Generate notes
        notes = []
        
        if timing_plan.get('source') == 'hcm2010_fallback':
            notes.append(f"ML plan rejected: {timing_plan.get('rejection_reason', 'unknown')}")
        elif timing_plan.get('source') == 'keep_current':
            notes.append("No change recommended - current timing is optimal")
            if timing_plan.get('rejection_reason'):
                notes.append(timing_plan.get('rejection_reason'))
            if timing_plan.get('note'):
                notes.append(timing_plan.get('note'))
        elif timing_plan.get('source') == 'keep_current_safety':
            notes.append("Keeping current timing due to safety constraints")
        
        if improvement > 0:
            notes.append(f"Estimated delay reduction: {improvement:.1f}%")
            notes.append(f"LOS improvement: {timing_plan.get('current_los', 'N/A')} -> {intersection_eval.get('LOS', 'N/A')}")
        elif improvement < 0:
            notes.append(f"Warning: Plan may increase delay by {-improvement:.1f}%")
        
        notes.append(f"Source: {timing_plan.get('source', 'unknown')}")
        notes.append("Note: Only green times can be changed. Yellow and red clearance times are preserved.")
        
        return {
            'intersection_id': intersection_id,
            'timestamp': timestamp,
            'time_window': time_window or 'single_interval',
            'cycle_length_recommended': round(timing_plan.get('cycle_length', 120), 1),
            'phases': phases,
            'recommended_change_score': round(max(0, min(100, improvement + 50)), 1),
            'notes': notes,
            'is_valid': timing_plan.get('is_valid', False),
            'los_before': timing_plan.get('current_los', 'N/A'),
            'los_after': intersection_eval.get('LOS', 'N/A'),
            'delay_before': round(current_delay, 1),
            'delay_after': round(new_delay, 1)
        }


def predict_for_intersection(
    predictor: TimingPlanPredictor,
    preprocessed_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    model_type: str = 'gradient_boosted',
    output_file: str = None
) -> List[Dict[str, Any]]:
    """
    Generate predictions for an intersection.
    
    When aggregate_mode is enabled in preprocessed data, generates a single
    timing recommendation for the entire time window (e.g., 12pm-6pm).
    
    Args:
        predictor: TimingPlanPredictor instance
        preprocessed_data: Preprocessed intersection data
        timing_data: Timing features from preprocessing
        model_type: Type of model to use
        output_file: Optional path to write NDJSON output
        
    Returns:
        List of recommendation dicts (single item when aggregated)
    """
    intersection_id = preprocessed_data.get('intersection_id')
    volume_df = preprocessed_data.get('volume_df')
    movement_cols = preprocessed_data.get('movement_cols', [])
    timing_features = timing_data
    vol_metadata = preprocessed_data.get('volume_metadata', {})
    
    if volume_df is None or len(volume_df) == 0:
        return []
    
    # Check if we're in aggregate mode
    aggregate_mode = vol_metadata.get('aggregate_mode', False)
    time_window = vol_metadata.get('time_window', None)
    
    # Extract phase movements
    phase_labels = list(timing_features.get('phase_greens', {}).keys())
    phase_movements = extract_phase_movements(phase_labels)
    
    # Get yellow and red clearance times
    yellow_times = timing_features.get('yellow_times', {})
    red_clearance = timing_features.get('red_clearance', {})
    
    # Current timing
    current_timing = {
        'cycle_length': timing_features.get('cycle_length', 120),
        'phase_greens': timing_features.get('phase_greens', {}),
        'yellow_times': yellow_times,
        'red_clearance': red_clearance
    }
    
    recommendations = []
    
    # Process data - either single aggregated row or multiple intervals
    for idx, row in volume_df.iterrows():
        # Get volumes for this interval/window
        volumes = {col: row[col] for col in movement_cols if col in row.index}
        
        # Get features
        feature_cols = [c for c in volume_df.columns 
                       if c not in ['DATE', 'TIME', 'datetime', 'INTID']]
        features = volume_df.iloc[[idx]][feature_cols]
        
        # Generate timestamp
        if 'datetime' in row.index and pd.notna(row['datetime']):
            timestamp = row['datetime'].isoformat()
        else:
            timestamp = datetime.now().isoformat()
        
        # Predict timing plan
        timing_plan = predictor.predict_timing_plan(
            intersection_id=intersection_id,
            features=features,
            volumes=volumes,
            phase_movements=phase_movements,
            current_timing=current_timing,
            model_type=model_type
        )
        
        # Generate recommendation JSON
        recommendation = predictor.generate_recommendation_json(
            intersection_id=intersection_id,
            timestamp=timestamp,
            timing_plan=timing_plan,
            yellow_times=yellow_times,
            red_clearance=red_clearance,
            time_window=time_window
        )
        
        # Add aggregation metadata if in aggregate mode
        if aggregate_mode:
            agg_meta = vol_metadata.get('aggregation', {})
            recommendation['intervals_aggregated'] = agg_meta.get('intervals_aggregated', 1)
            recommendation['aggregation_note'] = f"Timing optimized for average traffic during {time_window}"
        
        recommendations.append(recommendation)
    
    # Write NDJSON if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            for rec in recommendations:
                f.write(json.dumps(rec) + '\n')
        mode_desc = f"aggregated ({time_window})" if aggregate_mode else "per-interval"
        print(f"Wrote {len(recommendations)} {mode_desc} recommendation(s) to {output_file}")
    
    return recommendations


def predict_all(
    preprocessed_data: Dict[str, Dict[str, Any]],
    model_type: str = 'gradient_boosted',
    output_dir: str = 'output'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate predictions for all intersections.
    
    By default, generates a single timing recommendation per intersection
    optimized for the 12pm-6pm time window when preprocessed with
    aggregate_window=True.
    
    Args:
        preprocessed_data: Dict mapping intersection_id to preprocessed data
        model_type: Type of model to use
        output_dir: Directory for output files
        
    Returns:
        Dict mapping intersection_id to recommendations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictor = TimingPlanPredictor()
    all_recommendations = {}
    
    for int_id, data in preprocessed_data.items():
        if data.get('error'):
            print(f"Skipping {int_id}: {data['error']}")
            continue
        
        vol_metadata = data.get('volume_metadata', {})
        aggregate_mode = vol_metadata.get('aggregate_mode', False)
        time_window = vol_metadata.get('time_window', 'per-interval')
        
        print(f"Generating predictions for {int_id} ({time_window})...")
        
        output_file = os.path.join(output_dir, f'{int_id}_recommendations.ndjson')
        
        timing_features = data.get('timing_features', {})
        
        recommendations = predict_for_intersection(
            predictor=predictor,
            preprocessed_data=data,
            timing_data=timing_features,
            model_type=model_type,
            output_file=output_file
        )
        
        all_recommendations[int_id] = recommendations
        
        if recommendations:
            # Summary statistics
            improvements = [r['delay_before'] - r['delay_after'] for r in recommendations]
            avg_improvement = np.mean(improvements)
            mode_desc = "aggregated window" if aggregate_mode else f"{len(recommendations)} intervals"
            print(f"  Generated {len(recommendations)} recommendation(s) ({mode_desc})")
            print(f"  Average delay improvement: {avg_improvement:.1f}s")
    
    return all_recommendations


if __name__ == '__main__':
    # Import data modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from ingest import ingest_all
    from preprocess import preprocess_all, preprocess_intersection
    
    print("Loading data...")
    ingested = ingest_all()
    
    # Define the two rush hour periods with their phase plans
    # Period 1: 1:15pm - 2:45pm (13:15 - 14:45) -> Phase Plan 61
    # Period 2: 2:45pm - 6:45pm (14:45 - 18:45) -> Phase Plan 64
    
    RUSH_HOUR_PERIODS = [
        {
            'name': 'early_afternoon',
            'start_hour': 13,
            'start_minute': 15,
            'end_hour': 14,
            'end_minute': 45,
            'plan_number': 61,
            'time_window': '13:15-14:45'
        },
        {
            'name': 'pm_peak',
            'start_hour': 14,
            'start_minute': 45,
            'end_hour': 18,
            'end_minute': 45,
            'plan_number': 64,
            'time_window': '14:45-18:45'
        }
    ]
    
    os.makedirs('output', exist_ok=True)
    predictor = TimingPlanPredictor()
    
    volumes_data = ingested.get('volumes', {})
    timings_data = ingested.get('timings', {})
    matched = ingested.get('matched_intersections', [])
    
    print(f"\nGenerating predictions for {len(matched)} intersections...")
    print("Two timing recommendations per intersection:")
    print("  - Period 1: 1:15pm - 2:45pm (Phase Plan 61)")
    print("  - Period 2: 2:45pm - 6:45pm (Phase Plan 64)")
    
    for int_id in matched:
        volume_data = volumes_data.get(int_id, {})
        timing_data = timings_data.get(int_id, {})
        
        if volume_data.get('raw_df') is None:
            print(f"\nSkipping {int_id}: No volume data")
            continue
        
        print(f"\nProcessing {int_id}...")
        
        all_recommendations = []
        
        for period in RUSH_HOUR_PERIODS:
            # Filter volume data to the specific time period
            raw_df = volume_data.get('raw_df').copy()
            movement_cols = volume_data['metadata'].get('available_movements', [])
            
            # Parse datetime if not already done
            if 'datetime' not in raw_df.columns:
                raw_df['TIME'] = raw_df['TIME'].fillna('0000').astype(str).str.zfill(4)
                raw_df['DATE'] = raw_df['DATE'].fillna('').astype(str)
                raw_df['datetime'] = pd.to_datetime(
                    raw_df['DATE'] + ' ' + raw_df['TIME'],
                    format='%m/%d/%Y %H%M',
                    errors='coerce'
                )
            
            # Extract hour and minute
            raw_df['hour'] = raw_df['datetime'].dt.hour
            raw_df['minute'] = raw_df['datetime'].dt.minute
            
            # Filter to the specific time period
            start_time = period['start_hour'] * 60 + period['start_minute']
            end_time = period['end_hour'] * 60 + period['end_minute']
            raw_df['time_minutes'] = raw_df['hour'] * 60 + raw_df['minute']
            
            mask = (raw_df['time_minutes'] >= start_time) & (raw_df['time_minutes'] < end_time)
            period_df = raw_df[mask].copy()
            
            if len(period_df) == 0:
                print(f"  {period['time_window']}: No data available")
                continue
            
            # Get timing features for the specific phase plan
            from preprocess import TimingPreprocessor
            timing_processor = TimingPreprocessor()
            timing_features = timing_processor.extract_timing_features(
                timing_data, 
                plan_number=period['plan_number']
            )
            
            if timing_features.get('error'):
                print(f"  {period['time_window']}: Plan {period['plan_number']} not available")
                continue
            
            # Average volumes for the period (convert 15-min to hourly)
            volumes = {}
            for col in movement_cols:
                if col in period_df.columns:
                    volumes[col] = period_df[col].mean() * 4  # Convert to hourly
            
            # Extract phase movements
            phase_labels = list(timing_features.get('phase_greens', {}).keys())
            phase_movements = extract_phase_movements(phase_labels)
            
            # Current timing from the phase plan
            current_timing = {
                'cycle_length': timing_features.get('cycle_length', 120),
                'phase_greens': timing_features.get('phase_greens', {}),
                'yellow_times': timing_features.get('yellow_times', {}),
                'red_clearance': timing_features.get('red_clearance', {})
            }
            
            # Create feature DataFrame for prediction
            feature_data = {col: [period_df[col].mean()] for col in movement_cols if col in period_df.columns}
            feature_data['hour'] = [(period['start_hour'] + period['end_hour']) / 2]
            feature_data['total_volume'] = [sum(volumes.values()) / 4]  # Back to 15-min avg
            features = pd.DataFrame(feature_data)
            
            # Predict timing plan with optimization
            timing_plan = predictor.predict_timing_plan(
                intersection_id=int_id,
                features=features,
                volumes=volumes,
                phase_movements=phase_movements,
                current_timing=current_timing,
                model_type='gradient_boosted',
                use_optimizer=True
            )
            
            # Generate recommendation JSON
            recommendation = predictor.generate_recommendation_json(
                intersection_id=int_id,
                timestamp=datetime.now().isoformat(),
                timing_plan=timing_plan,
                yellow_times=timing_features.get('yellow_times', {}),
                red_clearance=timing_features.get('red_clearance', {}),
                time_window=period['time_window']
            )
            
            # Add period-specific metadata
            recommendation['phase_plan'] = period['plan_number']
            recommendation['period_name'] = period['name']
            recommendation['intervals_in_period'] = len(period_df)
            
            all_recommendations.append(recommendation)
            
            # Print summary
            improvement = recommendation.get('delay_before', 0) - recommendation.get('delay_after', 0)
            print(f"  {period['time_window']} (Plan {period['plan_number']}): "
                  f"LOS {recommendation.get('los_before', 'N/A')} -> {recommendation.get('los_after', 'N/A')}, "
                  f"Delay improvement: {improvement:.1f}s")
        
        # Write recommendations to file
        output_file = os.path.join('output', f'{int_id}_recommendations.ndjson')
        with open(output_file, 'w') as f:
            for rec in all_recommendations:
                f.write(json.dumps(rec) + '\n')
        
        print(f"  Wrote {len(all_recommendations)} recommendations to {output_file}")
    
    print("\nPrediction complete!")
    print("Each intersection now has two timing recommendations:")
    print("  1. 1:15pm - 2:45pm optimized from Phase Plan 61")
    print("  2. 2:45pm - 6:45pm optimized from Phase Plan 64")
