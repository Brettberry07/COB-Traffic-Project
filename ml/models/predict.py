"""
Prediction Module - Inference pipeline producing validated timing plans.

This module:
- Loads trained models
- Generates timing plan recommendations for each 15-minute interval
- Validates all plans using LOS.py
- Falls back to HCM2010 baseline if ML plans worsen LOS
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
from models.hcm2010 import HCM2010Optimizer, extract_phase_movements
from models.train import (
    GradientBoostedTimingModel,
    SequenceTimingModel,
    HybridTimingModel
)


class TimingPlanPredictor:
    """
    Generates validated timing plan recommendations.
    """
    
    # LOS degradation threshold (percentage)
    LOS_DEGRADATION_THRESHOLD = 5.0
    
    def __init__(
        self,
        model_dir: str = 'ml/models/trained',
        los_degradation_threshold: float = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
            los_degradation_threshold: Max allowed LOS degradation (%)
        """
        self.model_dir = model_dir
        self.los_wrapper = LOSWrapper()
        self.hcm2010 = HCM2010Optimizer()
        
        if los_degradation_threshold is not None:
            self.LOS_DEGRADATION_THRESHOLD = los_degradation_threshold
        
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
    
    def generate_hcm2010_baseline(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_timing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate HCM2010 baseline timing plan.
        
        Args:
            volumes: Traffic volumes per movement
            phase_movements: Phase to movement mapping
            current_timing: Current timing parameters
            
        Returns:
            HCM2010 optimized timing plan
        """
        return self.hcm2010.optimize_timing_plan(
            volumes=volumes,
            phase_movements=phase_movements,
            current_yellow=current_timing.get('yellow_times'),
            current_red_clearance=current_timing.get('red_clearance')
        )
    
    def predict_timing_plan(
        self,
        intersection_id: str,
        features: pd.DataFrame,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_timing: Dict[str, Any],
        model_type: str = 'gradient_boosted'
    ) -> Dict[str, Any]:
        """
        Predict timing plan using ML model with validation.
        
        Args:
            intersection_id: Intersection identifier
            features: Feature DataFrame for current interval
            volumes: Current traffic volumes
            phase_movements: Phase to movement mapping
            current_timing: Current timing parameters
            model_type: Type of model to use
            
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
                    
                    # Apply constraints
                    constrained_greens = {}
                    for phase, green in avg_greens.items():
                        constrained_greens[phase] = max(
                            self.hcm2010.MIN_GREEN,
                            min(self.hcm2010.MAX_GREEN, green)
                        )
                    
                    # Compute cycle length using HCM2010
                    critical_vols = self.hcm2010.compute_critical_volumes(
                        volumes, phase_movements
                    )
                    flow_ratios = self.hcm2010.compute_flow_ratios(critical_vols)
                    cycle_length = self.hcm2010.webster_optimal_cycle(
                        flow_ratios, len(phase_movements)
                    )
                    
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
        
        # Get HCM2010 baseline
        hcm_plan = self.generate_hcm2010_baseline(
            volumes, phase_movements, current_timing
        )
        hcm_delay = hcm_plan['evaluation']['intersection']['average_delay_s_per_veh']
        
        # Decide which plan to use
        if ml_plan is not None and ml_eval is not None:
            ml_delay = ml_eval['intersection']['average_delay_s_per_veh']
            
            # Check if ML plan worsens LOS beyond threshold
            if current_delay > 0:
                degradation = ((ml_delay - current_delay) / current_delay) * 100
            else:
                degradation = 0 if ml_delay <= current_delay else 100
            
            if degradation > self.LOS_DEGRADATION_THRESHOLD:
                # Reject ML plan, use HCM2010 baseline
                selected_plan = hcm_plan
                selected_plan['source'] = 'hcm2010_fallback'
                selected_plan['rejection_reason'] = f'ML plan worsened LOS by {degradation:.1f}%'
            else:
                # Use ML plan
                selected_plan = {
                    'cycle_length': ml_plan['cycle_length'],
                    'phase_greens': ml_plan['phase_greens'],
                    'evaluation': ml_eval,
                    'source': f'{model_type}_ml',
                    'improvement': (current_delay - ml_delay) / current_delay * 100 if current_delay > 0 else 0
                }
        else:
            # No ML model available, use HCM2010
            selected_plan = hcm_plan
            selected_plan['source'] = 'hcm2010_baseline'
        
        # Validate final plan
        is_valid, violations = self.los_wrapper.validate_timing_plan(
            cycle_length=selected_plan['cycle_length'],
            phase_greens=selected_plan['phase_greens'],
            yellow_times=current_timing.get('yellow_times'),
            red_clearance=current_timing.get('red_clearance')
        )
        
        if not is_valid:
            # Safety fallback
            selected_plan = hcm_plan
            selected_plan['source'] = 'hcm2010_safety_fallback'
            selected_plan['safety_violations'] = violations
        
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
        red_clearance: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate a JSON recommendation object for output.
        
        Args:
            intersection_id: Intersection identifier
            timestamp: ISO timestamp for the interval
            timing_plan: The selected timing plan
            yellow_times: Yellow times for each phase
            red_clearance: Red clearance times for each phase
            
        Returns:
            JSON-serializable recommendation dict
        """
        evaluation = timing_plan.get('evaluation', {})
        intersection_eval = evaluation.get('intersection', {})
        
        # Build phases list
        phases = []
        for phase, green in timing_plan.get('phase_greens', {}).items():
            phases.append({
                'phase': phase,
                'green': round(green, 1),
                'yellow': yellow_times.get(phase, 4.0),
                'red_clearance': red_clearance.get(phase, 1.0)
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
        
        if improvement > 0:
            notes.append(f"Estimated delay reduction: {improvement:.1f}%")
            notes.append(f"LOS improvement: {timing_plan.get('current_los', 'N/A')} -> {intersection_eval.get('LOS', 'N/A')}")
        elif improvement < 0:
            notes.append(f"Warning: Plan may increase delay by {-improvement:.1f}%")
        
        notes.append(f"Source: {timing_plan.get('source', 'unknown')}")
        
        return {
            'intersection_id': intersection_id,
            'timestamp': timestamp,
            'cycle_length': round(timing_plan.get('cycle_length', 120), 1),
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
    Generate predictions for all intervals of an intersection.
    
    Args:
        predictor: TimingPlanPredictor instance
        preprocessed_data: Preprocessed intersection data
        timing_data: Timing features from preprocessing
        model_type: Type of model to use
        output_file: Optional path to write NDJSON output
        
    Returns:
        List of recommendation dicts
    """
    intersection_id = preprocessed_data.get('intersection_id')
    volume_df = preprocessed_data.get('volume_df')
    movement_cols = preprocessed_data.get('movement_cols', [])
    timing_features = timing_data
    
    if volume_df is None or len(volume_df) == 0:
        return []
    
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
    
    # Process each interval
    for idx, row in volume_df.iterrows():
        # Get volumes for this interval
        volumes = {col: row[col] for col in movement_cols if col in row.index}
        
        # Get features for this interval
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
            red_clearance=red_clearance
        )
        
        recommendations.append(recommendation)
    
    # Write NDJSON if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            for rec in recommendations:
                f.write(json.dumps(rec) + '\n')
        print(f"Wrote {len(recommendations)} recommendations to {output_file}")
    
    return recommendations


def predict_all(
    preprocessed_data: Dict[str, Dict[str, Any]],
    model_type: str = 'gradient_boosted',
    output_dir: str = 'output'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate predictions for all intersections.
    
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
        
        print(f"Generating predictions for {int_id}...")
        
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
            print(f"  Generated {len(recommendations)} recommendations")
            print(f"  Average delay improvement: {avg_improvement:.1f}s")
    
    return all_recommendations


if __name__ == '__main__':
    # Import data modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from ingest import ingest_all
    from preprocess import preprocess_all
    
    print("Loading and preprocessing data...")
    ingested = ingest_all()
    preprocessed = preprocess_all(ingested)
    
    print("\nGenerating predictions...")
    recommendations = predict_all(
        preprocessed_data=preprocessed,
        model_type='gradient_boosted',
        output_dir='output'
    )
    
    print("\nPrediction complete!")
    print(f"Processed {len(recommendations)} intersections")
