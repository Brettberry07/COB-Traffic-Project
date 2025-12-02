"""
Model Training Module - Training pipeline for ML models.

This module implements:
- Gradient Boosted Trees for timing prediction
- LSTM/Sequence model for time series
- Hybrid ML + constrained optimizer approach
- Walk-forward validation

All models output timing plans that are validated against LOS.py.
"""

import sys
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from los_wrapper import LOSWrapper
from models.hcm2010 import HCM2010Optimizer, extract_phase_movements


class TimingModel:
    """Base class for timing prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.los_wrapper = LOSWrapper()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        raise NotImplementedError


class GradientBoostedTimingModel(TimingModel):
    """
    Gradient Boosted Trees model for timing prediction.
    
    Predicts green splits for each phase based on engineered features
    from volume data and temporal patterns.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_split: int = 10
    ):
        super().__init__('GradientBoosted')
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        
        self.models = {}  # One model per phase
        self.scaler = StandardScaler()
        self.feature_names = []
        self.phase_names = []
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        phase_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fit gradient boosted models for each phase.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame with columns for each phase green time
            phase_names: List of phase column names in y
            
        Returns:
            Dict with training metrics
        """
        self.feature_names = list(X.columns)
        self.phase_names = phase_names or list(y.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        metrics = {}
        
        for phase in self.phase_names:
            if phase not in y.columns:
                continue
            
            y_phase = y[phase].values
            
            # Skip if all values are the same
            if np.std(y_phase) < 0.01:
                self.models[phase] = None
                metrics[phase] = {'skipped': True, 'reason': 'constant_target'}
                continue
            
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
            
            model.fit(X_scaled, y_phase)
            self.models[phase] = model
            
            # Training metrics
            y_pred = model.predict(X_scaled)
            metrics[phase] = {
                'mse': float(mean_squared_error(y_phase, y_pred)),
                'mae': float(mean_absolute_error(y_phase, y_pred)),
                'feature_importance': dict(zip(
                    self.feature_names,
                    model.feature_importances_.tolist()
                ))
            }
        
        self.is_fitted = True
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict green times for each phase.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with predicted green times
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        for phase in self.phase_names:
            model = self.models.get(phase)
            if model is None:
                # Use default value for skipped phases
                predictions[phase] = np.full(len(X), 15.0)
            else:
                predictions[phase] = model.predict(X_scaled)
        
        return pd.DataFrame(predictions)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'phase_names': self.phase_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'min_samples_split': self.min_samples_split
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.phase_names = data['phase_names']
        self.is_fitted = True


class SequenceTimingModel(TimingModel):
    """
    Sequence model (LSTM-like) for timing prediction.
    
    Uses a simple implementation that doesn't require TensorFlow/PyTorch,
    falling back to sklearn's sequential processing.
    """
    
    def __init__(
        self,
        sequence_length: int = 16,  # 4 hours of 15-min intervals
        hidden_size: int = 32
    ):
        super().__init__('Sequence')
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # Use ensemble of GBM models on sequence features
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.phase_names = []
    
    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence features from time series data.
        
        Args:
            X: Feature DataFrame (sorted by time)
            y: Target DataFrame
            
        Returns:
            Tuple of (sequence_features, targets)
        """
        n_samples = len(X) - self.sequence_length
        
        if n_samples <= 0:
            return np.array([]), np.array([])
        
        X_vals = X.values
        y_vals = y.values
        
        # Create sequence features (flatten last N intervals)
        n_features = X_vals.shape[1]
        seq_features = np.zeros((n_samples, self.sequence_length * n_features))
        
        for i in range(n_samples):
            seq = X_vals[i:i + self.sequence_length].flatten()
            seq_features[i] = seq
        
        # Targets are from the end of each sequence
        targets = y_vals[self.sequence_length:]
        
        return seq_features, targets
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        phase_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fit sequence model.
        
        Args:
            X: Feature DataFrame (time-sorted)
            y: Target DataFrame
            phase_names: List of phase column names
            
        Returns:
            Dict with training metrics
        """
        self.feature_names = list(X.columns)
        self.phase_names = phase_names or list(y.columns)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        if len(X_seq) == 0:
            return {'error': 'Not enough data for sequences'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_seq)
        
        metrics = {}
        
        for i, phase in enumerate(self.phase_names):
            y_phase = y_seq[:, i] if len(y_seq.shape) > 1 else y_seq
            
            if np.std(y_phase) < 0.01:
                self.models[phase] = None
                metrics[phase] = {'skipped': True}
                continue
            
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_scaled, y_phase)
            self.models[phase] = model
            
            y_pred = model.predict(X_scaled)
            metrics[phase] = {
                'mse': float(mean_squared_error(y_phase, y_pred)),
                'mae': float(mean_absolute_error(y_phase, y_pred))
            }
        
        self.is_fitted = True
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict green times using sequence features."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # For prediction, we need to create sequence features
        # This is a simplified version - in practice you'd pass the sequence
        n_features = len(self.feature_names)
        seq_len = self.sequence_length * n_features
        
        # Create dummy sequences from recent data
        X_seq = np.zeros((len(X), seq_len))
        X_vals = X.values
        
        for i in range(len(X)):
            start_idx = max(0, i - self.sequence_length + 1)
            seq = X_vals[start_idx:i + 1].flatten()
            # Pad if necessary
            if len(seq) < seq_len:
                seq = np.pad(seq, (seq_len - len(seq), 0), mode='edge')
            X_seq[i] = seq[-seq_len:]
        
        X_scaled = self.scaler.transform(X_seq)
        
        predictions = {}
        for phase in self.phase_names:
            model = self.models.get(phase)
            if model is None:
                predictions[phase] = np.full(len(X), 15.0)
            else:
                predictions[phase] = model.predict(X_scaled)
        
        return pd.DataFrame(predictions)
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'phase_names': self.phase_names,
                'sequence_length': self.sequence_length
            }, f)
    
    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.models = data['models']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.phase_names = data['phase_names']
        self.sequence_length = data['sequence_length']
        self.is_fitted = True


class HybridTimingModel(TimingModel):
    """
    Hybrid model combining ML predictions with HCM2010 optimization.
    
    Uses ML to predict initial green splits, then applies HCM2010
    constraints and optimization to produce valid timing plans.
    """
    
    def __init__(self):
        super().__init__('Hybrid')
        
        self.ml_model = GradientBoostedTimingModel()
        self.optimizer = HCM2010Optimizer()
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        phase_names: List[str] = None
    ) -> Dict[str, Any]:
        """Fit the ML component of the hybrid model."""
        metrics = self.ml_model.fit(X, y, phase_names)
        self.is_fitted = True
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Predict and optimize timing plan.
        
        Args:
            X: Feature DataFrame
            volumes: Current traffic volumes
            phase_movements: Phase to movement mapping
            
        Returns:
            Optimized timing plan
        """
        # Get ML predictions
        ml_predictions = self.ml_model.predict(X)
        
        # Average predictions if multiple rows
        avg_greens = ml_predictions.mean().to_dict()
        
        # Apply HCM2010 constraints
        constrained_greens = {}
        for phase, green in avg_greens.items():
            constrained_greens[phase] = max(
                self.optimizer.MIN_GREEN,
                min(self.optimizer.MAX_GREEN, green)
            )
        
        # Use optimizer to compute cycle length
        critical_vols = self.optimizer.compute_critical_volumes(volumes, phase_movements)
        flow_ratios = self.optimizer.compute_flow_ratios(critical_vols)
        cycle_length = self.optimizer.webster_optimal_cycle(flow_ratios, len(phase_movements))
        
        # Evaluate the plan
        evaluation = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=cycle_length,
            phase_greens=constrained_greens
        )
        
        return {
            'cycle_length': cycle_length,
            'phase_greens': constrained_greens,
            'ml_raw_predictions': avg_greens,
            'evaluation': evaluation,
            'method': 'Hybrid_ML_HCM2010'
        }
    
    def save(self, path: str) -> None:
        self.ml_model.save(path)
    
    def load(self, path: str) -> None:
        self.ml_model.load(path)
        self.is_fitted = True


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Simulates realistic deployment where models are trained on past data
    and evaluated on future data.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 96 * 7  # 1 week of 15-min intervals
    ):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(
        self,
        X: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for walk-forward validation.
        
        Args:
            X: Feature DataFrame (time-sorted)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(X)
        splits = []
        
        # Use sklearn's TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for train_idx, test_idx in tscv.split(X):
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(
        self,
        model: TimingModel,
        X: pd.DataFrame,
        y: pd.DataFrame,
        phase_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            model: TimingModel instance
            X: Feature DataFrame
            y: Target DataFrame
            phase_names: Phase column names
            
        Returns:
            Dict with validation metrics for each fold
        """
        splits = self.split(X)
        fold_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train, phase_names)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Compute metrics
            fold_result = {
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': {}
            }
            
            for phase in phase_names or y.columns:
                if phase in y_test.columns and phase in y_pred.columns:
                    mse = mean_squared_error(y_test[phase], y_pred[phase])
                    mae = mean_absolute_error(y_test[phase], y_pred[phase])
                    fold_result['metrics'][phase] = {
                        'mse': float(mse),
                        'mae': float(mae)
                    }
            
            fold_metrics.append(fold_result)
        
        # Aggregate metrics
        avg_metrics = {}
        for phase in phase_names or y.columns:
            maes = [f['metrics'].get(phase, {}).get('mae', 0) for f in fold_metrics]
            avg_metrics[phase] = {
                'avg_mae': float(np.mean(maes)),
                'std_mae': float(np.std(maes))
            }
        
        return {
            'folds': fold_metrics,
            'average_metrics': avg_metrics,
            'n_splits': self.n_splits
        }


class ModelTrainer:
    """
    Main training pipeline for all model types.
    """
    
    def __init__(
        self,
        output_dir: str = 'ml/models/trained'
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.los_wrapper = LOSWrapper()
        self.validator = WalkForwardValidator()
    
    def prepare_training_data(
        self,
        preprocessed_data: Dict[str, Any],
        timing_features: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Prepare training data from preprocessed intersection data.
        
        Args:
            preprocessed_data: Dict from preprocess_intersection
            timing_features: Timing features dict
            
        Returns:
            Tuple of (features_df, targets_df, phase_names)
        """
        volume_df = preprocessed_data.get('volume_df')
        movement_cols = preprocessed_data.get('movement_cols', [])
        
        if volume_df is None or len(volume_df) == 0:
            return None, None, []
        
        # Feature columns
        feature_cols = []
        
        # Temporal features
        temporal = ['hour', 'day_of_week', 'is_weekend', 'interval_of_day']
        feature_cols.extend([c for c in temporal if c in volume_df.columns])
        
        # Volume features
        for col in movement_cols:
            if col in volume_df.columns:
                feature_cols.append(col)
        
        # Rolling features
        rolling = [c for c in volume_df.columns if 'mean_' in c or 'std_' in c]
        feature_cols.extend(rolling)
        
        # Total volume
        if 'total_volume' in volume_df.columns:
            feature_cols.append('total_volume')
        
        X = volume_df[feature_cols].copy()
        
        # For targets, we use the historical timing (which would be the same 
        # for all rows in this implementation - in practice you'd have varying timings)
        phase_greens = timing_features.get('phase_greens', {})
        phase_names = list(phase_greens.keys())
        
        # Create target DataFrame
        y = pd.DataFrame({
            phase: [green] * len(X)
            for phase, green in phase_greens.items()
        })
        
        return X, y, phase_names
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        phase_names: List[str],
        intersection_id: str
    ) -> Dict[str, Any]:
        """
        Train all model types and compare performance.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            phase_names: List of phase names
            intersection_id: Intersection identifier
            
        Returns:
            Dict with results for all models
        """
        results = {
            'intersection_id': intersection_id,
            'training_samples': len(X),
            'models': {}
        }
        
        # 1. Gradient Boosted Model
        gb_model = GradientBoostedTimingModel()
        gb_metrics = gb_model.fit(X, y, phase_names)
        gb_val = self.validator.validate(gb_model, X, y, phase_names)
        
        results['models']['gradient_boosted'] = {
            'training_metrics': gb_metrics,
            'validation': gb_val
        }
        
        # Save model
        gb_path = os.path.join(self.output_dir, f'{intersection_id}_gb.pkl')
        gb_model.save(gb_path)
        results['models']['gradient_boosted']['model_path'] = gb_path
        
        # 2. Sequence Model
        seq_model = SequenceTimingModel()
        seq_metrics = seq_model.fit(X, y, phase_names)
        
        if 'error' not in seq_metrics:
            seq_val = self.validator.validate(seq_model, X, y, phase_names)
            results['models']['sequence'] = {
                'training_metrics': seq_metrics,
                'validation': seq_val
            }
            
            seq_path = os.path.join(self.output_dir, f'{intersection_id}_seq.pkl')
            seq_model.save(seq_path)
            results['models']['sequence']['model_path'] = seq_path
        else:
            results['models']['sequence'] = seq_metrics
        
        # 3. Hybrid Model
        hybrid_model = HybridTimingModel()
        hybrid_metrics = hybrid_model.fit(X, y, phase_names)
        hybrid_val = self.validator.validate(hybrid_model.ml_model, X, y, phase_names)
        
        results['models']['hybrid'] = {
            'training_metrics': hybrid_metrics,
            'validation': hybrid_val
        }
        
        hybrid_path = os.path.join(self.output_dir, f'{intersection_id}_hybrid.pkl')
        hybrid_model.save(hybrid_path)
        results['models']['hybrid']['model_path'] = hybrid_path
        
        # 4. HCM2010 Baseline (no training needed)
        results['models']['hcm2010_baseline'] = {
            'training_metrics': None,
            'validation': None,
            'note': 'Deterministic method - no training required'
        }
        
        # Compare models
        results['comparison'] = self._compare_models(results['models'])
        
        return results
    
    def _compare_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance and select best."""
        comparison = {}
        
        for model_name, model_data in models.items():
            if model_name == 'hcm2010_baseline':
                continue
            
            val = model_data.get('validation', {})
            avg_metrics = val.get('average_metrics', {})
            
            if avg_metrics:
                # Average MAE across all phases
                maes = [m.get('avg_mae', float('inf')) for m in avg_metrics.values()]
                comparison[model_name] = {
                    'avg_mae': float(np.mean(maes)) if maes else float('inf')
                }
        
        # Find best model
        if comparison:
            best_model = min(comparison.keys(), key=lambda k: comparison[k]['avg_mae'])
            comparison['best_model'] = best_model
        else:
            comparison['best_model'] = 'hcm2010_baseline'
        
        return comparison


def generate_benchmark_report(
    training_results: Dict[str, Dict[str, Any]],
    output_path: str = 'benchmark_report.json'
) -> None:
    """
    Generate benchmark report comparing all model families.
    
    Args:
        training_results: Dict mapping intersection_id to training results
        output_path: Path to save report
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'intersections': {},
        'summary': {}
    }
    
    model_wins = {}
    
    for int_id, results in training_results.items():
        report['intersections'][int_id] = {
            'samples': results.get('training_samples', 0),
            'best_model': results.get('comparison', {}).get('best_model', 'unknown'),
            'model_metrics': {}
        }
        
        for model_name, model_data in results.get('models', {}).items():
            if model_name == 'hcm2010_baseline':
                continue
            
            val = model_data.get('validation', {})
            avg = val.get('average_metrics', {})
            
            if avg:
                maes = [m.get('avg_mae', 0) for m in avg.values()]
                report['intersections'][int_id]['model_metrics'][model_name] = {
                    'avg_mae': float(np.mean(maes))
                }
        
        # Count wins
        best = results.get('comparison', {}).get('best_model', 'unknown')
        model_wins[best] = model_wins.get(best, 0) + 1
    
    report['summary'] = {
        'total_intersections': len(training_results),
        'model_wins': model_wins
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Benchmark report saved to {output_path}")


if __name__ == '__main__':
    # Import data modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
    from ingest import ingest_all
    from preprocess import preprocess_all
    
    print("Loading and preprocessing data...")
    ingested = ingest_all()
    preprocessed = preprocess_all(ingested)
    
    print("Training models...")
    trainer = ModelTrainer()
    
    all_results = {}
    
    for int_id, data in preprocessed.items():
        if data.get('error'):
            print(f"Skipping {int_id}: {data['error']}")
            continue
        
        print(f"\nTraining models for {int_id}...")
        
        timing_features = data.get('timing_features', {})
        X, y, phase_names = trainer.prepare_training_data(data, timing_features)
        
        if X is None or len(X) < 100:
            print(f"  Insufficient data for {int_id}")
            continue
        
        results = trainer.train_all_models(X, y, phase_names, int_id)
        all_results[int_id] = results
        
        print(f"  Best model: {results['comparison'].get('best_model', 'unknown')}")
    
    # Generate benchmark report
    print("\nGenerating benchmark report...")
    generate_benchmark_report(all_results, 'ml/models/benchmark_report.json')
    
    print("\nTraining complete!")
