"""
Timing Optimizer Module - Advanced optimization for signal timing.

This module provides search-based optimization to find timing improvements
that the base ML model may miss. It:
1. Uses ML predictions as a starting point
2. Performs local search around predictions
3. Explores alternative timing patterns
4. Validates all changes against LOS constraints

This addresses the issue of ML models not finding improvements when
they exist.
"""

import sys
import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from los_wrapper import LOSWrapper


class OptimizationStrategy(Enum):
    """Strategies for timing optimization."""
    LOCAL_SEARCH = "local_search"
    GRID_SEARCH = "grid_search"
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC = "genetic"
    BAYESIAN = "bayesian"


@dataclass
class TimingConstraints:
    """Constraints for timing optimization."""
    min_green: float = 7.0
    max_green: float = 90.0
    min_cycle: float = 60.0
    max_cycle: float = 180.0
    min_yellow: float = 3.0
    min_red_clearance: float = 1.0


class TimingOptimizer:
    """
    Advanced timing optimizer using multiple search strategies.
    
    This optimizer explores the timing parameter space more thoroughly
    than simple ML predictions, finding improvements that the model misses.
    """
    
    def __init__(
        self,
        constraints: TimingConstraints = None,
        saturation_flow: int = 1900
    ):
        """
        Initialize the optimizer.
        
        Args:
            constraints: Timing constraints (uses defaults if None)
            saturation_flow: Saturation flow rate (veh/hr/lane)
        """
        self.constraints = constraints or TimingConstraints()
        self.los_wrapper = LOSWrapper(saturation_flow=saturation_flow)
    
    def _evaluate_plan(
        self,
        volumes: Dict[str, float],
        cycle_length: float,
        phase_greens: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Evaluate a timing plan and return delay and LOS.
        
        Args:
            volumes: Traffic volumes per movement
            cycle_length: Cycle length in seconds
            phase_greens: Green times per phase
            
        Returns:
            Tuple of (average_delay, LOS_grade)
        """
        try:
            eval_result = self.los_wrapper.evaluate_timing_plan(
                volumes=volumes,
                cycle_length=cycle_length,
                phase_greens=phase_greens
            )
            delay = eval_result['intersection']['average_delay_s_per_veh']
            los = eval_result['intersection']['LOS']
            return delay, los
        except Exception:
            return float('inf'), 'F'
    
    def _is_valid_plan(
        self,
        cycle_length: float,
        phase_greens: Dict[str, float]
    ) -> bool:
        """Check if a timing plan meets constraints."""
        if cycle_length < self.constraints.min_cycle or cycle_length > self.constraints.max_cycle:
            return False
        
        for phase, green in phase_greens.items():
            if green < self.constraints.min_green or green > self.constraints.max_green:
                return False
        
        return True
    
    def local_search(
        self,
        volumes: Dict[str, float],
        initial_plan: Dict[str, Any],
        max_iterations: int = 50,
        step_sizes: List[int] = None,
        improvement_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform local search optimization starting from initial plan.
        
        Systematically explores small adjustments to green times
        to find improvements.
        
        Args:
            volumes: Traffic volumes per movement
            initial_plan: Starting timing plan
            max_iterations: Maximum optimization iterations
            step_sizes: Green time adjustment steps to try
            improvement_threshold: Minimum delay improvement (seconds) to accept
            
        Returns:
            Optimized timing plan dict
        """
        step_sizes = step_sizes or [1, 2, 3, 5, 8]
        
        best_cycle = initial_plan.get('cycle_length', 120)
        best_greens = initial_plan.get('phase_greens', {}).copy()
        best_delay, best_los = self._evaluate_plan(volumes, best_cycle, best_greens)
        
        initial_delay = best_delay
        improvements_found = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try adjusting each phase
            for phase in list(best_greens.keys()):
                for step in step_sizes:
                    # Try increasing green
                    test_greens = best_greens.copy()
                    test_greens[phase] = min(
                        self.constraints.max_green,
                        test_greens[phase] + step
                    )
                    
                    if self._is_valid_plan(best_cycle, test_greens):
                        delay, los = self._evaluate_plan(volumes, best_cycle, test_greens)
                        
                        if delay < best_delay - improvement_threshold:
                            best_greens = test_greens
                            best_delay = delay
                            best_los = los
                            improved = True
                            improvements_found += 1
                            break
                    
                    # Try decreasing green
                    test_greens = best_greens.copy()
                    test_greens[phase] = max(
                        self.constraints.min_green,
                        test_greens[phase] - step
                    )
                    
                    if self._is_valid_plan(best_cycle, test_greens):
                        delay, los = self._evaluate_plan(volumes, best_cycle, test_greens)
                        
                        if delay < best_delay - improvement_threshold:
                            best_greens = test_greens
                            best_delay = delay
                            best_los = los
                            improved = True
                            improvements_found += 1
                            break
                
                if improved:
                    break
            
            # Try cycle length adjustments
            if not improved:
                for cycle_adj in [-10, -5, 5, 10]:
                    test_cycle = best_cycle + cycle_adj
                    
                    if self.constraints.min_cycle <= test_cycle <= self.constraints.max_cycle:
                        delay, los = self._evaluate_plan(volumes, test_cycle, best_greens)
                        
                        if delay < best_delay - improvement_threshold:
                            best_cycle = test_cycle
                            best_delay = delay
                            best_los = los
                            improved = True
                            improvements_found += 1
                            break
            
            if not improved:
                break
        
        return {
            'cycle_length': best_cycle,
            'phase_greens': best_greens,
            'delay': best_delay,
            'los': best_los,
            'initial_delay': initial_delay,
            'improvement': initial_delay - best_delay,
            'improvement_pct': (initial_delay - best_delay) / initial_delay * 100 if initial_delay > 0 else 0,
            'iterations': iteration + 1,
            'improvements_found': improvements_found,
            'method': 'local_search'
        }
    
    def grid_search(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        cycle_range: Tuple[float, float] = (60, 180),
        cycle_step: float = 10,
        green_range: Tuple[float, float] = (7, 60),
        green_step: float = 5,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform grid search over timing parameters.
        
        Useful for finding globally good solutions when local search
        might get stuck in local optima.
        
        Args:
            volumes: Traffic volumes per movement
            phase_movements: Phase to movement mapping
            cycle_range: Min and max cycle lengths to try
            cycle_step: Cycle length step size
            green_range: Min and max green times to try
            green_step: Green time step size
            top_n: Number of top solutions to return
            
        Returns:
            List of top N timing plans sorted by delay
        """
        phases = list(phase_movements.keys())
        
        # Generate cycle lengths to try
        cycles = np.arange(cycle_range[0], cycle_range[1] + 1, cycle_step)
        
        # Generate green time combinations (simplified - proportional splits)
        solutions = []
        
        for cycle in cycles:
            # Available green time (cycle - clearances)
            num_phases = len(phases)
            clearance_time = num_phases * 5  # yellow + red per phase
            available_green = cycle - clearance_time
            
            if available_green < num_phases * self.constraints.min_green:
                continue
            
            # Try different split ratios for major vs minor phases
            # Assuming phases 2, 6 (WB, EB through) are major
            for major_ratio in np.arange(0.3, 0.8, 0.1):
                minor_ratio = (1 - major_ratio) / max(1, num_phases - 2)
                
                greens = {}
                for i, phase in enumerate(phases):
                    # Assign more green to through phases (typically 2, 6)
                    if '2' in phase or '6' in phase:
                        green = available_green * major_ratio / 2
                    else:
                        green = available_green * minor_ratio
                    
                    green = max(self.constraints.min_green, 
                               min(self.constraints.max_green, green))
                    greens[phase] = green
                
                if self._is_valid_plan(cycle, greens):
                    delay, los = self._evaluate_plan(volumes, cycle, greens)
                    
                    solutions.append({
                        'cycle_length': cycle,
                        'phase_greens': greens,
                        'delay': delay,
                        'los': los,
                        'major_ratio': major_ratio,
                        'method': 'grid_search'
                    })
        
        # Sort by delay and return top N
        solutions.sort(key=lambda x: x['delay'])
        return solutions[:top_n]
    
    def adaptive_search(
        self,
        volumes: Dict[str, float],
        initial_plan: Dict[str, Any],
        phase_movements: Dict[str, List[str]],
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Adaptive search that adjusts strategy based on progress.
        
        Starts with local search, switches to broader exploration
        if stuck, then refines the best solution found.
        
        Args:
            volumes: Traffic volumes per movement
            initial_plan: Starting timing plan
            phase_movements: Phase to movement mapping
            max_iterations: Total iteration budget
            
        Returns:
            Best timing plan found
        """
        # Phase 1: Local search from initial plan (40% of budget)
        local_result = self.local_search(
            volumes=volumes,
            initial_plan=initial_plan,
            max_iterations=int(max_iterations * 0.4)
        )
        
        best_plan = {
            'cycle_length': local_result['cycle_length'],
            'phase_greens': local_result['phase_greens']
        }
        best_delay = local_result['delay']
        
        # Phase 2: Grid search for alternative solutions (if local search didn't improve much)
        if local_result['improvement_pct'] < 5:
            grid_results = self.grid_search(
                volumes=volumes,
                phase_movements=phase_movements,
                top_n=3
            )
            
            for grid_plan in grid_results:
                if grid_plan['delay'] < best_delay:
                    best_plan = {
                        'cycle_length': grid_plan['cycle_length'],
                        'phase_greens': grid_plan['phase_greens']
                    }
                    best_delay = grid_plan['delay']
        
        # Phase 3: Fine-tune best solution with smaller steps
        final_result = self.local_search(
            volumes=volumes,
            initial_plan=best_plan,
            max_iterations=int(max_iterations * 0.3),
            step_sizes=[1, 2],
            improvement_threshold=0.1
        )
        
        # Compute overall improvement
        initial_delay, _ = self._evaluate_plan(
            volumes,
            initial_plan.get('cycle_length', 120),
            initial_plan.get('phase_greens', {})
        )
        
        return {
            'cycle_length': final_result['cycle_length'],
            'phase_greens': final_result['phase_greens'],
            'delay': final_result['delay'],
            'los': final_result['los'],
            'initial_delay': initial_delay,
            'improvement': initial_delay - final_result['delay'],
            'improvement_pct': (initial_delay - final_result['delay']) / initial_delay * 100 if initial_delay > 0 else 0,
            'method': 'adaptive_search'
        }
    
    def optimize(
        self,
        volumes: Dict[str, float],
        current_plan: Dict[str, Any],
        ml_plan: Optional[Dict[str, Any]] = None,
        phase_movements: Optional[Dict[str, List[str]]] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.LOCAL_SEARCH
    ) -> Dict[str, Any]:
        """
        Main optimization entry point.
        
        Args:
            volumes: Traffic volumes per movement
            current_plan: Current timing plan
            ml_plan: Optional ML-predicted timing plan
            phase_movements: Phase to movement mapping (for grid search)
            strategy: Optimization strategy to use
            
        Returns:
            Optimized timing plan with metadata
        """
        # Start from ML plan if available and valid, otherwise current plan
        if ml_plan and self._is_valid_plan(
            ml_plan.get('cycle_length', 0),
            ml_plan.get('phase_greens', {})
        ):
            initial_plan = ml_plan
        else:
            initial_plan = current_plan
        
        if strategy == OptimizationStrategy.LOCAL_SEARCH:
            return self.local_search(volumes, initial_plan)
        
        elif strategy == OptimizationStrategy.GRID_SEARCH:
            if not phase_movements:
                return self.local_search(volumes, initial_plan)
            
            results = self.grid_search(volumes, phase_movements)
            if results:
                best = results[0]
                # Get current delay for comparison
                current_delay, _ = self._evaluate_plan(
                    volumes,
                    current_plan.get('cycle_length', 120),
                    current_plan.get('phase_greens', {})
                )
                best['initial_delay'] = current_delay
                best['improvement'] = current_delay - best['delay']
                best['improvement_pct'] = (current_delay - best['delay']) / current_delay * 100 if current_delay > 0 else 0
                return best
            return self.local_search(volumes, initial_plan)
        
        elif strategy == OptimizationStrategy.GRADIENT_DESCENT:
            # For now, fall back to local search with small steps
            return self.local_search(
                volumes, initial_plan,
                step_sizes=[1, 2],
                improvement_threshold=0.1
            )
        
        else:
            # Default: adaptive search
            return self.adaptive_search(
                volumes, initial_plan,
                phase_movements or {}
            )


class MultiPeriodOptimizer:
    """
    Optimizer for multiple time periods within a day.
    
    Instead of one timing for 12-6pm, optimizes for sub-periods:
    - 12-2pm (early afternoon)
    - 2-4pm (mid afternoon)  
    - 4-6pm (PM peak)
    """
    
    def __init__(self, base_optimizer: TimingOptimizer = None):
        """
        Initialize multi-period optimizer.
        
        Args:
            base_optimizer: Base timing optimizer to use
        """
        self.optimizer = base_optimizer or TimingOptimizer()
    
    def optimize_periods(
        self,
        volume_df,  # pd.DataFrame
        movement_cols: List[str],
        current_plan: Dict[str, Any],
        phase_movements: Dict[str, List[str]],
        periods: List[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Optimize timing for multiple time periods.
        
        Args:
            volume_df: DataFrame with volume data and 'hour' column
            movement_cols: List of movement column names
            current_plan: Current timing plan
            phase_movements: Phase to movement mapping
            periods: List of (start_hour, end_hour) tuples
            
        Returns:
            Dict with optimized timing for each period
        """
        periods = periods or [(12, 14), (14, 16), (16, 18)]
        
        results = {}
        
        for start_hour, end_hour in periods:
            period_key = f"{start_hour:02d}:00-{end_hour:02d}:00"
            
            # Filter to period
            mask = (volume_df['hour'] >= start_hour) & (volume_df['hour'] < end_hour)
            period_df = volume_df[mask]
            
            if len(period_df) == 0:
                continue
            
            # Average volumes for the period
            volumes = {}
            for col in movement_cols:
                if col in period_df.columns:
                    # Convert to hourly rate (assuming 15-min intervals)
                    volumes[col] = period_df[col].mean() * 4
            
            # Optimize
            optimized = self.optimizer.optimize(
                volumes=volumes,
                current_plan=current_plan,
                phase_movements=phase_movements,
                strategy=OptimizationStrategy.LOCAL_SEARCH
            )
            
            results[period_key] = {
                'optimized_plan': optimized,
                'volumes': volumes,
                'sample_size': len(period_df)
            }
        
        return results


if __name__ == '__main__':
    # Test the optimizer
    print("Testing TimingOptimizer...")
    
    optimizer = TimingOptimizer()
    
    # Sample volumes
    volumes = {
        'NBL': 100, 'NBT': 400, 'NBR': 80,
        'SBL': 120, 'SBT': 450, 'SBR': 90,
        'EBL': 80, 'EBT': 800, 'EBR': 60,
        'WBL': 70, 'WBT': 750, 'WBR': 50
    }
    
    current_plan = {
        'cycle_length': 120,
        'phase_greens': {
            '1 EBLT': 15, '2WB': 77, '3NBLT': 15, '4SB': 33,
            '5WBLT': 15, '6 EB': 77, '7SBLT': 17, '8NB': 31
        }
    }
    
    print("\nLocal Search Optimization:")
    result = optimizer.local_search(volumes, current_plan)
    
    print(f"  Initial delay: {result['initial_delay']:.1f}s")
    print(f"  Optimized delay: {result['delay']:.1f}s")
    print(f"  Improvement: {result['improvement']:.1f}s ({result['improvement_pct']:.1f}%)")
    print(f"  Iterations: {result['iterations']}")
    print(f"  LOS: {result['los']}")
    
    print("\n  Optimized greens:")
    for phase, green in result['phase_greens'].items():
        original = current_plan['phase_greens'].get(phase, 0)
        diff = green - original
        print(f"    {phase}: {green:.1f}s (was {original}s, diff: {diff:+.1f}s)")
