"""
HCM2010 Deterministic Green Split Optimizer.

This module implements the baseline HCM2010-compliant signal timing optimizer
that serves as both a fallback and a benchmark for ML-based approaches.

Reference: Highway Capacity Manual 2010, Chapter 18
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from los_wrapper import LOSWrapper


class HCM2010Optimizer:
    """
    HCM2010-compliant deterministic signal timing optimizer.
    
    This optimizer uses the Webster method and related HCM2010 techniques
    to compute optimal cycle lengths and green splits.
    """
    
    # Constraints
    MIN_CYCLE = 60  # seconds
    MAX_CYCLE = 180  # seconds
    MIN_GREEN = 7  # seconds (pedestrian minimum)
    MAX_GREEN = 90  # seconds
    DEFAULT_YELLOW = 4.0  # seconds
    DEFAULT_RED_CLEARANCE = 1.0  # seconds
    
    def __init__(
        self,
        saturation_flow: int = 1900,
        lost_time_per_phase: float = 4.0
    ):
        """
        Initialize the optimizer.
        
        Args:
            saturation_flow: Saturation flow rate in veh/hr/lane (default 1900)
            lost_time_per_phase: Lost time per phase change in seconds
        """
        self.saturation_flow = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        self.los_wrapper = LOSWrapper(saturation_flow=saturation_flow)
    
    def compute_critical_volumes(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute critical (maximum) volume for each phase.
        
        Args:
            volumes: Dict mapping movements to volumes (vph)
            phase_movements: Dict mapping phase labels to list of movements
            
        Returns:
            Dict mapping phase labels to critical volumes
        """
        critical_volumes = {}
        
        for phase, movements in phase_movements.items():
            phase_vols = [volumes.get(mov, 0) for mov in movements]
            critical_volumes[phase] = max(phase_vols) if phase_vols else 0
        
        return critical_volumes
    
    def compute_flow_ratios(
        self,
        critical_volumes: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute flow ratios (v/s) for each phase.
        
        Args:
            critical_volumes: Dict mapping phases to critical volumes
            
        Returns:
            Dict mapping phases to flow ratios
        """
        return {
            phase: vol / self.saturation_flow
            for phase, vol in critical_volumes.items()
        }
    
    def webster_optimal_cycle(
        self,
        flow_ratios: Dict[str, float],
        num_phases: int
    ) -> float:
        """
        Compute optimal cycle length using Webster's formula.
        
        C_opt = (1.5 * L + 5) / (1 - Y)
        
        Where:
            L = total lost time (typically 4s per phase change)
            Y = sum of critical flow ratios
        
        Args:
            flow_ratios: Dict mapping phases to flow ratios (v/s)
            num_phases: Number of signal phases
            
        Returns:
            Optimal cycle length in seconds
        """
        # Total lost time
        L = self.lost_time_per_phase * num_phases
        
        # Sum of critical flow ratios
        Y = sum(flow_ratios.values())
        
        if Y >= 0.95:
            # Oversaturated - use maximum cycle
            return self.MAX_CYCLE
        
        if Y <= 0:
            # Very low volume - use minimum cycle
            return self.MIN_CYCLE
        
        # Webster's formula
        C_opt = (1.5 * L + 5) / (1 - Y)
        
        # Constrain to valid range
        return max(self.MIN_CYCLE, min(self.MAX_CYCLE, C_opt))
    
    def allocate_green_splits(
        self,
        flow_ratios: Dict[str, float],
        cycle_length: float,
        num_phases: int
    ) -> Dict[str, float]:
        """
        Allocate green time to each phase proportional to flow ratios.
        
        Args:
            flow_ratios: Dict mapping phases to flow ratios
            cycle_length: Total cycle length in seconds
            num_phases: Number of signal phases
            
        Returns:
            Dict mapping phases to green times
        """
        # Total lost time
        total_lost = self.lost_time_per_phase * num_phases
        
        # Effective green time available
        effective_green = cycle_length - total_lost
        
        if effective_green <= 0:
            effective_green = cycle_length * 0.7  # Use 70% as green
        
        # Sum of flow ratios
        Y = sum(flow_ratios.values())
        
        green_times = {}
        
        if Y <= 0:
            # Equal split when no flow data
            equal_green = effective_green / num_phases
            for phase in flow_ratios.keys():
                green_times[phase] = max(self.MIN_GREEN, min(self.MAX_GREEN, equal_green))
        else:
            # Proportional allocation
            for phase, y in flow_ratios.items():
                g = (y / Y) * effective_green
                green_times[phase] = max(self.MIN_GREEN, min(self.MAX_GREEN, g))
        
        return green_times
    
    def optimize_timing_plan(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_yellow: Optional[Dict[str, float]] = None,
        current_red_clearance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compute optimal timing plan using HCM2010 methods.
        
        Args:
            volumes: Dict mapping movements to volumes (vph)
            phase_movements: Dict mapping phase labels to list of movements
            current_yellow: Current yellow times (optional, uses defaults)
            current_red_clearance: Current red clearance times (optional)
            
        Returns:
            Dict with optimal timing plan
        """
        # Compute critical volumes and flow ratios
        critical_vols = self.compute_critical_volumes(volumes, phase_movements)
        flow_ratios = self.compute_flow_ratios(critical_vols)
        
        num_phases = len(phase_movements)
        
        # Compute optimal cycle length
        cycle_length = self.webster_optimal_cycle(flow_ratios, num_phases)
        
        # Allocate green times
        green_times = self.allocate_green_splits(flow_ratios, cycle_length, num_phases)
        
        # Use provided or default yellow/red times
        yellow_times = current_yellow or {p: self.DEFAULT_YELLOW for p in green_times}
        red_clearance = current_red_clearance or {p: self.DEFAULT_RED_CLEARANCE for p in green_times}
        
        # Validate the plan
        is_valid, violations = self.los_wrapper.validate_timing_plan(
            cycle_length=cycle_length,
            phase_greens=green_times,
            yellow_times=yellow_times,
            red_clearance=red_clearance
        )
        
        # Evaluate the plan
        evaluation = self.los_wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=cycle_length,
            phase_greens=green_times,
            yellow_times=yellow_times,
            red_clearance=red_clearance
        )
        
        return {
            'cycle_length': round(cycle_length, 1),
            'phase_greens': {k: round(v, 1) for k, v in green_times.items()},
            'yellow_times': yellow_times,
            'red_clearance': red_clearance,
            'flow_ratios': {k: round(v, 4) for k, v in flow_ratios.items()},
            'critical_volumes': critical_vols,
            'is_valid': is_valid,
            'violations': violations,
            'evaluation': evaluation,
            'method': 'HCM2010_Webster'
        }
    
    def iterative_optimize(
        self,
        volumes: Dict[str, float],
        phase_movements: Dict[str, List[str]],
        current_plan: Dict[str, Any],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Iteratively optimize timing plan to minimize delay.
        
        This uses a simple hill-climbing approach to refine the Webster-based
        initial solution by adjusting green splits.
        
        Args:
            volumes: Dict mapping movements to volumes
            phase_movements: Dict mapping phases to movements
            current_plan: Current timing plan dict
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized timing plan
        """
        # Start with Webster solution
        best_plan = self.optimize_timing_plan(
            volumes=volumes,
            phase_movements=phase_movements,
            current_yellow=current_plan.get('yellow_times'),
            current_red_clearance=current_plan.get('red_clearance')
        )
        
        best_delay = best_plan['evaluation']['intersection']['average_delay_s_per_veh']
        
        # Try small adjustments
        for iteration in range(max_iterations):
            improved = False
            
            for phase in best_plan['phase_greens'].keys():
                # Try increasing green
                test_greens = best_plan['phase_greens'].copy()
                test_greens[phase] = min(self.MAX_GREEN, test_greens[phase] + 2)
                
                eval_result = self.los_wrapper.evaluate_timing_plan(
                    volumes=volumes,
                    cycle_length=best_plan['cycle_length'],
                    phase_greens=test_greens
                )
                
                new_delay = eval_result['intersection']['average_delay_s_per_veh']
                
                if new_delay < best_delay:
                    best_plan['phase_greens'] = test_greens
                    best_plan['evaluation'] = eval_result
                    best_delay = new_delay
                    improved = True
                    continue
                
                # Try decreasing green
                test_greens = best_plan['phase_greens'].copy()
                test_greens[phase] = max(self.MIN_GREEN, test_greens[phase] - 2)
                
                eval_result = self.los_wrapper.evaluate_timing_plan(
                    volumes=volumes,
                    cycle_length=best_plan['cycle_length'],
                    phase_greens=test_greens
                )
                
                new_delay = eval_result['intersection']['average_delay_s_per_veh']
                
                if new_delay < best_delay:
                    best_plan['phase_greens'] = test_greens
                    best_plan['evaluation'] = eval_result
                    best_delay = new_delay
                    improved = True
            
            if not improved:
                break
        
        best_plan['method'] = 'HCM2010_Iterative'
        return best_plan


def extract_phase_movements(phase_labels: List[str]) -> Dict[str, List[str]]:
    """
    Extract phase to movement mappings from phase labels.
    
    Args:
        phase_labels: List of phase label strings (e.g., ['1 EBLT', '2WB'])
        
    Returns:
        Dict mapping phase labels to lists of movements
    """
    import re
    
    phase_movements = {}
    
    for label in phase_labels:
        # Remove numeric prefix and spaces
        cleaned = re.sub(r'^[\d\s]+', '', label.strip())
        cleaned = cleaned.replace(' ', '').upper()
        
        if not cleaned or cleaned.lower() in ('not used', 'notused', 'offset'):
            continue
        
        movements = []
        
        # Extract direction
        direction_match = re.match(r'^(NB|SB|EB|WB)', cleaned)
        if not direction_match:
            continue
        
        direction = direction_match.group(1)
        suffix = cleaned[len(direction):]
        
        # Parse movement types
        if not suffix:
            movements = [f'{direction}L', f'{direction}T', f'{direction}R']
        elif 'L' in suffix and 'T' in suffix:
            movements = [f'{direction}L', f'{direction}T']
        elif 'R' in suffix and 'T' in suffix:
            movements = [f'{direction}R', f'{direction}T']
        elif 'L' in suffix:
            movements = [f'{direction}L']
        elif 'T' in suffix:
            movements = [f'{direction}T']
        elif 'R' in suffix:
            movements = [f'{direction}R']
        else:
            movements = [f'{direction}L', f'{direction}T', f'{direction}R']
        
        if movements:
            phase_movements[label] = movements
    
    return phase_movements


if __name__ == '__main__':
    import json
    
    # Example usage
    optimizer = HCM2010Optimizer()
    
    # Sample volumes (vph)
    volumes = {
        'NBL': 100, 'NBT': 400, 'NBR': 80,
        'SBL': 120, 'SBT': 450, 'SBR': 90,
        'EBL': 80, 'EBT': 800, 'EBR': 60,
        'WBL': 70, 'WBT': 750, 'WBR': 50
    }
    
    # Phase movements
    phase_labels = ['1 EBLT', '2WB', '3NBLT', '4SB', '5WBLT', '6 EB', '7SBLT', '8NB']
    phase_movements = extract_phase_movements(phase_labels)
    
    print("Phase Movements:")
    for phase, movs in phase_movements.items():
        print(f"  {phase}: {movs}")
    
    # Optimize
    result = optimizer.optimize_timing_plan(volumes, phase_movements)
    
    print("\nOptimized Timing Plan:")
    print(f"  Cycle Length: {result['cycle_length']}s")
    print(f"  Method: {result['method']}")
    print(f"  Valid: {result['is_valid']}")
    
    print("\n  Phase Greens:")
    for phase, green in result['phase_greens'].items():
        print(f"    {phase}: {green}s")
    
    print(f"\n  Intersection LOS: {result['evaluation']['intersection']['LOS']}")
    print(f"  Average Delay: {result['evaluation']['intersection']['average_delay_s_per_veh']}s")
