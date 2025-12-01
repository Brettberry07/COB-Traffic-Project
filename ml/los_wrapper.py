"""
LOS Wrapper - Interface to LOS.py for ML models.

This module provides a clean interface for the ML system to interact with
the LOS.py module for computing Level of Service and validating timing plans.
"""

import sys
import os

# Add parent directory to path to import LOS module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Optional, Tuple
from LOS import (
    compute_los_for_intersection,
    compute_control_delay,
    determine_los,
    compute_intersection_los,
    parse_phase_timing_csv,
    parse_volume_csv,
    parse_phase_label,
    get_green_time_for_movement,
    DEFAULT_SATURATION_FLOW,
    DEFAULT_ANALYSIS_PERIOD,
    LOS_THRESHOLDS,
    MOVEMENT_COLS,
    MAX_DELAY_CAP,
)


class LOSWrapper:
    """Wrapper class for LOS calculations used by ML models."""
    
    # LOS grade ordering for comparison (F is worst, A is best)
    LOS_ORDER = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    # Safety constraints for timing plans
    MIN_CYCLE_LENGTH = 60  # seconds
    MAX_CYCLE_LENGTH = 180  # seconds
    MIN_GREEN_TIME = 7  # seconds (pedestrian minimum)
    MIN_YELLOW_TIME = 3.0  # seconds
    MIN_RED_CLEARANCE = 1.0  # seconds
    MAX_GREEN_TIME = 90  # seconds
    
    def __init__(self, saturation_flow: int = DEFAULT_SATURATION_FLOW):
        """
        Initialize the LOS wrapper.
        
        Args:
            saturation_flow: Saturation flow rate in veh/hr/lane (default 1900)
        """
        self.saturation_flow = saturation_flow
    
    def compute_los(
        self,
        phase_csv: str,
        volume_csv: str,
        plan_number: int = 25,
        aggregation: str = 'peak_hour'
    ) -> Dict[str, Any]:
        """
        Compute LOS for an intersection from CSV files.
        
        Args:
            phase_csv: Path to phase timing CSV file
            volume_csv: Path to volume CSV file
            plan_number: Signal timing plan number to use
            aggregation: 'peak_hour' or 'total' for volume aggregation
            
        Returns:
            Dict with LOS results including per-lane and intersection-level metrics
        """
        return compute_los_for_intersection(
            phase_csv=phase_csv,
            volume_csv=volume_csv,
            saturation_flow=self.saturation_flow,
            plan_number=plan_number,
            aggregation=aggregation
        )
    
    def compute_delay_for_movement(
        self,
        volume_vph: float,
        green_time: float,
        cycle_length: float
    ) -> Tuple[float, float, float]:
        """
        Compute control delay for a single movement.
        
        Args:
            volume_vph: Volume in vehicles per hour
            green_time: Effective green time in seconds
            cycle_length: Cycle length in seconds
            
        Returns:
            Tuple of (delay_seconds, degree_of_saturation, g_over_C)
        """
        return compute_control_delay(
            volume_vph=volume_vph,
            saturation_flow=self.saturation_flow,
            green_time=green_time,
            cycle_length=cycle_length
        )
    
    def evaluate_timing_plan(
        self,
        volumes: Dict[str, float],
        cycle_length: float,
        phase_greens: Dict[str, float],
        yellow_times: Optional[Dict[str, float]] = None,
        red_clearance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a proposed timing plan against volume data.
        
        Args:
            volumes: Dict mapping movements (e.g., 'NBT') to volumes in vph
            cycle_length: Total cycle length in seconds
            phase_greens: Dict mapping phase labels to green times
            yellow_times: Optional dict mapping phases to yellow times
            red_clearance: Optional dict mapping phases to red clearance times
            
        Returns:
            Dict with evaluation results including per-lane delay and LOS
        """
        # Build movement to green time mapping
        movement_greens = {}
        for phase_label, green_time in phase_greens.items():
            movements = parse_phase_label(phase_label)
            for mov in movements:
                if mov in movement_greens:
                    movement_greens[mov] = max(movement_greens[mov], green_time)
                else:
                    movement_greens[mov] = green_time
        
        per_lane_results = []
        
        for movement, volume_vph in volumes.items():
            if volume_vph <= 0:
                continue
            
            # Get green time for this movement
            green_time, match_type = get_green_time_for_movement(
                movement, movement_greens, cycle_length
            )
            
            # Compute delay
            delay, X, g_over_C = self.compute_delay_for_movement(
                volume_vph=volume_vph,
                green_time=green_time,
                cycle_length=cycle_length
            )
            
            per_lane_results.append({
                'movement': movement,
                'volume_vph': volume_vph,
                'g_s': round(green_time, 1),
                'g_over_C': round(g_over_C, 3),
                'degree_of_saturation': round(X, 3),
                'delay_s_per_veh': round(delay, 1),
                'LOS': determine_los(delay),
                'match_type': match_type
            })
        
        # Compute intersection-level LOS
        intersection_summary = compute_intersection_los(per_lane_results)
        
        return {
            'cycle_length_s': cycle_length,
            'phase_greens': phase_greens,
            'yellow_times': yellow_times or {},
            'red_clearance': red_clearance or {},
            'per_lane': per_lane_results,
            'intersection': intersection_summary
        }
    
    def validate_timing_plan(
        self,
        cycle_length: float,
        phase_greens: Dict[str, float],
        yellow_times: Optional[Dict[str, float]] = None,
        red_clearance: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a timing plan meets safety constraints.
        
        Args:
            cycle_length: Proposed cycle length in seconds
            phase_greens: Dict mapping phase labels to green times
            yellow_times: Optional dict mapping phases to yellow times
            red_clearance: Optional dict mapping phases to red clearance times
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check cycle length
        if cycle_length < self.MIN_CYCLE_LENGTH:
            violations.append(f"Cycle length {cycle_length}s < minimum {self.MIN_CYCLE_LENGTH}s")
        if cycle_length > self.MAX_CYCLE_LENGTH:
            violations.append(f"Cycle length {cycle_length}s > maximum {self.MAX_CYCLE_LENGTH}s")
        
        # Check green times
        for phase, green in phase_greens.items():
            if green < self.MIN_GREEN_TIME:
                violations.append(f"Phase {phase} green {green}s < minimum {self.MIN_GREEN_TIME}s")
            if green > self.MAX_GREEN_TIME:
                violations.append(f"Phase {phase} green {green}s > maximum {self.MAX_GREEN_TIME}s")
        
        # Check yellow times if provided
        if yellow_times:
            for phase, yellow in yellow_times.items():
                if yellow < self.MIN_YELLOW_TIME:
                    violations.append(f"Phase {phase} yellow {yellow}s < minimum {self.MIN_YELLOW_TIME}s")
        
        # Check red clearance if provided
        if red_clearance:
            for phase, red in red_clearance.items():
                if red < self.MIN_RED_CLEARANCE:
                    violations.append(f"Phase {phase} red clearance {red}s < minimum {self.MIN_RED_CLEARANCE}s")
        
        return len(violations) == 0, violations
    
    def compare_los(self, los1: str, los2: str) -> int:
        """
        Compare two LOS grades.
        
        Args:
            los1: First LOS grade (A-F)
            los2: Second LOS grade (A-F)
            
        Returns:
            -1 if los1 is better, 0 if equal, 1 if los2 is better
        """
        order1 = self.LOS_ORDER.get(los1.upper(), 5)
        order2 = self.LOS_ORDER.get(los2.upper(), 5)
        
        if order1 < order2:
            return -1
        elif order1 > order2:
            return 1
        else:
            return 0
    
    def is_improvement(
        self,
        current_delay: float,
        proposed_delay: float,
        threshold_percent: float = 5.0
    ) -> bool:
        """
        Check if a proposed plan improves delay by at least threshold percent.
        
        Args:
            current_delay: Current average delay in seconds
            proposed_delay: Proposed average delay in seconds
            threshold_percent: Minimum improvement threshold (default 5%)
            
        Returns:
            True if proposed is better by at least threshold percent
        """
        if current_delay <= 0:
            return proposed_delay <= 0
        
        improvement = (current_delay - proposed_delay) / current_delay * 100
        return improvement >= threshold_percent
    
    def get_los_threshold(self, los: str) -> Tuple[float, float]:
        """
        Get the delay thresholds for a given LOS grade.
        
        Args:
            los: LOS grade (A-F)
            
        Returns:
            Tuple of (min_delay, max_delay) for the grade
        """
        return LOS_THRESHOLDS.get(los.upper(), (0, float('inf')))
    
    def parse_phase_timing(self, file_path: str, plan_number: int = 25) -> Dict[str, Any]:
        """
        Parse a phase timing CSV file.
        
        Args:
            file_path: Path to phase timing CSV
            plan_number: Plan number to extract
            
        Returns:
            Dict with parsed timing data
        """
        return parse_phase_timing_csv(file_path, plan_number=plan_number)
    
    def parse_volume(self, file_path: str, aggregation: str = 'peak_hour') -> Dict[str, Any]:
        """
        Parse a volume CSV file.
        
        Args:
            file_path: Path to volume CSV
            aggregation: 'peak_hour' or 'total'
            
        Returns:
            Dict with parsed volume data
        """
        return parse_volume_csv(file_path, aggregation=aggregation)
    
    @staticmethod
    def determine_los(delay: float) -> str:
        """
        Determine LOS grade from delay value.
        
        Args:
            delay: Control delay in seconds
            
        Returns:
            LOS grade (A-F)
        """
        return determine_los(delay)
    
    @staticmethod
    def get_movement_columns() -> List[str]:
        """Get list of standard movement column names."""
        return MOVEMENT_COLS.copy()
