"""
Tests for LOS integration and wrapper functionality.
"""

import sys
import os
import pytest

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))

from LOS import (
    determine_los,
    compute_control_delay,
    compute_intersection_los,
    parse_phase_label,
    LOS_THRESHOLDS
)
from ml.los_wrapper import LOSWrapper


class TestDetermineLOS:
    """Tests for LOS grade determination."""
    
    def test_los_a(self):
        """Test LOS A threshold (0-10 seconds)."""
        assert determine_los(0) == 'A'
        assert determine_los(5) == 'A'
        assert determine_los(9.9) == 'A'
    
    def test_los_b(self):
        """Test LOS B threshold (10-20 seconds)."""
        assert determine_los(10) == 'B'
        assert determine_los(15) == 'B'
        assert determine_los(19.9) == 'B'
    
    def test_los_c(self):
        """Test LOS C threshold (20-35 seconds)."""
        assert determine_los(20) == 'C'
        assert determine_los(25) == 'C'
        assert determine_los(34.9) == 'C'
    
    def test_los_d(self):
        """Test LOS D threshold (35-55 seconds)."""
        assert determine_los(35) == 'D'
        assert determine_los(45) == 'D'
        assert determine_los(54.9) == 'D'
    
    def test_los_e(self):
        """Test LOS E threshold (55-80 seconds)."""
        assert determine_los(55) == 'E'
        assert determine_los(70) == 'E'
        assert determine_los(79.9) == 'E'
    
    def test_los_f(self):
        """Test LOS F threshold (>80 seconds)."""
        assert determine_los(80) == 'F'
        assert determine_los(100) == 'F'
        assert determine_los(150) == 'F'


class TestControlDelay:
    """Tests for control delay calculation."""
    
    def test_low_volume(self):
        """Test delay calculation for low volume conditions."""
        delay, X, g_over_C = compute_control_delay(
            volume_vph=100,
            saturation_flow=1900,
            green_time=40,
            cycle_length=90
        )
        
        assert delay < 20  # Should be LOS A-B
        assert X < 0.2  # Low degree of saturation
        assert 0.4 < g_over_C < 0.5
    
    def test_moderate_volume(self):
        """Test delay calculation for moderate volume conditions."""
        delay, X, g_over_C = compute_control_delay(
            volume_vph=500,
            saturation_flow=1900,
            green_time=30,
            cycle_length=100
        )
        
        assert 10 < delay < 80  # Should be LOS B-E
        assert 0.7 < X < 1.0  # Moderate degree of saturation
    
    def test_high_volume_oversaturated(self):
        """Test delay calculation for oversaturated conditions."""
        delay, X, g_over_C = compute_control_delay(
            volume_vph=1500,
            saturation_flow=1900,
            green_time=20,
            cycle_length=100
        )
        
        assert X > 1.0  # Oversaturated
        assert delay >= 80  # Should be LOS F
    
    def test_zero_green_time(self):
        """Test handling of zero green time."""
        delay, X, g_over_C = compute_control_delay(
            volume_vph=500,
            saturation_flow=1900,
            green_time=0,
            cycle_length=100
        )
        
        assert delay == 150  # Max delay cap
    
    def test_zero_cycle_length(self):
        """Test handling of zero cycle length."""
        delay, X, g_over_C = compute_control_delay(
            volume_vph=500,
            saturation_flow=1900,
            green_time=30,
            cycle_length=0
        )
        
        assert delay == 150  # Max delay cap


class TestIntersectionLOS:
    """Tests for intersection-level LOS calculation."""
    
    def test_weighted_average(self):
        """Test weighted average delay calculation."""
        per_lane = [
            {'movement': 'NBT', 'volume_vph': 400, 'delay_s_per_veh': 20},
            {'movement': 'SBT', 'volume_vph': 600, 'delay_s_per_veh': 30},
            {'movement': 'EBT', 'volume_vph': 500, 'delay_s_per_veh': 25},
        ]
        
        result = compute_intersection_los(per_lane)
        
        # Expected: (400*20 + 600*30 + 500*25) / 1500 = 25.67
        expected_avg = (400*20 + 600*30 + 500*25) / 1500
        
        assert abs(result['average_delay_s_per_veh'] - expected_avg) < 0.1
        assert result['total_volume_vph'] == 1500
        assert result['LOS'] == 'C'  # 25.67 is in C range
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = compute_intersection_los([])
        
        assert result['average_delay_s_per_veh'] == 0
        assert result['total_volume_vph'] == 0


class TestParsePhaseLabel:
    """Tests for phase label parsing."""
    
    def test_eblt(self):
        """Test parsing of EBLT phase."""
        movements = parse_phase_label('1 EBLT')
        assert 'EBL' in movements
        assert 'EBT' in movements
        assert len(movements) == 2
    
    def test_wb_full_direction(self):
        """Test parsing of full direction (WB)."""
        movements = parse_phase_label('2WB')
        assert 'WBL' in movements
        assert 'WBT' in movements
        assert 'WBR' in movements
        assert len(movements) == 3
    
    def test_sbrt(self):
        """Test parsing of SBRT phase."""
        movements = parse_phase_label('8SBRT')
        assert 'SBR' in movements
        assert 'SBT' in movements
        assert len(movements) == 2
    
    def test_not_used(self):
        """Test handling of 'not used' phases."""
        movements = parse_phase_label('4(Not used)')
        assert len(movements) == 0
    
    def test_offset(self):
        """Test handling of 'Offset' label."""
        movements = parse_phase_label('Offset')
        assert len(movements) == 0


class TestLOSWrapper:
    """Tests for the LOSWrapper class."""
    
    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = LOSWrapper()
        assert wrapper.saturation_flow == 1900
        
        wrapper2 = LOSWrapper(saturation_flow=1800)
        assert wrapper2.saturation_flow == 1800
    
    def test_compare_los(self):
        """Test LOS comparison."""
        wrapper = LOSWrapper()
        
        assert wrapper.compare_los('A', 'B') == -1  # A is better
        assert wrapper.compare_los('B', 'A') == 1   # A is better
        assert wrapper.compare_los('C', 'C') == 0   # Equal
        assert wrapper.compare_los('F', 'A') == 1   # A is better
    
    def test_is_improvement(self):
        """Test improvement detection."""
        wrapper = LOSWrapper()
        
        # 10% improvement
        assert wrapper.is_improvement(100, 90, threshold_percent=5.0) == True
        
        # 3% improvement (below threshold)
        assert wrapper.is_improvement(100, 97, threshold_percent=5.0) == False
        
        # Worsening
        assert wrapper.is_improvement(100, 110, threshold_percent=5.0) == False
    
    def test_validate_timing_plan(self):
        """Test timing plan validation."""
        wrapper = LOSWrapper()
        
        # Valid plan
        is_valid, violations = wrapper.validate_timing_plan(
            cycle_length=120,
            phase_greens={'1 EBLT': 30, '2WB': 40},
            yellow_times={'1 EBLT': 4.0, '2WB': 4.5},
            red_clearance={'1 EBLT': 1.0, '2WB': 1.0}
        )
        assert is_valid
        assert len(violations) == 0
        
        # Invalid - cycle too short
        is_valid, violations = wrapper.validate_timing_plan(
            cycle_length=30,
            phase_greens={'1 EBLT': 10, '2WB': 10}
        )
        assert not is_valid
        assert any('Cycle length' in v for v in violations)
        
        # Invalid - green too short
        is_valid, violations = wrapper.validate_timing_plan(
            cycle_length=120,
            phase_greens={'1 EBLT': 3, '2WB': 40}
        )
        assert not is_valid
        assert any('green' in v.lower() for v in violations)
    
    def test_evaluate_timing_plan(self):
        """Test timing plan evaluation."""
        wrapper = LOSWrapper()
        
        volumes = {
            'NBT': 400, 'SBT': 450,
            'EBT': 800, 'WBT': 750
        }
        
        result = wrapper.evaluate_timing_plan(
            volumes=volumes,
            cycle_length=120,
            phase_greens={
                '1 EBLT': 30, '2WB': 30,
                '3NBLT': 20, '4SB': 20
            }
        )
        
        assert 'intersection' in result
        assert 'per_lane' in result
        assert 'LOS' in result['intersection']
        assert 'average_delay_s_per_veh' in result['intersection']


class TestLOSThresholds:
    """Tests for LOS threshold definitions."""
    
    def test_thresholds_exist(self):
        """Test that all LOS grades have thresholds."""
        for grade in ['A', 'B', 'C', 'D', 'E', 'F']:
            assert grade in LOS_THRESHOLDS
            assert len(LOS_THRESHOLDS[grade]) == 2
    
    def test_thresholds_sequential(self):
        """Test that thresholds are sequential."""
        grades = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for i in range(len(grades) - 1):
            current_max = LOS_THRESHOLDS[grades[i]][1]
            next_min = LOS_THRESHOLDS[grades[i + 1]][0]
            assert current_max == next_min


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
