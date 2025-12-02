"""
Tests for HCM2010 optimizer and model training.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models'))

from ml.models.hcm2010 import HCM2010Optimizer, extract_phase_movements


class TestHCM2010Optimizer:
    """Tests for HCM2010 optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        return HCM2010Optimizer()
    
    @pytest.fixture
    def sample_volumes(self):
        return {
            'NBL': 100, 'NBT': 400, 'NBR': 80,
            'SBL': 120, 'SBT': 450, 'SBR': 90,
            'EBL': 80, 'EBT': 800, 'EBR': 60,
            'WBL': 70, 'WBT': 750, 'WBR': 50
        }
    
    @pytest.fixture
    def phase_movements(self):
        return {
            '1 EBLT': ['EBL', 'EBT'],
            '2WB': ['WBL', 'WBT', 'WBR'],
            '3NBLT': ['NBL', 'NBT'],
            '4SB': ['SBL', 'SBT', 'SBR']
        }
    
    def test_compute_critical_volumes(self, optimizer, sample_volumes, phase_movements):
        """Test critical volume computation."""
        critical = optimizer.compute_critical_volumes(sample_volumes, phase_movements)
        
        assert '1 EBLT' in critical
        assert critical['1 EBLT'] == 800  # max(EBL=80, EBT=800)
        assert critical['3NBLT'] == 400   # max(NBL=100, NBT=400)
    
    def test_compute_flow_ratios(self, optimizer):
        """Test flow ratio computation."""
        critical_vols = {'phase1': 950, 'phase2': 475}
        
        ratios = optimizer.compute_flow_ratios(critical_vols)
        
        assert ratios['phase1'] == 950 / 1900
        assert ratios['phase2'] == 475 / 1900
    
    def test_webster_optimal_cycle_normal(self, optimizer):
        """Test Webster's formula for normal conditions."""
        flow_ratios = {'p1': 0.2, 'p2': 0.15, 'p3': 0.1, 'p4': 0.1}
        
        cycle = optimizer.webster_optimal_cycle(flow_ratios, 4)
        
        # Y = 0.55, L = 16
        # C = (1.5*16 + 5) / (1 - 0.55) = 29/0.45 ≈ 64.4
        assert 60 <= cycle <= 180  # Within bounds
    
    def test_webster_optimal_cycle_oversaturated(self, optimizer):
        """Test Webster's formula for oversaturated conditions."""
        flow_ratios = {'p1': 0.4, 'p2': 0.3, 'p3': 0.2, 'p4': 0.15}  # Y = 1.05
        
        cycle = optimizer.webster_optimal_cycle(flow_ratios, 4)
        
        assert cycle == 180  # Should be max cycle due to oversaturation
    
    def test_allocate_green_splits(self, optimizer):
        """Test green time allocation."""
        flow_ratios = {'p1': 0.2, 'p2': 0.1, 'p3': 0.1, 'p4': 0.1}
        
        greens = optimizer.allocate_green_splits(flow_ratios, 100, 4)
        
        # Check all greens are within bounds
        for phase, green in greens.items():
            assert 7 <= green <= 90
    
    def test_optimize_timing_plan(self, optimizer, sample_volumes, phase_movements):
        """Test full optimization pipeline."""
        result = optimizer.optimize_timing_plan(sample_volumes, phase_movements)
        
        assert 'cycle_length' in result
        assert 'phase_greens' in result
        assert 'is_valid' in result
        assert 'evaluation' in result
        
        assert 60 <= result['cycle_length'] <= 180
        assert result['is_valid']
    
    def test_optimizer_constraints(self, optimizer, sample_volumes, phase_movements):
        """Test that optimizer respects safety constraints."""
        result = optimizer.optimize_timing_plan(sample_volumes, phase_movements)
        
        for phase, green in result['phase_greens'].items():
            assert green >= optimizer.MIN_GREEN
            assert green <= optimizer.MAX_GREEN


class TestExtractPhaseMovements:
    """Tests for phase movement extraction."""
    
    def test_standard_phases(self):
        """Test extraction of standard phase labels."""
        labels = ['1 EBLT', '2WB', '3NBLT', '4SB']
        movements = extract_phase_movements(labels)
        
        assert '1 EBLT' in movements
        assert 'EBL' in movements['1 EBLT']
        assert 'EBT' in movements['1 EBLT']
    
    def test_full_direction(self):
        """Test extraction of full direction phases."""
        labels = ['2WB']
        movements = extract_phase_movements(labels)
        
        assert len(movements['2WB']) == 3
        assert 'WBL' in movements['2WB']
        assert 'WBT' in movements['2WB']
        assert 'WBR' in movements['2WB']
    
    def test_not_used_phase(self):
        """Test handling of 'not used' phases."""
        labels = ['1 EBLT', '4(Not used)']
        movements = extract_phase_movements(labels)
        
        assert '1 EBLT' in movements
        assert '4(Not used)' not in movements
    
    def test_empty_labels(self):
        """Test handling of empty labels."""
        labels = []
        movements = extract_phase_movements(labels)
        
        assert len(movements) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
