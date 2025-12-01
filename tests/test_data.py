"""
Tests for data ingestion and preprocessing.
"""

import sys
import os
import pytest
import tempfile
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'data'))

from ml.data.ingest import (
    parse_time_value,
    ingest_volume_file,
    ingest_phase_timing_file,
    ingest_all
)
from ml.data.preprocess import (
    VolumePreprocessor,
    TimingPreprocessor,
    preprocess_intersection
)


class TestParseTimeValue:
    """Tests for time value parsing."""
    
    def test_excel_format(self):
        """Test parsing Excel-escaped format."""
        assert parse_time_value('="0015"') == '0015'
        assert parse_time_value('="0000"') == '0000'
        assert parse_time_value('="1230"') == '1230'
    
    def test_plain_format(self):
        """Test parsing plain time format."""
        assert parse_time_value('0015') == '0015'
        assert parse_time_value('1200') == '1200'
    
    def test_quoted_format(self):
        """Test parsing quoted format."""
        assert parse_time_value('"0015"') == '0015'
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        assert parse_time_value(None) is None
        assert parse_time_value(np.nan) is None
    
    def test_invalid_format(self):
        """Test handling of invalid formats."""
        assert parse_time_value('abc') is None
        assert parse_time_value('12') is None


class TestIngestVolumeFile:
    """Tests for volume file ingestion."""
    
    @pytest.fixture
    def sample_volume_csv(self):
        """Create a sample volume CSV file."""
        content = """Turning Movement Count,
15 Minute Counts,
DATE,TIME,INTID,NBL,NBT,NBR,SBL,SBT,SBR,EBL,EBT,EBR,WBL,WBT,WBR
10/11/2025,="0000",6,0,11,0,23,9,5,1,39,*,2,44,*
10/11/2025,="0015",6,0,11,5,22,8,7,2,43,*,1,49,*
10/11/2025,="0030",6,0,9,3,12,6,6,2,27,10,0,41,5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_ingest_volume_file(self, sample_volume_csv):
        """Test ingesting a volume file."""
        try:
            result = ingest_volume_file(sample_volume_csv)
            
            assert 'raw_df' in result
            assert result['raw_df'] is not None
            assert 'metadata' in result
            assert result['metadata']['row_count'] == 3
            
            # Check missing value tracking
            assert 'EBR' in result['metadata']['missing_values']
            assert result['metadata']['missing_values']['EBR'] == 2
        finally:
            os.unlink(sample_volume_csv)
    
    def test_ingest_nonexistent_file(self):
        """Test handling of nonexistent file."""
        result = ingest_volume_file('/nonexistent/file.csv')
        
        assert result['raw_df'] is None
        assert 'error' in result['metadata']


class TestIngestPhaseTimingFile:
    """Tests for phase timing file ingestion."""
    
    @pytest.fixture
    def sample_timing_csv(self):
        """Create a sample timing CSV file."""
        content = """,Plan,,,,,,,,,
,,,,,,,,,,
Phase,,1 EBLT,2WB,3NBLT,4SB,5WBLT,6 EB,7SBLT,8NB,Offset
,25,15,77,15,33,15,77,17,31,125
,28,15,94,41,30,15,94,21,50,177
Yellow Change,,4,4.5,4,4.5,4,4.5,4,4.5,
Red Clearence,,1,1,1,1,1,1,1,1,
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content)
            return f.name
    
    def test_ingest_timing_file(self, sample_timing_csv):
        """Test ingesting a timing file."""
        try:
            result = ingest_phase_timing_file(sample_timing_csv)
            
            assert 'plans' in result
            assert 25 in result['plans']
            assert 28 in result['plans']
            
            # Check plan 25
            plan_25 = result['plans'][25]
            assert 'phase_greens' in plan_25
            assert '1 EBLT' in plan_25['phase_greens']
            assert plan_25['phase_greens']['1 EBLT'] == 15
            assert plan_25['phase_greens']['2WB'] == 77
            
            # Check yellow times
            assert 'yellow_times' in result
            assert result['yellow_times'].get('1 EBLT') == 4
            
            # Check red clearance
            assert 'red_clearance' in result
            assert result['red_clearance'].get('1 EBLT') == 1
        finally:
            os.unlink(sample_timing_csv)


class TestVolumePreprocessor:
    """Tests for volume data preprocessing."""
    
    def test_parse_datetime(self):
        """Test datetime parsing."""
        df = pd.DataFrame({
            'DATE': ['10/11/2025', '10/11/2025'],
            'TIME': ['0000', '0015']
        })
        
        processor = VolumePreprocessor()
        result = processor.parse_datetime(df)
        
        assert 'datetime' in result.columns
        assert result['datetime'].iloc[0].hour == 0
        assert result['datetime'].iloc[0].minute == 0
        assert result['datetime'].iloc[1].minute == 15
    
    def test_impute_missing_values(self):
        """Test missing value imputation."""
        df = pd.DataFrame({
            'DATE': ['10/11/2025'] * 5,
            'TIME': ['0000', '0015', '0030', '0045', '0100'],
            'NBT': [10, np.nan, np.nan, 15, 20],
            'EBT': [100, 110, np.nan, 120, 130]
        })
        
        processor = VolumePreprocessor()
        df = processor.parse_datetime(df)
        result, counts = processor.impute_missing_values(df, ['NBT', 'EBT'])
        
        # Check no NaN values remain
        assert result['NBT'].isna().sum() == 0
        assert result['EBT'].isna().sum() == 0
        
        # Check imputation counts
        assert counts['NBT'] == 2
        assert counts['EBT'] == 1
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        df = pd.DataFrame({
            'DATE': ['10/11/2025', '10/12/2025'],  # Saturday, Sunday
            'TIME': ['0800', '1400']
        })
        
        processor = VolumePreprocessor()
        df = processor.parse_datetime(df)
        result = processor.extract_temporal_features(df)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'interval_of_day' in result.columns
        
        assert result['hour'].iloc[0] == 8
        assert result['hour'].iloc[1] == 14
    
    def test_extract_rolling_features(self):
        """Test rolling feature extraction."""
        df = pd.DataFrame({
            'DATE': ['10/11/2025'] * 10,
            'TIME': [f'{h:02d}00' for h in range(10)],
            'NBT': list(range(10, 110, 10))
        })
        
        processor = VolumePreprocessor(rolling_windows=[2, 4])
        df = processor.parse_datetime(df)
        result = processor.extract_rolling_features(df, ['NBT'])
        
        assert 'NBT_mean_2' in result.columns
        assert 'NBT_mean_4' in result.columns
        assert 'NBT_std_2' in result.columns


class TestTimingPreprocessor:
    """Tests for timing data preprocessing."""
    
    def test_extract_timing_features(self):
        """Test timing feature extraction."""
        timing_data = {
            'plans': {
                25: {
                    'phase_greens': {
                        '1 EBLT': 15, '2WB': 77, '3NBLT': 15, '4SB': 33,
                        '5WBLT': 15, '6 EB': 77, '7SBLT': 17, '8NB': 31
                    },
                    'offset': 125
                }
            },
            'yellow_times': {'1 EBLT': 4, '2WB': 4.5},
            'red_clearance': {'1 EBLT': 1, '2WB': 1}
        }
        
        processor = TimingPreprocessor()
        result = processor.extract_timing_features(timing_data, plan_number=25)
        
        assert result['plan_used'] == 25
        assert 'cycle_length' in result
        assert result['cycle_length'] > 0
        assert 'green_splits' in result
        assert result['num_phases'] == 8
    
    def test_missing_plan(self):
        """Test handling of missing plan."""
        timing_data = {
            'plans': {
                25: {'phase_greens': {'1 EBLT': 15, '2WB': 30}, 'offset': None}
            },
            'yellow_times': {},
            'red_clearance': {}
        }
        
        processor = TimingPreprocessor()
        
        # Request plan 99 which doesn't exist - should fallback to 25
        result = processor.extract_timing_features(timing_data, plan_number=99)
        
        assert result['plan_used'] == 25


class TestIntegration:
    """Integration tests for the data pipeline."""
    
    def test_ingest_all_with_real_data(self):
        """Test ingesting all data from the actual data directories."""
        # Check if volume and times directories exist
        volume_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'volume')
        times_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'times')
        
        if not os.path.exists(volume_dir) or not os.path.exists(times_dir):
            pytest.skip("Data directories not found")
        
        result = ingest_all(volume_dir=volume_dir, times_dir=times_dir)
        
        assert 'volumes' in result
        assert 'timings' in result
        assert 'matched_intersections' in result
        assert len(result['matched_intersections']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
