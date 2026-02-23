"""Test file for ReStraV Visual Validator implementation"""

import unittest
from unittest.mock import Mock
from crates.restrav_validator import ReStraVDetector, VisualValidator, Image

class TestReStraVImplementation(unittest.TestCase):
    def test_restrav_detector_creation(self):
        """Test that ReStraVDetector can be created"""
        detector = ReStraVDetector()
        self.assertIsNotNone(detector)
        
    def test_visual_validator_trait(self):
        """Test that ReStraVDetector implements VisualValidator trait"""
        detector = ReStraVDetector()
        self.assertTrue(hasattr(detector, 'analyze'))
        self.assertTrue(hasattr(detector, 'set_thresholds'))
        self.assertTrue(hasattr(detector, 'get_stats'))
        
    def test_analyze_method_signature(self):
        """Test the analyze method signature"""
        detector = ReStraVDetector()
        image = Image(640, 480)
        
        # Should not raise an exception
        result = detector.analyze([image])
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_synthetic'))
        self.assertTrue(hasattr(result, 'curvature_score'))
        self.assertTrue(hasattr(result, 'stepwise_distance'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'frame_anomalies'))
        
    def test_set_thresholds_method(self):
        """Test the set_thresholds method"""
        detector = ReStraVDetector()
        thresholds = {
            'min_confidence': 0.9,
            'curvature_threshold': 0.7,
            'distance_threshold': 0.6
        }
        
        # Should not raise an exception
        detector.set_thresholds(thresholds)
        
    def test_get_stats_method(self):
        """Test the get_stats method"""
        detector = ReStraVDetector()
        stats = detector.get_stats()
        self.assertIsNotNone(stats)
        self.assertTrue(hasattr(stats, 'total_frames'))
        self.assertTrue(hasattr(stats, 'synthetic_detections'))
        self.assertTrue(hasattr(stats, 'avg_processing_time_ms'))

if __name__ == '__main__':
    unittest.main()
