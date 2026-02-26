"""Integration tests for Cognitive Safety Validator with ReStraV"""

import unittest
from unittest.mock import Mock
aegis = pytest = None
try:
    from crates.aegis_core.layer7.cognitive_safety import CognitiveSafetyValidator
    from crates.restrav_validator import Image
except ImportError:
    import pytest
    pytestmark = pytest.mark.skip(reason="aegis_core/restrav_validator Python bindings not installed")

class TestCognitiveSafetyIntegration(unittest.TestCase):
    def test_cognitive_safety_validator_creation(self):
        """Test that CognitiveSafetyValidator can be created"""
        validator = CognitiveSafetyValidator()
        self.assertIsNotNone(validator)
        
    def test_validate_visual_input_method_exists(self):
        """Test that validate_visual_input method exists"""
        validator = CognitiveSafetyValidator()
        self.assertTrue(hasattr(validator, 'validate_visual_input'))
        
    def test_validate_visual_input_with_frames(self):
        """Test validate_visual_input with mock frames"""
        validator = CognitiveSafetyValidator()
        frames = [Image(640, 480), Image(640, 480)]
        
        # Should not raise an exception
        result = validator.validate_visual_input(frames)
        # We can't fully test the logic without the feature flag enabled
        # but we can verify the method signature works

if __name__ == '__main__':
    unittest.main()
