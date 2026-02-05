"""
Integration test for ReStraV Visual Validator
"""

def test_visual_validator_exists():
    """
    Test that the visual validator module can be imported
    """
    try:
        # This should work if the module is properly implemented
        from crates.aegis_core.src.layer7.visual_validator import ReStraVDetector
        assert True
    except ImportError as e:
        assert False, f"Failed to import ReStraVDetector: {e}"

def test_visual_validator_interface():
    """
    Test that ReStraVDetector implements the expected interface
    """
    try:
        from crates.aegis_core.src.layer7.visual_validator import ReStraVDetector, VisualValidator
        
        detector = ReStraVDetector()
        # Check that it implements the trait
        assert hasattr(detector, 'analyze')
        assert hasattr(detector, 'set_thresholds')
        assert hasattr(detector, 'get_stats')
        assert True
    except Exception as e:
        assert False, f"ReStraVDetector does not implement expected interface: {e}"
