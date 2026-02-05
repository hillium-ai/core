"""
Verification script for WP-042 ReStraV Visual Validator Implementation
"""

def test_implementation_exists():
    """
    Verify that all required components exist
    """
    # Check that the main files exist
    import os
    
    files_to_check = [
        'crates/aegis_core/src/layer7/visual_validator.rs',
        'crates/aegis_core/src/layer7/mod.rs',
        'crates/aegis_core/Cargo.toml',
        'tests/test_visual_validator.py',
    ]
    
    for file_path in files_to_check:
        assert os.path.exists(file_path), f"Required file {file_path} does not exist"
    
    print("✅ All required files exist")
    
    # Check that the feature flag is defined
    with open('crates/aegis_core/Cargo.toml', 'r') as f:
        content = f.read()
        assert 'visual-validation' in content, "visual-validation feature not found in Cargo.toml"
        assert 'onnxruntime' in content or 'image' in content, "Required dependencies not found"
    
    print("✅ Feature flag and dependencies correctly defined")
    
    print("✅ WP-042 implementation verification complete")
