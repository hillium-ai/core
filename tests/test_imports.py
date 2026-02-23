import sys
sys.path.insert(0, '/workspace')

# Test that core modules can be imported
try:
    from loqus_core.motor.rerooter_network import RerooterNetwork
    print('✓ RerooterNetwork import successful')
    
    # Test instantiation
    network = RerooterNetwork()
    print('✓ RerooterNetwork instantiation successful')
    
    # Test forward method exists
    print('✓ forward method exists:', hasattr(network, 'forward'))
    print('✓ forward_scriptable method exists:', hasattr(network, 'forward_scriptable'))
    
    print('All core imports working!')
    
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()