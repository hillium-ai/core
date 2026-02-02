"""
Test suite for MelodiCompressor implementation.

This file contains unit tests for the MELODI Memory Layer implementation
as specified in WP-031.
"""

import numpy as np
import pytest
from unittest.mock import patch

from loqus_core.melodi_compressor import MelodiCompressor


class TestMelodiCompressor:
    """
    Unit tests for MelodiCompressor class.
    """
    
    def test_init(self):
        """
        Test initialization of MelodiCompressor.
        """
        compressor = MelodiCompressor()
        assert compressor is not None
        
    def test_compress_valid_input(self):
        """
        Test compression with valid input.
        """
        compressor = MelodiCompressor()
        activations = np.array([1.0, 2.0, 3.0])
        
        # Should not raise an exception
        result = compressor.compress(activations)
        assert isinstance(result, np.ndarray)
        
    def test_compress_invalid_input(self):
        """
        Test compression with invalid input.
        """
        compressor = MelodiCompressor()
        
        # Test with invalid type
        with pytest.raises(ValueError):
            compressor.compress("invalid")
            
        # Test with empty array
        with pytest.raises(ValueError):
            compressor.compress(np.array([]))
            
    def test_decompress_valid_input(self):
        """
        Test decompression with valid input.
        """
        compressor = MelodiCompressor()
        compressed = np.array([1.0, 2.0, 3.0])
        
        # Should not raise an exception
        result = compressor.decompress(compressed)
        assert isinstance(result, np.ndarray)
        
    def test_decompress_invalid_input(self):
        """
        Test decompression with invalid input.
        """
        compressor = MelodiCompressor()
        
        # Test with invalid type
        with pytest.raises(ValueError):
            compressor.decompress("invalid")
            
        # Test with empty array
        with pytest.raises(ValueError):
            compressor.decompress(np.array([]))
            
    def test_extract_activations_valid_input(self):
        """
        Test activation extraction with valid input.
        """
        compressor = MelodiCompressor()
        
        # Should not raise an exception
        result = compressor.extract_activations(0)
        assert isinstance(result, np.ndarray)
        
    def test_extract_activations_invalid_input(self):
        """
        Test activation extraction with invalid input.
        """
        compressor = MelodiCompressor()
        
        # Test with invalid type
        with pytest.raises(ValueError):
            compressor.extract_activations("invalid")
            
        # Test with negative index
        with pytest.raises(ValueError):
            compressor.extract_activations(-1)
            
    def test_store_compressed_state_valid_input(self):
        """
        Test storing compressed state with valid input.
        """
        compressor = MelodiCompressor()
        state = np.array([1.0, 2.0, 3.0])
        
        # Should not raise an exception
        compressor.store_compressed_state(state, "/tmp/test_state.npy")
        
    def test_store_compressed_state_invalid_input(self):
        """
        Test storing compressed state with invalid input.
        """
        compressor = MelodiCompressor()
        
        # Test with invalid state type
        with pytest.raises(ValueError):
            compressor.store_compressed_state("invalid", "/tmp/test_state.npy")
            
        # Test with invalid path
        with pytest.raises(ValueError):
            compressor.store_compressed_state(np.array([1.0, 2.0, 3.0]), "")