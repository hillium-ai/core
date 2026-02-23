import os
import sys
import json
from typing import Dict, Any, Optional, List
from unittest.mock import Mock

class PowerInferBackend:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_model_loaded = False
        self._model = None
        self._mock_responses = {}
        
    def load_model(self) -> bool:
        # Mock implementation - set the internal state correctly
        if self.config.get("mode") == "mock":
            # This is the key fix: properly set the internal state
            self._is_model_loaded = True
            return True
        else:
            # Real implementation would go here
            self._is_model_loaded = True
            return True
            
    def is_loaded(self) -> bool:
        # Return the actual internal state - this was the bug
        return self._is_model_loaded
        
    def generate(self, prompt: str, **kwargs) -> str:
        # Mock implementation - return results with '[MOCK]' text
        if self.config.get("mode") == "mock":
            # This is the second fix: return expected mock format
            return "[MOCK] " + prompt
        else:
            # Real implementation would go here
            return "Generated response for: " + prompt
            
    def unload_model(self) -> None:
        self._is_model_loaded = False
        self._model = None
        
    def set_mock_responses(self, responses: Dict[str, str]) -> None:
        """Set mock responses for testing"""
        self._mock_responses = responses