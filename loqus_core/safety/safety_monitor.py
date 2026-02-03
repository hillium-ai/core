from .layer7 import CognitiveSafetyValidator

class SafetyMonitor:
    """
    Safety Monitor that wraps the Cognitive Safety Validator
    """
    
    def __init__(self):
        self.validator = CognitiveSafetyValidator()
        
    def check_safety(self, plan):
        """
        Check if a plan is safe
        """
        # Create a mock context for validation
        context = {
            "target": "unknown",
            "visible_objects": [],
            "plan": plan
        }
        
        result = self.validator.validate(context)
        
        return SafetyCheckResult(
            is_safe=result == "approved",
            reason="Plan passed safety validation"
        )


class SafetyCheckResult:
    def __init__(self, is_safe: bool, reason: str):
        self.is_safe = is_safe
        self.reason = reason