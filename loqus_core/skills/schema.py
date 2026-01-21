from pydantic import BaseModel, ValidationError, validator
from typing import List, Dict, Any
import json
import re


class SkillPackage(BaseModel):
    skill_id: str
    name: str
    description: str
    code: str
    test_cases: List[Dict[str, Any]]

    class Config:
        # Allow extra fields if needed (for forward compatibility)
        extra = "allow"

    @validator('skill_id')
    def skill_id_must_be_valid_uuid(cls, v):
        # UUID format validation
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        if not uuid_pattern.match(v):
            raise ValueError('skill_id must be a valid UUID')
        return v

    @validator('name', 'description')
    def validate_string_fields(cls, v):
        # Basic sanitization
        if not isinstance(v, str):
            raise ValueError('Name and description must be strings')
        sanitized = v.replace('<', '&lt;').replace('>', '&gt;')
        return sanitized

    @validator('code')
    def validate_code_field(cls, v):
        # Basic validation for code field
        if not isinstance(v, str):
            raise ValueError('Code must be a string')
        return v

    @validator('test_cases')
    def validate_test_cases(cls, v):
        # Validate that test_cases is a list of dictionaries with required structure
        if not isinstance(v, list):
            raise ValueError('test_cases must be a list')
        for i, test_case in enumerate(v):
            if not isinstance(test_case, dict):
                raise ValueError(f'test_cases[{i}] must be a dictionary')
            if 'input' not in test_case or 'output' not in test_case:
                raise ValueError(f'test_cases[{i}] must contain "input" and "output" keys')
        return v


# Sample test case
if __name__ == "__main__":
    valid_skill = {
        "skill_id": "12345678-1234-1234-1234-123456789012",
        "name": "Test Skill",
        "description": "A test skill for demonstration",
        "code": "def test_function(): return 'Hello, World!'",
        "test_cases": [
            {"input": {"x": 1}, "output": "Hello, World!"}
        ]
    }

    try:
        skill_package = SkillPackage(**valid_skill)
        print("Skill package validation passed.")
    except ValidationError as e:
        print("Validation failed:", e)
