from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class PlanStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SelfEvaluation:
    score: float
    justification: str
    missing_aspects: List[str] = field(default_factory=list)

@dataclass
class InspectionReport:
    auditor_id: str
    verdict: str
    critique: str
    timestamp: float

@dataclass
class ActionPlan:
    goal: str
    steps: List[str]
    status: PlanStatus = PlanStatus.PENDING
    depth: int = 1
    evaluation: Optional[SelfEvaluation] = None
    inspections: List[InspectionReport] = field(default_factory=list)
