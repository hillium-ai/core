import logging
import json
import fcntl
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Any, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)

class RecordType(str, Enum):
    """Type classification for memory records."""
    Observation = "Observation"      # sensor facts, detections, measured state
    Action = "Action"                # executed action (what we did)
    Outcome = "Outcome"              # measured result of an action
    DerivedSummary = "DerivedSummary"# compression/summaries derived from records
    Prediction = "Prediction"        # forecast, simulation output
    Plan = "Plan"                    # candidate plan (not executed)

class TimeReference(str, Enum):
    """Temporal classification for records."""
    Past = "Past"
    Present = "Present"
    FutureHypothesis = "FutureHypothesis"

@dataclass
class BeliefRecord:
    """
    A belief record representing an observation, action, or insight.
    
    Aligned with HilliumOS v8.6+ Epistemic Discipline standard.
    Records ≠ Predictions - this distinction is critical for Aegis Layer 7.
    """
    # Required fields
    belief_type: str           # "observation", "user_correction", "inferred"
    content: str               # The belief content
    
    # Auto-generated
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Classification
    record_type: RecordType = RecordType.Observation
    time_reference: TimeReference = TimeReference.Past
    source: str = "system"     # sensor / solver / llm / human / synapp
    
    # Scoring
    confidence: float = 0.5    # 0.0 - 1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    causal_parent_ids: Sequence[str] = field(default_factory=tuple)
    ttl_seconds: Optional[int] = None  # For predictions/plans
    
    def __post_init__(self):
        """Validate fields on initialization."""
        if not isinstance(self.belief_type, str) or not self.belief_type:
            logger.error(f"Invalid belief_type: {self.belief_type}")
            raise ValueError("belief_type must be a non-empty string")
        
        if not isinstance(self.content, str):
            logger.error(f"Invalid content type: {type(self.content)}")
            raise ValueError("content must be a string")
        
        if not 0.0 <= self.confidence <= 1.0:
            logger.error(f"confidence out of range: {self.confidence}")
            raise ValueError("confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            'id': self.id,
            'belief_type': self.belief_type,
            'content': self.content,
            'record_type': self.record_type.value,
            'time_reference': self.time_reference.value,
            'source': self.source,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata,
            'causal_parent_ids': list(self.causal_parent_ids),
            'ttl_seconds': self.ttl_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BeliefRecord':
        """Deserialize from dictionary."""
        return cls(
            id=data['id'],
            belief_type=data['belief_type'],
            content=data['content'],
            record_type=RecordType(data.get('record_type', 'Observation')),
            time_reference=TimeReference(data.get('time_reference', 'Past')),
            source=data.get('source', 'system'),
            confidence=data['confidence'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            metadata=data.get('metadata', {}),
            causal_parent_ids=tuple(data.get('causal_parent_ids', [])),
            ttl_seconds=data.get('ttl_seconds'),
        )

@dataclass
class SessionSummary:
    """
    Compresses a conversation session into 3 lines.
    Used for consolidation to episodic memory.
    """
    summary: str              # The 3-line summary
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    source_belief_ids: Sequence[str] = field(default_factory=tuple)
    
    def __post_init__(self):
        if not isinstance(self.summary, str):
            logger.error(f"Invalid summary type: {type(self.summary)}")
            raise ValueError("summary must be a string")
        
        # Warn if summary exceeds 3 lines
        lines = self.summary.strip().split('\n')
        if len(lines) > 3:
            logger.warning(f"SessionSummary exceeds 3 lines: {len(lines)} lines")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'summary': self.summary,
            'created_at': self.created_at.isoformat(),
            'source_belief_ids': list(self.source_belief_ids),
        }


class BeliefStore:
    """Simple file-based belief storage for MVP."""
    
    def __init__(self, db_path: str = ".levitate/beliefs.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write_db({})
    
    def _read_db(self) -> Dict[str, Any]:
        with open(self.db_path, 'r') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                content = f.read()
                if not content.strip():
                    return {}
                return json.loads(content)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    
    def _write_db(self, data: Dict[str, Any]):
        with open(self.db_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    
    def create_belief(self, belief: BeliefRecord) -> str:
        """Store a new belief."""
        db = self._read_db()
        db[belief.id] = belief.to_dict()
        self._write_db(db)
        logger.info(f"Created belief: {belief.id}")
        return belief.id
    
    def get_belief(self, belief_id: str) -> Optional[BeliefRecord]:
        """Retrieve a belief by ID."""
        db = self._read_db()
        data = db.get(belief_id)
        if data:
            return BeliefRecord.from_dict(data)
        return None
    
    def list_beliefs(self) -> list[BeliefRecord]:
        """List all beliefs."""
        db = self._read_db()
        return [BeliefRecord.from_dict(v) for v in db.values()]
    
    def delete_belief(self, belief_id: str) -> bool:
        """Delete a belief."""
        db = self._read_db()
        if belief_id in db:
            del db[belief_id]
            self._write_db(db)
            logger.info(f"Deleted belief: {belief_id}")
            return True
        return False