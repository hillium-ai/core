import pytest
from loqus_core.cognitive_council.alignment_monitor import (
    CollectiveAlignmentMonitor,
    AlignmentStrategy,
    Vote,
    BiasDetection,
    AlignmentAnalysis
)


def test_vote_dataclass():
    vote = Vote(
        agent_id="agent_1",
        decision="test_decision",
        confidence=0.8,
        reasoning="test_reasoning"
    )
    assert vote.agent_id == "agent_1"
    assert vote.decision == "test_decision"
    assert vote.confidence == 0.8
    assert vote.reasoning == "test_reasoning"


def test_bias_detection_dataclass():
    bias = BiasDetection(
        bias_type="demographic",
        severity=0.7,
        affected_votes=["agent_1"],
        evidence="test evidence"
    )
    assert bias.bias_type == "demographic"
    assert bias.severity == 0.7
    assert bias.affected_votes == ["agent_1"]
    assert bias.evidence == "test evidence"


def test_alignment_analysis_dataclass():
    vote = Vote(
        agent_id="agent_1",
        decision="test_decision",
        confidence=0.8,
        reasoning="test_reasoning"
    )
    analysis = AlignmentAnalysis(
        original_votes=[vote],
        detected_biases=[],
        corrected_decision="corrected_decision",
        equity_score=0.85,
        strategy_applied=AlignmentStrategy.HYBRID
    )
    assert analysis.original_votes == [vote]
    assert analysis.detected_biases == []
    assert analysis.corrected_decision == "corrected_decision"
    assert analysis.equity_score == 0.85
    assert analysis.strategy_applied == AlignmentStrategy.HYBRID


def test_collective_alignment_monitor_init():
    monitor = CollectiveAlignmentMonitor()
    assert monitor.strategy == AlignmentStrategy.HYBRID


def test_collective_alignment_monitor_with_strategy():
    monitor = CollectiveAlignmentMonitor(strategy=AlignmentStrategy.MASK)
    assert monitor.strategy == AlignmentStrategy.MASK


def test_alignment_strategy_enum():
    assert AlignmentStrategy.MASK.value == "mask"
    assert AlignmentStrategy.MIRROR.value == "mirror"
    assert AlignmentStrategy.HYBRID.value == "hybrid"
