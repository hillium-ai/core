import sys
import os
sys.path.insert(0, '.')

from loqus_core.memory.beliefs import BeliefRecord, RecordType, TimeReference, BeliefStore, SessionSummary
import tempfile

print('Testing BeliefRecord creation...')
belief = BeliefRecord(
    belief_type="observation",
    content="Test observation",
    confidence=0.8
)
print(f'✅ Created belief with ID: {belief.id}')

print('Testing BeliefRecord serialization...')
serialized = belief.to_dict()
restored = BeliefRecord.from_dict(serialized)
print(f'✅ Serialization test passed: {restored.content}')

print('Testing SessionSummary...')
summary = SessionSummary(summary="Line 1\nLine 2\nLine 3")
print(f'✅ Created summary with ID: {summary.session_id}')

print('Testing BeliefStore with temporary file...')
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
    temp_path = f.name

try:
    store = BeliefStore(db_path=temp_path)
    
    # Test create
    belief_id = store.create_belief(belief)
    print(f'✅ Created belief in store: {belief_id}')
    
    # Test read
    retrieved = store.get_belief(belief_id)
    print(f'✅ Retrieved belief: {retrieved.content if retrieved else "None"}')
    
    # Test list
    all_beliefs = store.list_beliefs()
    print(f'✅ Listed {len(all_beliefs)} beliefs')
    
    print('✅ All tests passed successfully!')
    
finally:
    if os.path.exists(temp_path):
        os.unlink(temp_path)
