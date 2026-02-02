import torch
import torch.nn as nn
import torch.nn.functional as F

class RerooterNetwork(nn.Module):
    def __init__(self):
        super(RerooterNetwork, self).__init__()
        # Simple MLP for policy and value prediction
        self.fc1 = nn.Linear(2 + 2 + 100, 128)  # start + goal + local_map
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 8)  # 8 actions (8-connected grid)
        self.value_head = nn.Linear(64, 1)

    def forward(self, start, goal, local_map):
        # Concatenate inputs
        x = torch.cat([start, goal, local_map], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value


# Export functionality for TorchScript
if __name__ == "__main__":
    # Create and export a model for testing
    model = RerooterNetwork()
    
    # Create dummy inputs
    start = torch.randn(1, 2)
    goal = torch.randn(1, 2)
    local_map = torch.randn(1, 100)
    
    # Test forward pass
    policy_logits, value = model(start, goal, local_map)
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Export to TorchScript using script (not trace)
    scripted_model = torch.jit.script(model)
    scripted_model.save("rerooter.pt")
    print("Model exported to rerooter.pt")