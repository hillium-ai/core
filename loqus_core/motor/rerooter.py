import torch
import torch.nn as nn
import torch.nn.functional as F


class RerooterNetwork(nn.Module):
    def __init__(self):
        super(RerooterNetwork, self).__init__()
        # Simple MLP for policy and value prediction
        self.fc1 = nn.Linear(2 + 2 + 64, 128)  # start + goal + local_map (flattened)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, 8)  # 8 actions (8-connected grid)
        self.value_head = nn.Linear(64, 1)

    def forward(self, start, goal, local_map):
        # Flatten local_map to (batch_size, 64)
        local_map = local_map.view(local_map.size(0), -1)
        
        # Ensure start and goal are batched (add batch dimension if needed)
        if start.dim() == 1:
            start = start.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        
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
    
    # Create dummy inputs with batch dimension
    start = torch.randn(1, 2)
    goal = torch.randn(1, 2)
    local_map = torch.randn(1, 8, 8)  # 8x8 grid
    
    # Test forward pass
    policy_logits, value = model(start, goal, local_map)
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Export to TorchScript using script (not trace)
    scripted_model = torch.jit.script(model)
    scripted_model.save("rerooter.pt")
    print("Model exported to rerooter.pt")