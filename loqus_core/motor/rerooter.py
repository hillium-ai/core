import torch
import torch.nn as nn


class RerooterNetwork(nn.Module):
    """
    RerooterNetwork implements the policy network for √LTS planning.
    
    Input:
        start: [x, y] (normalized coordinates)
        goal: [x, y] (normalized coordinates)
        local_map: Tensor (Grid around robot)
    
    Output:
        policy_logits: Action probabilities (8-connected grid steps)
        value: Estimated steps-to-goal
    """
    
    def __init__(self):
        super().__init__()
        # MLP for policy and value prediction
        self.policy_head = nn.Sequential(
            nn.Linear(2 + 2 + 64*64, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8-connected grid steps
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 + 2 + 64*64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, start, goal, local_map):
        # Validate inputs
        if not isinstance(start, torch.Tensor) or not isinstance(goal, torch.Tensor):
            raise ValueError("start and goal must be torch tensors")
        
        if not isinstance(local_map, torch.Tensor):
            raise ValueError("local_map must be a torch tensor")
        
        # Flatten local_map
        x = torch.cat([start, goal, local_map.view(-1)], dim=0)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def export(self, path):
        """Export the model to TorchScript"""
        # Create dummy inputs for tracing
        dummy_start = torch.randn(2)
        dummy_goal = torch.randn(2)
        dummy_local_map = torch.randn(64, 64)
        
        # Trace the model
        traced = torch.jit.trace(self, (dummy_start, dummy_goal, dummy_local_map))
        torch.jit.save(traced, path)
