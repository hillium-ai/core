import torch
import torch.nn as nn
from typing import Tuple, Optional


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
    
    def __init__(self, map_size: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.map_size = map_size
        self.hidden_dim = hidden_dim
        
        # Store the expected input size for validation
        self.expected_input_size = 2 + 2 + map_size * map_size
        
        # MLP for processing the combined features
        # We'll make this work with variable input sizes by using a flexible approach
        self.mlp = nn.Sequential(
            nn.Linear(self.expected_input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8 + 1)  # 8 actions + 1 value
        )
        
        # Separate policy and value outputs
        self.policy_head = nn.Linear(8 + 1, 8)  # 8 actions
        self.value_head = nn.Linear(8 + 1, 1)   # 1 value
        
    def forward(self, start: torch.Tensor, goal: torch.Tensor, local_map: torch.Tensor):
        """
        Forward pass through the rerooter network.
        
        Args:
            start: Tensor of shape (batch_size, 2) representing start positions
            goal: Tensor of shape (batch_size, 2) representing goal positions
            local_map: Tensor of shape (batch_size, 1, map_size, map_size) representing local map
            
        Returns:
            policy_logits: Tensor of shape (batch_size, 8) representing action logits
            value: Tensor of shape (batch_size, 1) representing value estimate
        """
        # Flatten the local map
        flat_map = local_map.view(local_map.size(0), -1)
        
        # Concatenate all inputs
        x = torch.cat([start, goal, flat_map], dim=1)
        
        # Process through MLP
        combined = self.mlp(x)
        
        # Split into policy and value
        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return policy_logits, value

    def forward_scriptable(self, start: torch.Tensor, goal: torch.Tensor, local_map: torch.Tensor):
        """
        Forward pass that can be used with TorchScript
        """
        return self.forward(start, goal, local_map)

    def export_to_torchscript(self, path: str):
        """Export the model to TorchScript"""
        # Create dummy inputs for tracing
        dummy_start = torch.randn(1, 2)
        dummy_goal = torch.randn(1, 2)
        dummy_local_map = torch.randn(1, 1, self.map_size, self.map_size)
        
        # Trace the model
        traced = torch.jit.trace(self, (dummy_start, dummy_goal, dummy_local_map))
        torch.jit.save(traced, path)
        return traced