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
        # Flatten local_map
        x = torch.cat([start, goal, local_map.view(-1)], dim=0)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
