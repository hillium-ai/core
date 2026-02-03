import torch
import torch.nn as nn
import torch.nn.functional as F


class RerooterNetwork(nn.Module):
    def __init__(self, map_size=32, hidden_dim=128):
        super(RerooterNetwork, self).__init__()
        
        # Define the network layers
        self.map_size = map_size
        self.hidden_dim = hidden_dim
        
        # Input layer
        self.input_layer = nn.Linear(2 + 2 + map_size * map_size, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(2)  # 2 hidden layers
        ])
        
        # Output layers
        self.policy_logits = nn.Linear(hidden_dim, 8)  # 8-connected grid steps
        self.value = nn.Linear(hidden_dim, 1)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, start, goal, local_map):
        # Concatenate inputs
        x = torch.cat([start, goal, local_map.view(local_map.size(0), -1)], dim=1)
        
        # Forward pass through network
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        
        # Output
        policy_logits = self.policy_logits(x)
        value = self.value(x)
        
        return policy_logits, value
    
    def forward_scriptable(self, start, goal, local_map):
        """
        Forward pass optimized for scripting.
        
        Args:
            start: Start position tensor of shape (batch_size, 2)
            goal: Goal position tensor of shape (batch_size, 2)
            local_map: Local map tensor of shape (batch_size, 1, map_size, map_size)
            
        Returns:
            Tuple of (policy_logits, value) tensors
        """
        # This method is designed to be compatible with TorchScript
        return self.forward(start, goal, local_map)