from loqus_core.motor.rerooter import RerooterNetwork, RerooterController
import torch

print('Testing RerooterNetwork...')
model = RerooterNetwork()
start = torch.randn(2)
goal = torch.randn(2)
local_map = torch.randn(8, 8)
policy_logits, value = model(start, goal, local_map)
print(f'Forward pass successful. Policy logits shape: {policy_logits.shape}, Value shape: {value.shape}')

print('Testing RerooterController...')
controller = RerooterController()
action, value = controller.plan([0.0, 0.0], [1.0, 1.0], torch.randn(8, 8).tolist())
print(f'Planning successful. Action: {action}, Value: {value}')

print('All tests passed!')