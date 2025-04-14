import torch
from ddqn import DuelingDQN
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_model_inference(model, input_tensor, device, num_runs=100):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure time
    import time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

# Example state and action dimensions from the EdgeCache environment
state_dim = 200   # Adjust based on your actual environment
action_dim = 300  # Adjust based on your actual environment

# Create model
model = DuelingDQN(state_dim, action_dim, hidden_dim=128)

# Count parameters
total_params = count_parameters(model)
print(f"Model has {total_params:,} trainable parameters")

# Create a sample input
sample_input = torch.randn(1, state_dim)  # Batch size of 1

# Profile on CPU
cpu_device = torch.device("cpu")
cpu_time = profile_model_inference(model, sample_input, cpu_device)
print(f"Average inference time on CPU: {cpu_time*1000:.2f} ms")

# Profile on GPU if available
if torch.cuda.is_available():
    cuda_device = torch.device("cuda")
    cuda_time = profile_model_inference(model, sample_input, cuda_device)
    print(f"Average inference time on GPU: {cuda_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/cuda_time:.2f}x")
else:
    print("CUDA not available for comparison") 