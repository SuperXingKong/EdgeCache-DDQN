import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("Current CUDA Device:", torch.cuda.current_device())
else:
    print("No CUDA device available")

# Check performance
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
print("Using device:", device)

# Create a sample tensor and measure time
import time

# Large matrix multiplication to see performance difference
size = 5000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warmup
_ = torch.matmul(a, b)
if torch.cuda.is_available():
    torch.cuda.synchronize()
    
# Benchmark
start_time = time.time()
c = torch.matmul(a, b)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()

print(f"Matrix multiplication took {end_time - start_time:.4f} seconds") 