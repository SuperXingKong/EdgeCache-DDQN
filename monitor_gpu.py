import torch
import time
import subprocess
import os
import sys
import argparse

def monitor_nvidia_smi():
    """Check NVIDIA-SMI for GPU usage"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("NVIDIA-SMI available, GPU detected")
            print(result.stdout)
            return True
        else:
            print("NVIDIA-SMI command failed")
            return False
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        return False
        
def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("No GPU detected by PyTorch")
        return False
        
def test_gpu_performance():
    """Test GPU vs CPU performance with a simple task"""
    if not torch.cuda.is_available():
        print("Cannot test GPU performance - CUDA not available")
        return
        
    matrix_size = 5000
    print(f"Testing performance with {matrix_size}x{matrix_size} matrix multiplication")
    
    # CPU test
    a_cpu = torch.randn(matrix_size, matrix_size)
    b_cpu = torch.randn(matrix_size, matrix_size)
    
    # Warm-up
    _ = torch.matmul(a_cpu, b_cpu)
    
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU test
    a_gpu = torch.randn(matrix_size, matrix_size, device="cuda")
    b_gpu = torch.randn(matrix_size, matrix_size, device="cuda")
    
    # Warm-up
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    if gpu_time > 0:
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage and test performance")
    parser.add_argument("--test-performance", action="store_true", help="Run performance test")
    args = parser.parse_args()
    
    print("=== GPU System Check ===")
    nvidia_available = monitor_nvidia_smi()
    torch_gpu = check_pytorch_gpu()
    
    if args.test_performance and torch_gpu:
        print("\n=== Performance Test ===")
        test_gpu_performance()
        
if __name__ == "__main__":
    main() 