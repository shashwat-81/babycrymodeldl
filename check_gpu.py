import torch
import os

print("🔍 GPU Information Check")
print("=" * 40)

# Set environment for discrete GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Set to GPU 0 and test
    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    print(f"\nCurrent Device: {current_device}")
    print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
    
    # Test tensor operations
    try:
        x = torch.randn(1000, 1000, device='cuda:0')
        y = torch.randn(1000, 1000, device='cuda:0')
        z = torch.mm(x, y)
        print(f"✅ GPU {current_device} tensor operations working")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.3f} GB")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        print("✅ Memory cleared")
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("❌ No CUDA GPUs available")

print("\n" + "=" * 40)