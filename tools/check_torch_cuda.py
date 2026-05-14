import torch
import sys
print('torch:', getattr(torch, '__version__', 'unknown'))
print('cuda_available:', torch.cuda.is_available())
try:
    print('cuda_count:', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('cuda_name0:', torch.cuda.get_device_name(0))
except Exception as e:
    print('cuda_info_error:', e)
sys.exit(0)
