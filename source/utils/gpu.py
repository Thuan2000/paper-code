import GPUtil
import os

# Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getAvailable('memory', limit=2)
print(DEVICE_ID_LIST)
print('Run on GPU', ','.join([str(i) for i in DEVICE_ID_LIST]))

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in DEVICE_ID_LIST])