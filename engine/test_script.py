import torch
print(torch.__version__)

print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)