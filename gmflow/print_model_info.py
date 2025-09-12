import torch.nn as nn

def print_conv_kernel_info(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_shape = module.weight.shape
            # weight_shape 是 [out_channels, in_channels, kernel_height, kernel_width]
            kernel_size = (weight_shape[2], weight_shape[3])
            in_channels = weight_shape[1]
            out_channels = weight_shape[0]
            print(f"Layer: {name}")
            print(f"  |-> Weight Shape: {weight_shape}")
            print(f"  |-> In Channels: {in_channels}, Out Channels: {out_channels}")
            print(f"  |-> Inferred Kernel Size: {kernel_size}\n")
