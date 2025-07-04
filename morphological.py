import torch
import torch.nn.functional as F
from typing import Union, Tuple

def erosion_binary(image: torch.Tensor, kernel: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """conv2d erosion, only suitable for binary inputs."""
    original_dim = image.dim()
    if original_dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    
    kernel_sum = kernel.sum()
    conv_kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    
    result = image.clone()
    
    for _ in range(iterations):
        # Apply grouped convolution (each channel processed independently)
        conv_result = F.conv2d(result, conv_kernel, padding=(kernel.shape[0]//2, kernel.shape[1]//2), groups=C)
        # Output is 1 only where all kernel positions were 1
        result = (conv_result >= kernel_sum - 0.5).float()
    
    if original_dim == 2:
        result = result.squeeze(0).squeeze(0)
    elif original_dim == 3:
        result = result.squeeze(0)
    
    return result


def dilation_binary(image: torch.Tensor, kernel: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """conv2d dilation, only suitable for binary inputs."""
    original_dim = image.dim()
    if original_dim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    
    # Create convolution kernel for each channel
    conv_kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    
    result = image.clone()
    
    for _ in range(iterations):
        # Apply grouped convolution
        conv_result = F.conv2d(result, conv_kernel, padding=(kernel.shape[0]//2, kernel.shape[1]//2), groups=C)
        # Output is 1 where any kernel position was 1
        result = (conv_result > 0.5).float()
    
    if original_dim == 2:
        result = result.squeeze(0).squeeze(0)
    elif original_dim == 3:
        result = result.squeeze(0)
    
    return result

erosion = erosion_binary
dilation = dilation_binary

# Common structuring elements
def get_kernel(size: Union[int, Tuple[int, int]], shape: str = "square") -> torch.Tensor:
    """
    Create common structuring elements.
    
    Args:
        shape: 'cross', 'square', 'circle'
        size: Size of the structuring element (int or (height, width))
        
    Returns:
        Structuring element tensor
    """
    if isinstance(size, int):
        size = (size, size)
    
    h, w = size
    
    if shape == 'square':
        return torch.ones(h, w)
    
    elif shape == 'cross':
        kernel = torch.zeros(h, w)
        kernel[h//2, :] = 1
        kernel[:, w//2] = 1
        return kernel
    
    elif shape == 'circle':
        kernel = torch.zeros(h, w)
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 2
        
        for i in range(h):
            for j in range(w):
                if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                    kernel[i, j] = 1
        return kernel
    
    else:
        raise ValueError(f"Unknown shape: {shape}")


# Compound operations
def opening(image: torch.Tensor, kernel: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    Morphological opening (erosion followed by dilation).
    Useful for removing small objects/noise.
    """
    return dilation(erosion(image, kernel, iterations), kernel, iterations)


def closing(image: torch.Tensor, kernel: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    Morphological closing (dilation followed by erosion).
    Useful for filling small holes.
    """
    return erosion(dilation(image, kernel, iterations), kernel, iterations)


def gradient(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Morphological gradient (dilation - erosion).
    Useful for edge detection.
    """
    return dilation(image, kernel) - erosion(image, kernel)
