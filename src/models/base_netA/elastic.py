from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torch.distributions import Uniform

class Elastic(nn.Module):
    def __init__(self, nz, transformation, datasetmean, datasetstd, neurons=6):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.log_alpha = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.log_disp = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.disp_scale = 0.1
        self.random_seed = 42
        
    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        alpha = torch.exp(self.log_alpha)
        disp = torch.exp(self.log_disp).clamp(0, 1)
        img_transformed = elastic_transform_2d(x, 
                                                kernel_size=(3,3), 
                                                sigma=sigma, 
                                                alpha=alpha, 
                                                disp_scale=disp,
                                                random_seed=self.random_seed)
        return img_transformed, None


def elastic_transform_2d(tensor: torch.Tensor, 
                         kernel_size: Tuple[int, int] = (3,3), 
                         sigma: Tuple[float, float] = (4.,4.), 
                         alpha: Tuple[float, float] = (32.,32.),
                         disp_scale: Tuple[float, float] = (0.1, 0.1), 
                         random_seed: Optional = None) -> torch.Tensor:
    r"""Applies elastic transform of images as described in [Simard2003]_.

    Args:
        img (torch.Tensor): input image.
        kernel_size (Tuple[int, int]): the size of the Gaussian kernel. Default:(3,3).
        sigma (Tuple[float, float]): the standard deviation of the Gaussian in the y and x directions, respecitvely. 
                                     Larger sigma results in smaller pixel displacements. Default:(4,4).
        alpha (Tuple[float, float]):  the scaling factor that controls the intensity of the deformation
                                  in the y and x directions, respectively. Default:(32,32).
        disp_scale (Tuple[float, float]):  the scaling factor that controls the intensity of the displacement
                                  in the y and x directions, respectively. Default:(0.1, 0.1).
        random_seed (Optional): random seed for generating the displacement vector. Default:None.
        

    Returns:
        img (torch.Tensor): the elastically transformed input image.

    References:
        [Simard2003]: Simard, Steinkraus and Platt, "Best Practices for
                      Convolutional Neural Networks applied to Visual Document Analysis", in
                      Proc. of the International Conference on Document Analysis and
                      Recognition, 2003.
    """
    generator = torch.Generator(device='cpu')
    if random_seed is not None:
        generator.manual_seed(random_seed)

    n, c, h, w = tensor.shape
    
    # Get Gaussian kernel for 'y' and 'x' displacement
    kernel_y = get_gaussian_kernel2d(kernel_size, (sigma[0], sigma[0]))[None]
    kernel_x = get_gaussian_kernel2d(kernel_size, (sigma[1], sigma[1]))[None]

    # Convolve over a random displacement matrix and scale them with 'alpha'
    disp = torch.rand(n, 2, h, w, generator=generator).to(device=tensor.device)
    disp[:,0] *= disp_scale[0]
    disp[:,1] *= disp_scale[1]

    disp_y = kornia.filters.filter2D(disp[:,[0]], kernel=kernel_y, border_type='reflect') * alpha[0]
    disp_x = kornia.filters.filter2D(disp[:,[1]], kernel=kernel_x, border_type='reflect') * alpha[1]

    # scale displacements
    disp_y *= alpha[0]
    disp_x *= alpha[1]

    # stack and normalize displacement
    disp = torch.cat([disp_y, disp_x], dim=1).squeeze(0).permute(0,2,3,1)
    
    # Warp image based on displacement matrix
    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)
    warped =  F.grid_sample(tensor, (grid + disp).clamp(-1,1), align_corners=True)
    
    return warped


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float().to(device=sigma.device) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.
    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`
    Examples::
        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])
        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d

def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.
    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        force_even (bool): overrides requirement for odd kernel size.
    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.
    Shape:
        - Output: :math:`(\text{kernel_size})`
    Examples::
        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])
        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d