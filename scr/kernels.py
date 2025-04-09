import numpy as np
import torch
from sklearn.base import BaseEstimator
from scr.utils import (
    uniform, rectangle, triangle,
    epanechnikov, biweight, tricube,
    gaussian, silverman
)

VALID_KERNELS_LIST = [
    "uniform",
    "rectangle",
    "triangle",
    "epanechnikov",
    "biweight",
    "tricube",
    "gaussian",
    "silverman",
]

def l2_norm_functional(x, Xt, tau):
    """
    Compute the L2 norm between functional data points x(tau) and Xt(tau).
    :param x: Single functional data point (torch.Tensor or numpy.ndarray of size len(tau)).
    :param Xt: Batch of functional data points (torch.Tensor or numpy.ndarray of size (batch, len(tau))).
    :param tau: Discretized domain [0, 1] (torch.Tensor or numpy.ndarray of size len(tau)).
    :return: L2 norms (torch.Tensor of size batch).
    """
    # Ensure all inputs are torch.Tensors
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=torch.float32)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    if not isinstance(Xt, torch.Tensor):
        Xt = torch.tensor(Xt, dtype=torch.float32)

    # Ensure tau is on the same device as x and Xt
    tau = tau.to(x.device)

    # Compute L2 norm
    diff = x - Xt
    return torch.sqrt(torch.sum(diff**2, dim=-1))  # Compute L2 norm


def space_kernel(kernel, x, Xt, tau, bandwidth):
    """
    Space kernel for functional data using L2 norm.
    :param kernel: Kernel function.
    :param x: Single functional data point (torch.Tensor of size len(tau)).
    :param Xt: Batch of functional data points (torch.Tensor of size (batch, len(tau))).
    :param tau: Discretized domain [0, 1] (torch.Tensor or numpy.ndarray of size len(tau)).
    :param bandwidth: Bandwidth (float).
    :return: Kernel values (torch.Tensor of size batch).
    """
    # Ensure tau is a torch.Tensor
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=torch.float32)

    # Compute L2 norms
    l2_norms = l2_norm_functional(x, Xt, tau) / bandwidth

    # Apply kernel to L2 norms
    kernel_vec_val = kernel(l2_norms)
    return kernel_vec_val




def time_kernel(kernel, aT, tT, bandwidth):
    """
    Time kernel.
    :param kernel: Kernel function.
    :param aT: Scaled time values (batched).
    :param tT: Target time value.
    :param bandwidth: Bandwidth (float).
    :return: Kernel values (vector).
    """
    atT_scaled = (tT - aT) / bandwidth
    return kernel(atT_scaled)

class Kernel(BaseEstimator):
    def __init__(
            self,
            *,
            T=100,
            tau=None,
            bandwidth=1.0,
            space_kernel="gaussian",
            time_kernel="gaussian",
            device="cpu",
            VALID_KERNELS_DIC={
                "uniform": uniform,
                "rectangle": rectangle,
                "triangle": triangle,
                "epanechnikov": epanechnikov,
                "biweight": biweight,
                "tricube": tricube,
                "gaussian": gaussian,
                "silverman": silverman,
            }
    ):
        self.T = T
        self.tau = tau if tau is not None else np.linspace(0, 1, 100)
        self.bandwidth = bandwidth
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.device = device
        self.VALID_KERNELS_DIC = VALID_KERNELS_DIC

        if self.space_kernel in VALID_KERNELS_LIST and self.time_kernel in VALID_KERNELS_LIST:
            self.skernel_name = self.VALID_KERNELS_DIC[self.space_kernel]
            self.tkernel_name = self.VALID_KERNELS_DIC[self.time_kernel]
        else:
            raise ValueError("Kernel type not supported")

    def fit(self, X, t):
        """
        Fit the kernel to the functional data.
        :param X: Functional data (T x len(tau)).
        :param t: Target time index (scalar).
        :return: Weights for time step t.
        """
        X_t = X[t, :]  # Functional data at time t
        a = torch.arange(self.T, dtype=torch.float16, device=self.device)  
        
        # Calculate time kernel values
        time_vals = time_kernel(self.tkernel_name, a / self.T, t / self.T, self.bandwidth)

        # Calculate space kernel values
        space_vals = space_kernel(self.skernel_name, X_t, X, torch.tensor(self.tau, dtype=torch.float32), self.bandwidth)

        # Combine time and space kernel values
        ts_vals = time_vals * space_vals

        # Normalize the weights
        weights_t = ts_vals / ts_vals.sum()
        return weights_t.to(self.device)
