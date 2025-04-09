import math
import torch

import platform
if platform.system() == 'Darwin':
    device = torch.device("mps")
elif platform.system() == 'Linux':
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def uniform(z):
    z = torch.tensor(z).to(device)
    return torch.where(torch.abs(z) <= 1., torch.tensor(1.0, device=z.device), torch.tensor(0.0, device=z.device))

def rectangle(z):
    z = torch.tensor(z).to(device)
    return torch.where(torch.abs(z) <= 1., torch.tensor(0.5, device=z.device), torch.tensor(0.0, device=z.device))

def triangle(z):
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    return torch.where(torch.abs(z) <= 1., 1 - torch.abs(z), torch.zeros_like(z))

def epanechnikov(z):
    z = torch.tensor(z, device=device)
    abs_z = torch.abs(z)
    return torch.where(abs_z <= 1, 3 / 4 * (1 - z**2), torch.zeros_like(abs_z))

def biweight(z):
    abs_z = torch.abs(z)
    return torch.where(abs_z <= 1, 15 / 16 * (1 - z**2)**2, torch.zeros_like(z))

def tricube(z):
    abs_z = torch.abs(z)
    return torch.where(abs_z <= 1, (1 - abs_z ** 3) ** 3, torch.zeros_like(abs_z))

def gaussian(z):
    abs_z = torch.abs(z)
    return 1./torch.sqrt(2 * torch.as_tensor(torch.pi)) * torch.exp(-abs_z ** 2 / 2)

def silverman(z):
    abs_z = torch.abs(z)
    return 1/2 * torch.exp(-abs_z / math.sqrt(2)) * torch.sin(abs_z / math.sqrt(2) + torch.pi / 4)


class ECDFTorch(torch.nn.Module):
    def __init__(self, x, weights=None, side='right'):
        super(ECDFTorch, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        if len(x.shape) != 1:
            msg = 'x must be 1-dimensional'
            raise ValueError(msg)

        nobs = len(x)
        if weights is not None:
            assert len(weights) == nobs
            sweights = torch.sum(weights)
            assert sweights > 0.
            sorted = torch.argsort(x).int()
            x = x[sorted]
            y = torch.cumsum(weights[sorted], dim=0)
            y = y / sweights
            self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
            self.y = torch.cat((torch.tensor([0], device=y.device), y))
            self.n = self.x.shape[0]

        else:
            x = torch.sort(x)[0]
            y = torch.linspace(1. / nobs, 1, nobs, device=x.device)
            self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
            self.y = torch.cat((torch.tensor([0], device=y.device), y))
            self.n = self.x.shape[0]

    def forward(self, time):
        tind = torch.searchsorted(self.x, time, side=self.side) - 1
        return self.y[tind].to(device)

def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1), 1 / p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by torch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")