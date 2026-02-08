
import math
import torch
from torch import nn
from torch.nn import functional as F



def ste_sign(x: torch.Tensor, threshold: float = 0) -> torch.Tensor:
    """Straight-Through Estimator for sign function - forward uses sign, backward uses identity"""
    return x + (torch.sign(x - threshold) - x).detach()

def ste_clip(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Straight-Through Estimator for clamp - forward uses clamp, backward uses identity"""
    return x + (torch.clamp(x, min_val, max_val) - x).detach()

class BitLinear(nn.Module):
    """
    BitLinear layer with 1.58-bit quantization.
    Implements weight binarization and activation quantization with proper scaling.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            act_bits: int = 4,
            eps: float = 1e-5
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters similar to nn.Linear"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight binarization and activation quantization.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Weight binarization: center weights and binarize to {-1, +1}
        w_centered = self.weight - self.weight.mean(dim=1, keepdim=True)
        w_bin = ste_sign(w_centered)

        # Compute weight scale factor (beta): shape (1, out_features) for broadcasting
        beta = self.weight.abs().mean(dim=1, keepdim=True).t()

        # Normalize activations using LayerNorm
        xln = F.layer_norm(x, (x.shape[-1],), eps=self.eps)

        # Activation quantization: compute quantization range and scale factor
        qb = 2 ** (self.act_bits - 1) - 1  # Quantization range: e.g., 127 for 8-bit
        gama = xln.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)  # Per-sample gamma

        # Quantize activations to integer range [-qb, qb-1]
        xq = xln * (qb / gama)
        xq = ste_clip(xq, -qb + self.eps, qb - self.eps)

        # Binary matmul (without bias - we'll add it after rescaling)
        y = F.linear(xq, w_bin, None)

        # Rescale output: y = y / (wscale * xscale) = y * beta * gama / qb
        y = y * (beta * gama / qb)

        # Add bias after rescaling (if present)
        if self.bias is not None:
            y = y + self.bias

        return y

    def __repr__(self) -> str:
        """String representation showing layer configuration"""
        return (f'{self.__class__.__name__}('
                f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'act_bits={self.act_bits})')




