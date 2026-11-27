"""
Custom Cauchy Kernel for GPyTorch - Student-t process kernel (nu=1)
Compatible with GPVAE Jazbec/SPA/Casale models.
"""
import gpytorch
import torch


class CauchyKernel(gpytorch.kernels.Kernel):
    """
    Cauchy kernel: k(x, x') = σ² / (1 + ||x - x'||² / ℓ²)

    - has_lengthscale=True: Uses GPyTorch's automatic raw_lengthscale + Positive constraint
    - Works directly with ScaleKernel(outputscale), GP.initialize(hypers)
    """
    has_lengthscale = True  # Enables standard lengthscale param/constraint

    def forward(self, x1, x2, diag=False, **params):
        # Squared Euclidean distances (broadcast-safe, shape-consistent)
        dist = self.covar_dist(x1, x2, diag=diag, square_dist=True)

        # Cauchy formula: 1 / (1 + d² / ℓ²)
        # self.lengthscale auto-expanded: (..., 1, 1)
        denom = 1.0 + dist / (self.lengthscale ** 2)
        output = 1.0 / denom

        return output

# Usage in your GP models (e.g., GPVAE_Casale_model.py or SVGPVAE_Jazbec_model.py)
# if kernel_gp_prior == "CAUCHY":
#     base_kernel = gpytorch.kernels.ScaleKernel(CauchyKernel())
#     gp_prior = GP(..., base_kernel=base_kernel, ...)
