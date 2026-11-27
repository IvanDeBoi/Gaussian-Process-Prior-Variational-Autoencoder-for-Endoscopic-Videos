"""GPyTorch exact GP used as the prior over time in the GP-VAE models.

Kernel (RBF, Matern or Cauchy) times an output scale, with hard bounds on every hyper-parameter:
lengthscale in [1, 30] frames (or [0.025, 2] when frame numbers are scaled to [0, 1]), output scale in
[0.5, 10] and noise in [1e-3, 1]. Initial values are drawn with numpy's global random generator (3 draws), so
building a GP changes the numpy random state. Unchanged from the original Models/Building_Blocks/GP.py except
that `use_x_scaling` is a parameter instead of an import from config_endo.py.
"""
import numpy as np
import torch
import gpytorch
from gpvae_endo.gp.kernels import CauchyKernel


class GP(gpytorch.models.ExactGP):
    def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood=None,
            device="cpu",
            kernel="RBF",
            verbose=True,
            use_x_scaling: bool = False,
    ):
        train_x = train_x.to(device=device)
        train_y = train_y.to(device=device, dtype=train_x.dtype)

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        super().__init__(train_x, train_y, likelihood)
        self.device = device
        self.mean_module = gpytorch.means.ConstantMean()

        # ---------------------------------------------------------
        # 1. Define Strict Boundaries (Constraints)
        # ---------------------------------------------------------
        # These act as "Hard Walls" for the optimizer.
        # Use normalized-domain constraints when x is scaled to [0, 1],
        # otherwise use frame-domain constraints for raw frame numbers.
        if use_x_scaling:
            ls_lower, ls_upper = 0.025, 2.0
            ls_prior = gpytorch.priors.GammaPrior(10.0, 10.0)  # mean 1 in normalized units
            init_ls_max_offset = 0.5
            ls_mode = "scaled-x"
        else:
            ls_lower, ls_upper = 1.0, 30.0
            ls_prior = gpytorch.priors.GammaPrior(4.0, 0.5)  # mean 8 frames, fairly narrow
            init_ls_max_offset = 7.0
            ls_mode = "raw-frame-x"
        os_lower, os_upper = 0.5, 10.0
        noise_lower, noise_upper = 1e-3, 1.0

        if verbose:
            print(f"GP constraint mode: {ls_mode}")
            print(f"Strict Boundaries: LS [{ls_lower}, {ls_upper}], "
                  f"OS [{os_lower}, {os_upper}], Noise [{noise_lower}, {noise_upper}]")

        # -------------------------------
        # Kernel Selection
        # -------------------------------
        ard_dims = train_x.shape[-1]
        if kernel == "RBF":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel == "MATERN":
            base_kernel = gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims)
        elif kernel == "CAUCHY":
            base_kernel = CauchyKernel(ard_num_dims=ard_dims)
        else:
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        # ---------------------------------------------------------
        # 2. Apply Constraints & Priors
        # ---------------------------------------------------------

        # --- Lengthscale ---
        self.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(ls_lower, ls_upper)
        )
        self.covar_module.base_kernel.register_prior(
            "lengthscale_prior", ls_prior, "lengthscale"
        )

        # --- Outputscale ---
        os_prior = gpytorch.priors.GammaPrior(25.0, 25.0)  # Mean 1, very tight
        self.covar_module.register_constraint(
            "raw_outputscale", gpytorch.constraints.Interval(os_lower, os_upper)
        )
        self.covar_module.register_prior(
            "outputscale_prior", os_prior, "outputscale"
        )

        # --- Noise ---
        noise_prior = gpytorch.priors.LogNormalPrior(loc=np.log(0.05), scale=0.2)
        self.likelihood.noise_covar.register_constraint(
            "raw_noise", gpytorch.constraints.Interval(noise_lower, noise_upper)
        )
        self.likelihood.noise_covar.register_prior(
            "noise_prior", noise_prior, "noise"
        )

        # ---------------------------------------------------------
        # 3. Safe Initialization (Guaranteed inside boundaries)
        # ---------------------------------------------------------
        # We sample within a safe sub-range of your boundaries
        init_dtype = train_x.dtype
        init_ls = torch.tensor(
            ls_lower + (init_ls_max_offset * np.random.rand()),
            device=device,
            dtype=init_dtype,
        )
        init_os = torch.tensor(os_lower + (0.5 * np.random.rand()), device=device, dtype=init_dtype)
        init_noise = torch.tensor(noise_lower + (0.04 * np.random.rand()), device=device, dtype=init_dtype)

        self.initialize(**{
            "covar_module.base_kernel.lengthscale": init_ls,
            "covar_module.outputscale": init_os,
            "likelihood.noise_covar.noise": init_noise,
        })

        self.to(device=device, dtype=init_dtype)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
