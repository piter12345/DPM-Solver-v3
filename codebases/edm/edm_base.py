import ml_collections
import torch
import torch.nn.functional as F
import math
import numpy as np
import os
import pickle

        

class NoiseScheduleEDM:
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.zeros_like(t).to(torch.float64)

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.ones_like(t).to(torch.float64)

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t.to(torch.float64)

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """

        return -torch.log(t).to(torch.float64)

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return torch.exp(-lamb).to(torch.float64)

class EDM:
    def get_default_config(self):
        self.data = data = ml_collections.ConfigDict()
        data.dataset = "CIFAR10"
        data.image_size = 32
        data.random_flip = True
        data.centered = False
        data.uniform_dequantization = False
        data.num_channels = 3

    def __init__(self, ckp_path):
        self.get_default_config()
        self.noise_schedule = NoiseScheduleEDM()
        self.ckp_path = ckp_path

    def get_timesteps(self, N, device):
        """Constructs the noise schedule of Karras et al. (2022)."""

        rho = 7.0  # 7.0 is the value used in the paper

        sigma_min: float = np.exp(-self.lambda_0)
        sigma_max: float = np.exp(-self.lambda_T)
        ramp = np.linspace(0, 1, N + 1)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        lambdas = torch.Tensor(-np.log(sigmas)).to(device)
        timesteps = self.noise_schedule.inverse_lambda(lambdas)

        indexes = list(
            (self.statistics_steps * (lambdas - self.lambda_T) / (self.lambda_0 - self.lambda_T))
            .round()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        return indexes, timesteps

    def create_model(self, device):
        # Load network.
        print(f'Loading network from "{self.ckp_path}"...')
        with open(self.ckp_path, "rb") as f:
            self.net = pickle.load(f)["ema"].to(device)

    def model_fn(self, x, t, cond=None):
        return self.net(x, t, cond)
