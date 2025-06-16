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

    def __init__(self,
                 ckp_path,
                 device,
                 num_steps=18,
                 sigma_min=0.002,
                 sigma_max=80,
                 rho=7):
        self.get_default_config()
        self.noise_schedule = NoiseScheduleEDM()

        self.num_steps=num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        t_0 = sigma_min
        t_T = sigma_max
        self.lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device)).detach().cpu().numpy()
        self.lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device)).detach().cpu().numpy()
        self.ckp_path = ckp_path

    def get_timesteps(self, device):
        """Constructs the noise schedule of Karras et al. (2022)."""
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=device)

        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        return step_indices, t_steps

    def load_model(self, device):
        # Load network.
        print(f'Loading network from "{self.ckp_path}"...')
        with open(self.ckp_path, "rb") as f:
            self.net = pickle.load(f)["ema"].to(device)

    def model_fn(self, x, t, cond=None):
        return self.net(x, t, cond)
