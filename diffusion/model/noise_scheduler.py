import os, yaml
import torch
from train_systems_part.part import TrainSystemPart


class NoiseScheduler:
    def __init__(self, config, device):
        self.config = config
        self.diffusion_config = config.get("diffusion",{})
        self.device = device
        self.num_steps = self.diffusion_config.get("num_steps", 100)
        self.beta_start = float(self.diffusion_config.get("beta_start", 1e-4))
        self.beta_end = float(self.diffusion_config.get("beta_end", 0.02))
        self.noise_param()

    def noise_param(self):
        self.alphas = 0
        self.alphas_bars = 1
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=self.device)

        self.alphas = 1- self.betas
        self.alphas_bars = torch.cumprod(self.alphas, dim=0)
        return self.betas, self.alphas, self.alphas_bars

    def add_noise(self, x0, noise, t):
        alpha_bars_t = self.alphas_bars[t]
        alpha_bars_t = alpha_bars_t.view(-1, 1, 1)
        xt = torch.sqrt(alpha_bars_t) * x0 + torch.sqrt(1 - alpha_bars_t) * noise
        return xt




if __name__ == "__main__":
    train_sys = TrainSystemPart()
    args = train_sys.parse_args()
    config = train_sys.load_config(args.config)
    noise_scheduler = NoiseScheduler(config)
    x0 = torch.randn(4, 2, 1024)
    noise = torch.randn_like(x0) # 随机高斯噪声
    step_t = torch.randint(0,noise_scheduler.num_steps, (4,))
    xt = noise_scheduler.add_noise(x0, noise, step_t)
    print(xt.shape)
