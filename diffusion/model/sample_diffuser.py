import argparse 
import os, yaml, time
import logging as log
import torch 
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from train_systems_part.part import TrainSystemPart
from diffusion.model.denoise_model import Denoiser
from diffusion.model.noise_scheduler import NoiseScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="how to inefrence")
    parser.add_argument("-c", "--config", 
                       type=str, 
                       default="diffusion/config/test.yaml", 
                       help="Path to the config file")
    
    parser.add_argument("--ckpt", 
                       type=str, 
                       default="data/output/diffusion/diffusion_best.ckpt", 
                       help="Path to the best ckpt")
    parser.add_argument("-l", "--label", 
                       type=str, 
                       default="cloth", 
                       help="generate what kind of data")
    
    parser.add_argument("-n", "--num",
                       type=int, 
                       default=170, 
                       help="generate how many")
    
    parser.add_argument("-o", "--output-dir", 
                       type=str, 
                       default="data/train_data2_30ep/cloth", 
                       help="generate where")
        


    args = parser.parse_args()
    return args

class DiffusionInferencer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.train_config = config.get("train", {})
        self.data_config = config.get("data",{})
        self.diffusion_config = config.get("diffusion",{})
        self.device_config = self.train_config.get("device", "auto")
        self.inference_mode = "ddim"
        if self.device_config == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_config)

    def load_checkpoint(self):
        ckpt_path = Path(self.args.ckpt)
        if os.path.exists(ckpt_path):
            log.info(f"Loading ckpt path {ckpt_path}")
        else:
            raise FileNotFoundError
        ckpt = torch.load(ckpt_path, map_location=self.device)
        log.info(f"Load ckpt:{ckpt_path} completed")
        return ckpt
    
    def build_denoiser(self, ckpt):
        ckpt_cfg = ckpt["cfg"]
        denoise_model = Denoiser(ckpt_cfg)
        denoise_model.load_state_dict(ckpt["model_state"])
        denoise_model.to(self.device)
        denoise_model.eval()
        return denoise_model
    
    def build_scheduler(self, ckpt):
        ckpt_cfg = ckpt["cfg"]
        scheduler = NoiseScheduler(ckpt_cfg, self.device)
        return scheduler
    
    def compute_class_mean_std(self, class_name):
        data_dir = Path(self.data_config["data_dir"])
        class_dir = data_dir / class_name

        if not os.path.exists(class_dir):
            raise FileNotFoundError
        files = list(class_dir.glob("*.xlsx"))
        if files is None:
            raise FileNotFoundError
        sample_means = []
        sample_stds = []
        for file_path in files:
            df = pd.read_excel(file_path)
            high_col = f"{class_name}_high"
            low_col = f"{class_name}_low"
            signal = df[[high_col, low_col]]
            mean = signal.mean(axis=0)
            std = signal.std(axis=0)
            sample_means.append(mean.to_numpy())
            sample_stds.append(std.to_numpy())
        sample_means = np.stack(sample_means)
        sample_stds = np.stack(sample_stds)
        class_mean = sample_means.mean(axis=0)
        class_std = sample_stds.mean(axis=0)
        return class_mean, class_std

    def sampler(self):
        ckpt = self.load_checkpoint()
        num = self.args.num
        sequence_length = self.data_config.get("sequence_length", 1024)
        device = self.device
        x = torch.randn(num, 2, sequence_length).to(device)
        class_names = self.data_config["class_names"]
        label_id = class_names.index(self.args.label)
        labels = torch.full((num,), label_id).long().to(device)

        denoiser = self.build_denoiser(ckpt)
        scheduler = self.build_scheduler(ckpt)
        num_steps = scheduler.num_steps
        denoiser.eval()
        with torch.no_grad():
            
            for t in range(num_steps-1, -1, -1):
                step_t = torch.full((num,), t, dtype=torch.long, device=self.device)
                pred_noise = denoiser(x, step_t, labels)
                beta_t = scheduler.betas[t].view(1, 1, 1)
                alpha_t = scheduler.alphas[t].view(1, 1, 1)
                alpha_bar_t = scheduler.alphas_bars[t].view(1, 1, 1)
                if self.inference_mode == "ddpm":
                    mean = (x - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
                    if t > 0:
                        noise = torch.randn_like(x)
                        x = mean + torch.sqrt(beta_t) * noise
                    else:
                        x = mean
                if self.inference_mode == "ddim":
                    prev_t = t - 1

                    if prev_t >= 0:
                        alpha_bar_prev = scheduler.alphas_bars[prev_t].view(1, 1, 1)
                    else:
                        alpha_bar_prev = torch.ones_like(alpha_bar_t)
                    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                    x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * pred_noise
        return x
    
    def save_samples(self):
        samples = self.sampler()
        samples = samples.detach().cpu()
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        class_mean, class_std = self.compute_class_mean_std(self.args.label)

        for index, sample in enumerate(samples, start=1):
            signal = sample.T
            signal = signal.numpy()
            signal = signal * class_std + class_mean

            # print(signal.shape)
            class_name = self.args.label
            
            df = pd.DataFrame({
                f"{class_name}_high": signal[:, 0],
                f"{class_name}_low": signal[:, 1],
            })

            output_path = output_dir / f"{class_name}_diffusion_{index:03d}.xlsx"
            df.to_excel(output_path, index=False)
            log.info(f"Inferencing....")
            log.info(f"data saved: {output_path}")
        self.compute_class_mean_std(self.args.label)


if __name__ == "__main__":
    log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
    train_sys = TrainSystemPart()
    # args = train_sys.parse_args()

    args = parse_args()
    config = train_sys.load_config(args.config)
    diff_inferencer = DiffusionInferencer(config, args)

    diff_inferencer.save_samples()
    # print(x.shape)
    # print(labels)
