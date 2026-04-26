import argparse 
import os, yaml, time
import logging as log
import torch 
import pandas as pd


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
                mean = (x - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean
        return x
    
    def save_samples(self):
        samples = self.sampler()
        samples = samples.detach().cpu()
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples, start=1):
            signal = sample.T
            signal = signal.numpy()
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
