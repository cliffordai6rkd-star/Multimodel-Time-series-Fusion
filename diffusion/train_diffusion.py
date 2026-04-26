import os, time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from train_systems_part.part import TrainSystemPart
from diffusion.model.noise_scheduler import NoiseScheduler
from diffusion.model.denoise_model import Denoiser
from train_systems_part.base_train import BaseTrain

class TrainDiffusion(BaseTrain):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.noise_scheduler = NoiseScheduler(config, device=self.device)
        self.denoise_model = Denoiser(config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.denoise_model.parameters(), lr = float(self.train_config.get("lr", 1e-3)))
        self.best_loss = float("inf")
        self.history = {"diffusion_loss": []}




    def train_one_batch(self, batch):
        x0 = batch["signal"].to(self.device)  # lowdim
        # x0 = torch.randn(4, 2, 1024)
        batch_size = x0.shape[0]
        noise = torch.randn_like(x0)
        step_t = torch.randint(0, self.noise_scheduler.num_steps, (batch_size,), device = self.device)
        label = batch["label"].to(self.device).long()
        xt = self.noise_scheduler.add_noise(x0, noise, step_t)
        pred_noise = self.denoise_model(xt, step_t, label)
        loss = self.criterion(pred_noise, noise)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_one_epoch(self, epoch):
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        total_loss = 0
        batch_count = 0

        for batch in progress_bar:
            batch_loss = self.train_one_batch(batch)
            total_loss += batch_loss
            batch_count += 1
            progress_bar.set_postfix(loss=batch_loss)

        avg_loss = total_loss / batch_count
        self.history["diffusion_loss"].append(avg_loss)

        return avg_loss
    
    def train(self):
        epochs = int(self.train_config["num_epochs"])        
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            print(f"diffuse loss{avg_loss}")
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_diffusion_checkpoint(epoch, avg_loss)
            self.plot_history()


    def save_diffusion_checkpoint(self, epoch, loss):
        monitor_keys = {
            "loss": loss
        }
        save_ckpt_path = os.path.join(self.save_dir,"diffusion_best.ckpt")      
        model_state = self.denoise_model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        checkpoint = {
            "epoch": epoch,
            "cfg": self.config,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "monitor_keys": monitor_keys
        }
        torch.save(checkpoint, save_ckpt_path)

    def plot_history(self):
        
        y = self.history["diffusion_loss"]
        x = range(1, len(y) + 1)
        plt.figure()
        plt.plot(x, y, label="avg_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.show()
        plt.legend()
        plt.savefig(self.save_dir / "diffusion_loss_curve.png")
        plt.close()


    



if __name__ == "__main__":
    train_sys = TrainSystemPart()
    start_time = time.perf_counter()
    args = train_sys.parse_args()
    config = train_sys.load_config(args.config)
    trainer = TrainDiffusion(config)
    trainer.train()
    train_time = time.perf_counter() - start_time
    train_sys.print_train_summary(args, trainer, train_time)