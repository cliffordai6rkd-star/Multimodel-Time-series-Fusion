import argparse 
import os, yaml, time
import logging as log
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from tqdm import tqdm
from pathlib import Path

from mtf.datatset.data_loader import MultiModalDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class BaseTrain:
    def __init__(self, config):
        self.config = config
        self.data_config = config.get("data",{})
        self.train_config = config.get("train", {})

        self.seed = config.get("seed", 42)
        self.set_seed()

        self.device_config = self.train_config.get("device", "cuda:0")
    
        self.device = self.resolve_device()
        self.best_val_loss = float("inf")
        self.save_dir = Path(self.train_config.get("save_dir", "output/baseline"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader, self.val_loader = self.build_dataloader()
        
        self.history = {
              "train_loss": [],
              "train_acc": [],
              "val_loss": [],
              "val_acc": []
        
        }


    def resolve_device(self):
      device = self.device_config
      if device == "auto":
        if torch.cuda.is_available():
              return torch.device("cuda")
        else:
              return torch.device("cpu")
      else:  
          return torch.device(device)
        
    def set_seed(self):     
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        
    def build_dataloader(self):
        dataset = MultiModalDataset(self.data_config)

        val_ratio = self.data_config["val_ratio"]
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        # 确保每次运行训练对数据集的切分一样
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size = int(self.train_config["batch_size"]),
            shuffle = True,
            num_workers = int(self.data_config["num_workers"])
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = int(self.train_config["batch_size"]),
            shuffle = False,
            num_workers = int(self.data_config["num_workers"])
        )
        return train_loader, val_loader

    def save_checkpoint(self, epoch, train_loss, train_acc, val_loss,val_acc):
        monitor_keys = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        save_ckpt_path = os.path.join(self.save_dir,"best.ckpt")      
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        check_point = {
            "epoch": epoch,
            "cfg": self.config,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "monitor_keys": monitor_keys
        }
        torch.save(check_point, save_ckpt_path)

    def plot_history(self):
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]
        if train_loss is None or len(train_loss) == 0:
           return
        epochs = range(1, len(train_loss) + 1)
        # x = epochs
        # y = train_loss
        # label = "train_loss"
        plt.figure()
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_dir / "loss_curve.png")
        plt.close()
