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
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from mtf.datatset.data_loader import MultiModalDataset
from mtf.model.model import MultiModalClassifier
from train_systems_part.part import TrainSystemPart

class Train():
    def __init__(self, config):
        self.config = config
        self.data_config = config.get("data",{})
        self.train_config = config.get("train", {})

        self.seed = config.get("seed", 42)
        self.set_seed()

        self.device_config = self.train_config.get("device", "cuda:0")
    
        self.device = self.resolve_device()
        self.train_loader, self.val_loader = self.build_dataloader()
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()

        self.best_val_loss = 10000000
        self.save_dir = Path(self.train_config.get("save_dir", "output/baseline"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

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

    def build_model(self):
        classifier = MultiModalClassifier(self.config)
        model = classifier.to(self.device)
        return model
    
    def build_criterion(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def build_optimizer(self):
        params = self.model.parameters()
        lr = float(self.train_config["lr"])
        weight_decay = float(self.train_config["weight_decay"])
        optimizer = torch.optim.AdamW(
            params=params,
            lr=lr,
            weight_decay=weight_decay)
        return optimizer

    def train_one_batch(self, batch):

        signal = batch["signal"].to(self.device)  # lowdim
        image = batch["image"].to(self.device)    # image
        logits = self.model(signal, image)
        label = batch["label"].to(self.device)  
        loss = self.criterion(logits, label)  # 计算loss
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()             # 反向传播
        self.optimizer.step()

        pred = logits.argmax(dim=1)
        correct_count = (pred == label).sum().item()
        sample_count = label.size(0)


        return loss.item(), correct_count, sample_count

    
    def train_one_epoch(self,epoch):
        self.model.train()

        total_loss, total_correct, total_samples = 0, 0, 0

        log.info(f"Training.........")
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            batch_loss, batch_correct, batch_samples = self.train_one_batch(batch)

            total_loss += batch_loss * batch_samples
            total_correct += batch_correct
            total_samples += batch_samples

            batch_acc = batch_correct / batch_samples

            progress_bar.set_postfix(loss=batch_loss, acc=batch_acc)
            # log.info(f"traning batch{batch_samples}")
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def train(self):
        epochs = int(self.train_config["num_epochs"])
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            log.info(f"epoch {epoch}/{epochs}, train:loss{train_loss}&acc{train_acc},val:loss{val_loss}&acc{val_acc}")
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc)
                log.info(f"save best check point for so far:acc{val_acc};loss:{val_loss}")
        self.plot_history()


    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        log.info(f"Validating.........")

        with torch.no_grad():  # 关闭梯度计算
            for batch in self.val_loader:
                signal = batch["signal"].to(self.device)  # lowdim
                image = batch["image"].to(self.device)    # image
                logits = self.model(signal, image)
                label = batch["label"].to(self.device)  
                loss = self.criterion(logits, label)  # 计算loss

                pred = logits.argmax(dim=1)
                batch_samples = label.size(0)
                batch_correct = (pred == label).sum().item()

                total_loss += loss.item() * batch_samples
                total_correct += batch_correct
                total_samples += batch_samples

        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
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
        plt.grid(True)
        plt.show()
        plt.savefig(self.save_dir / "loss_curve.png")
        plt.close()


if __name__ == "__main__":
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_sys = TrainSystemPart()
    start_time = time.perf_counter()
    args = train_sys.parse_args()
    config = train_sys.load_config(args.config)
    trainer = Train(config)
    # print(f"{len(trainer.train_loader.dataset)},{len(trainer.val_loader.dataset)}")

    trainer.train()
    # print(loss_value)
    train_time = time.perf_counter() - start_time
    train_sys.print_train_summary(args, trainer, train_time)



