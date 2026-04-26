import argparse 
import os, yaml, time
import logging as log
from tqdm import tqdm
from pathlib import Path


class TrainSystemPart:
    def parse_args(self):   
        parser = argparse.ArgumentParser(description="train config")
        parser.add_argument("-c", "--config", 
                            type=str, 
                            default="config/img.yaml", 
                            help="Path to the config file")
    
        args = parser.parse_args()
        return args
    
    def load_config(self, config_path):
        config_path = Path(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
    
        return config
    
    def format_time(self, seconds):
        seconds = int(seconds) 
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def print_train_summary(self, args, trainer, train_time):
        line = "=" * 60
        print(line)
        print("Training Finished")
        print(line)
        print(f"Config Path     : {args.config}")
        print(f"Device          : {trainer.device}")
        print(f"Epochs          : {trainer.train_config['num_epochs']}")
        print(f"Best Val LOSS   : {trainer.best_val_loss:}")
        print(f"Checkpoint Dir  : {trainer.save_dir}")
        print(f"Total Time      : {self.format_time(train_time)}")
        print(line)

