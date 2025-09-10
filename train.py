import os
import random
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from UNetModel import UNet
from dataloader import Dataset


path_to_data = r"..\leaf-disease-segmentation-dataset\data"

def train(batch_size = 4,
          gradient_accumulation_steps = 1,
          num_epochs = 1, 
          learning_rate = 1e-3,
          image_size = 128,
          num_workers = 0,
          experiment_name = "unet_train"):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, cpu=True)

    path_to_experiment = os.path.join("experiments", experiment_name)
    if not os.path.exists(path_to_experiment):
        os.makedirs(path_to_experiment)
    
    micro_batch = batch_size // gradient_accumulation_steps
    train_dataset = Dataset(path_to_data, train=True, image_size=image_size)
    test_dataset = Dataset(path_to_data, train=False, image_size=image_size)

    train_dataloader = DataLoader(train_dataset, batch_size=micro_batch, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=micro_batch, shuffle=False, num_workers=num_workers)

    loss_fn = nn.CrossEntropyLoss() 
    model = UNet(
        in_channels=3,
        num_classes=150,
        start_dim=32,
        dim_mults=[1,2,4],
        residual_block_per_group=1,
        groupnorm_num_group=8,
        interpolate_upsample=True
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
        )

    for epoch in range(num_epochs):
        accelerator.print(f"Epoch: [{epoch+1}/{num_epochs}]")
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []    

        accumulated_loss = 0
        accumulated_acc = 0
        progress_bar = tqdm(range(len(train_dataloader)//gradient_accumulation_steps), 
                            desc="Training", disable=not accelerator.is_main_process)
        
        model.train()

        for images, labels in train_dataloader:
            with accelerator.accumulate(model):

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                accumulated_loss += loss/gradient_accumulation_steps
                predicted = torch.argmax(outputs, dim=1)
                acc = (predicted == labels).sum()/torch.numel(predicted)
                accumulated_acc += acc/gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                    acc_gathered = accelerator.gather_for_metrics(accumulated_acc)

                    train_loss.append(torch.mean(loss_gathered).item())
                    train_acc.append(torch.mean(acc_gathered).item())

                    accumulated_loss = 0
                    accumulated_acc = 0
                    progress_bar.update(1)
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        for images, labels in test_dataloader:
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                accumulated_loss += loss/gradient_accumulation_steps
                predicted = torch.argmax(outputs, dim=1)
                acc = (predicted == labels).sum()/torch.numel(predicted)
                accumulated_acc += acc/gradient_accumulation_steps

                loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                acc_gathered = accelerator.gather_for_metrics(accumulated_acc)

                test_loss.append(torch.mean(loss_gathered).item())
                test_acc.append(torch.mean(acc_gathered).item())
        # average loss and accuracy
        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_acc = np.mean(train_acc)
        test_acc = np.mean(test_acc)
        accelerator.print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        accelerator.print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        accelerator.wait_for_everyone()
        accelerator.save_state(os.path.join(path_to_experiment, f"epoch_{epoch+1}.pt"))
        accelerator.print(f"Saved model checkpoint to {path_to_experiment}")
        accelerator.print(f"Epoch: [{epoch+1}/{num_epochs}]")

if __name__ == "__main__":
    train(num_epochs=2) 
