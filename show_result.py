import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from UNetModel import UNet
from dataloader import Dataset
from torch.utils.data import DataLoader

def load_model(checkpoint_dir):
    from accelerate import Accelerator
    model = UNet(
        in_channels=3,
        num_classes=150,
        start_dim=32,
        dim_mults=[1,2,4],
        residual_block_per_group=1,
        groupnorm_num_group=8,
        interpolate_upsample=True
    )
    accelerator = Accelerator(cpu=True)
    accelerator.load_state(checkpoint_dir)
    model.eval()
    return model

def show_results(model, dataset, num_samples=3):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, (image, label) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            output = model(image)
            prediction = torch.argmax(output, dim=1).squeeze().numpy()
            
            # Convert tensors to numpy for display
            img = image.squeeze().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(label.squeeze().numpy(), cmap='tab20')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(prediction, cmap='tab20')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load dataset
    data_path = r"..\leaf-disease-segmentation-dataset\data"
    test_dataset = Dataset(data_path, train=False, image_size=128)
    
    # Load trained model
    checkpoint_dir = r"experiments\unet_train\epoch_2.pt"
    model = load_model(checkpoint_dir)
    
    # Show results
    show_results(model, test_dataset)
