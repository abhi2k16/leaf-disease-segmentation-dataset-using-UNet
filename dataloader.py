# %%
"""# Download latest version
import kagglehub
path = kagglehub.dataset_download("fakhrealam9537/leaf-disease-segmentation-dataset")
print("Path to dataset files:", path)"""
#%%
import os
import numpy as np
import random
from PIL import Image
#%%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
#%%
# function for random split 
def split_dataset(data_path, train_ratio=0.8):
    """
    function for random split the image and label folder image into 
    train and validation and put into separate train and validation folder
    """
    import shutil
    import random
    images_path = os.path.join(data_path, "images")
    annotations_path = os.path.join(data_path, "annotations")
    # Create train/val directories
    for split in ["training", "validation"]:
        os.makedirs(os.path.join(data_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(data_path, "annotations", split), exist_ok=True)
    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_files)
    # Split files
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Move files
    for files, split in [(train_files, "training"), (val_files, "validation")]:
        for img_file in files:
            # Move image
            shutil.move(os.path.join(images_path, img_file), 
                       os.path.join(data_path, "images", split, img_file))
            # Move corresponding annotation
            ann_file = img_file.replace('.jpg', '.png').replace('.png', '.png')
            if os.path.exists(os.path.join(annotations_path, ann_file)):
                shutil.move(os.path.join(annotations_path, ann_file),
                           os.path.join(data_path, "annotations", split, ann_file))

# Dataset Class
class Dataset(Dataset):
    def __init__(self, 
                 data_path,
                 train = True, 
                 image_size = 128,
                 random_crop_ratio = (0.2, 1),
                 interfernce_mode = False):
        self.data_path = data_path
        self.train = train
        self.image_size = image_size
        self.interference_mode = interfernce_mode
        self.min_ratio, self.max_ratio = random_crop_ratio
        if train:
            split = "training"
        else:
            split = "validation"
        self.path_to_images = os.path.join(data_path, "images", split)
        self.path_to_annotations = os.path.join(data_path, "annotations", split)
        self.file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]
    
        print(f"number of images: {len(self.file_roots)}")
        print(f"first image name: {self.file_roots[0]}")
        # Resize and normalize transforms
        self.resize = transforms.Resize((image_size, image_size))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.random_crop = transforms.RandomResizedCrop(size = (image_size, image_size), scale=(self.min_ratio, self.max_ratio))
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        #self.vertical_flip = transforms.RandomVerticalFlip(p=0.5)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)
    
    def __getitem__(self, idx):
        image_name = self.file_roots[idx]
        image_path = os.path.join(self.path_to_images, image_name + ".jpg")
        annotation_path = os.path.join(self.path_to_annotations, image_name + ".png")
        image = Image.open(image_path).convert("RGB")
        annotation = Image.open(annotation_path).convert("L")



        if self.train and (not self.interference_mode):
            if random.random() < 0.5:
                image = self.resize(image)
                annotation = self.resize(annotation)
            else:
                min_size = min(image.size)
                random_ratio = random.uniform(self.min_ratio, self.max_ratio)

                corp_size = int(min_size * random_ratio)
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(corp_size, corp_size))
                image = F.crop(image, i, j, h, w)
                annotation = F.crop(annotation, i, j, h, w)
                image = self.resize(image)
                annotation = self.resize(annotation)
            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                annotation = self.horizontal_flip(annotation)
        else:
            image = self.resize(image)
            annotation = self.resize(annotation)
        
        image = self.to_tensor(image)
        annotation = torch.tensor(np.array(annotation)).long()
        image = self.normalize(image)

        return image, annotation

if __name__ == "__main__":
    data_path = r"C:\Users\abhij\Desktop\CompVisionImageProc\CV_DLModels\UNetModel\leaf-disease-segmentation-dataset\data"
    
    # Check if training/validation folders exist in images and annotation folder
    train_img_path = os.path.join(data_path, "images", "training")
    val_img_path = os.path.join(data_path, "images", "validation")
    train_ann_path = os.path.join(data_path, "annotations", "training")
    val_ann_path = os.path.join(data_path, "annotations", "validation")
    
    if all(os.path.exists(p) for p in [train_img_path, val_img_path, train_ann_path, val_ann_path]):
        print("Training and validation folders exist in both images and annotations")
    else:
        print("Splitting dataset...")
        split_dataset(data_path)
 
    dataset = Dataset(data_path)
    for sample in dataset:
        print(sample[0])
        print(sample[1])
        break
  