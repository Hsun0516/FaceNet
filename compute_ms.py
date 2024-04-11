import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for imgs, _ in tqdm(loader):
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std

pipeline = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])
dataset = ImageFolder('Data/', transform=pipeline)
print(compute_mean_std(dataset))