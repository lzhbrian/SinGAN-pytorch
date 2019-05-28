"""
author: lzhbrian (https://lzhbrian.me)
date: 2019.5.28
"""

import torch
import torchvision
import torchvision.transforms as transforms
import PIL
from PIL import Image


class SinganDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.convert('RGB')
        img = self.transform(img)
        return img


def get_dataloader(scale, batch_size=1, multiple=3):

    img_list = ['resources/test1.png'] * multiple
    transform_list = [
        transforms.Resize((scale, scale), PIL.Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    data_transforms = transforms.Compose(transform_list)
    dataset = SinganDataset(img_list, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader
