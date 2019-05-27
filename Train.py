import numpy as np
import torch
import argparse
import torchvision
from torchvision import datasets, transforms
from utils.sampler import RandomIdentitySampler
from utils.resnet import resnet50
from utils.model import ft_net

parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=60)
parser.add_argument('--lr_decay_epochs', type=int, default=40)
parser.add_argument('--model_save_dir', type=str)
parser.add_argument('--img_h', type=int, default=256)
parser.add_argument('--img_w', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

image_dir = args.dataset_dir

data_transform = transforms.Compose([
    transforms.Resize((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

image_datasets['train'] = datasets.ImageFolder(os.path.join(image_dir), data_transform)

dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.batch_size,
                                            sampler=RandomIdentitySampler(image_datasets['train'].imgs),
                                            num_workers=8)
dataset_sizes = len(image_datasets['train'])

inputs, classes = next(iter(dataloaders))