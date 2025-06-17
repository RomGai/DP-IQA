from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
import re
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import sys

# KonIQ
koniq_csv_file = "koniq/koniq10k_distributions_sets.csv"
koniq_image_dir = "koniq/1024x768"
# CLIVE
clive_mat_dir = 'ChallengeDB_release/Data/AllImages_release.mat'
clive_score_file = 'ChallengeDB_release/Data/AllMOS_release.mat'
clive_img_dir = 'ChallengeDB_release/Images'
# LIVEFB
livefb_csv_file = "livefb_database/labels_image.csv"
livefb_img_dir = "livefb_database"
# SPAQ
spaq_csv_file = "spaq/MOS and Image attribute scores.xlsx"
spaq_img_dir = "spaq/SPAQ/TestImage"

transform_in_the_wild = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

transform_synthetic_train = transforms.Compose(
                [
                    transforms.Resize((1024,768)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((512,512)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ])

transform_synthetic_test = transforms.Compose(
                [
                    transforms.Resize((1024,768)),
                    transforms.RandomCrop((512,512)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ])

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename

class KonIQDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=transform_in_the_wild):
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image_score = torch.tensor(self.image_frame.iloc[idx, 7])/100.0

        if self.transform:
            image= self.transform(image)

        return image, image_score.float()

class CLIVEDataset(Dataset):
    def __init__(self, mat_file, root_dir,score_file, transform=transform_in_the_wild):
        self.image_name = sio.loadmat(mat_file)["AllImages_release"]
        self.root_dir = root_dir
        self.scores=sio.loadmat(score_file)["AllMOS_release"][0]
        self.transform = transform

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.image_name[idx][0][0]))
        image = Image.open(img_name).convert('RGB')

        image_score = torch.tensor(self.scores[idx])/100.0

        if self.transform:
            image= self.transform(image)

        return image,image_score.float()

class LIVEFBDataset(Dataset):
    def __init__(self, csv_file, root_dir,transform=transform_in_the_wild):
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image_score = torch.tensor(self.image_frame.iloc[idx, 1])/100.0

        if self.transform:
            image= self.transform(image)

        return image,image_score.float()

class SPAQDataset(Dataset):
    def __init__(self, csv_file, root_dir,transform=transform_in_the_wild):
        self.image_frame = pd.read_excel(csv_file, header=None,skiprows=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        image_score = torch.tensor(self.image_frame.iloc[idx, 1])/100.0

        if self.transform:
            image= self.transform(image)

        return image,image_score.float()

def load_dataset(name,save_indices,load_indices=False,
                 train_indices_path=None,test_indices_path=None):
    if(name=="koniq"):
        koniq_dataset = KonIQDataset(csv_file=koniq_csv_file, root_dir=koniq_image_dir, transform=transform_in_the_wild)
        if(load_indices==False):
            koniq_dataset_size = len(koniq_dataset)
            koniq_train_size = int(koniq_dataset_size * 0.8)
            koniq_test_size = koniq_dataset_size - koniq_train_size
            train_dataset, test_dataset = random_split(koniq_dataset, [koniq_train_size, koniq_test_size])
            if(save_indices==True):
                train_indices = train_dataset.indices
                test_indices = test_dataset.indices
                torch.save(train_indices, 'koniq_train_indices.pth')
                torch.save(test_indices, 'koniq_test_indices.pth')
        elif(load_indices==True):
            train_indices = torch.load(train_indices_path)
            test_indices = torch.load(test_indices_path)
            train_dataset = torch.utils.data.Subset(koniq_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(koniq_dataset, test_indices)

        return train_dataset, test_dataset
    elif(name=="clive"):
        clive_dataset = CLIVEDataset(mat_file=clive_mat_dir, root_dir=clive_img_dir, score_file=clive_score_file,
                                     transform=transform_in_the_wild)
        if (load_indices == False):
            clive_dataset_size = len(clive_dataset)
            clive_train_size = int(clive_dataset_size * 0.8)
            clive_test_size = clive_dataset_size - clive_train_size
            train_dataset, test_dataset = random_split(clive_dataset, [clive_train_size, clive_test_size])
            if(save_indices==True):
                train_indices = train_dataset.indices
                test_indices = test_dataset.indices
                torch.save(train_indices, 'clive_train_indices.pth')
                torch.save(test_indices, 'clive_test_indices.pth')
        elif(load_indices==True):
            train_indices = torch.load(train_indices_path)
            test_indices = torch.load(test_indices_path)
            train_dataset = torch.utils.data.Subset(clive_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(clive_dataset, test_indices)
        return train_dataset, test_dataset
    elif(name=="livefb"):
        livefb_dataset = LIVEFBDataset(csv_file=livefb_csv_file, root_dir=livefb_img_dir,
                                       transform=transform_in_the_wild)
        if (load_indices == False):
            livefb_dataset_size = len(livefb_dataset)
            livefb_train_size = int(livefb_dataset_size * 0.8)
            livefb_test_size = livefb_dataset_size - livefb_train_size
            train_dataset, test_dataset = random_split(livefb_dataset, [livefb_train_size, livefb_test_size])
            if(save_indices==True):
                train_indices = train_dataset.indices
                test_indices = test_dataset.indices
                torch.save(train_indices, 'livefb_train_indices.pth')
                torch.save(test_indices, 'livefb_test_indices.pth')
        elif(load_indices==True):
            train_indices = torch.load(train_indices_path)
            test_indices = torch.load(test_indices_path)
            train_dataset = torch.utils.data.Subset(livefb_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(livefb_dataset, test_indices)
        return train_dataset,test_dataset
    elif(name=="spaq"):
        spaq_dataset = SPAQDataset(csv_file=spaq_csv_file, root_dir=spaq_img_dir, transform=transform_in_the_wild)
        if (load_indices == False):
            spaq_dataset_size = len(spaq_dataset)
            spaq_train_size = int(spaq_dataset_size * 0.8)
            spaq_test_size = spaq_dataset_size - spaq_train_size
            train_dataset, test_dataset = random_split(spaq_dataset, [spaq_train_size, spaq_test_size])
            if(save_indices==True):
                train_indices = train_dataset.indices
                test_indices = test_dataset.indices
                torch.save(train_indices, 'spaq_train_indices.pth')
                torch.save(test_indices, 'spaq_test_indices.pth')
        elif(load_indices==True):
            train_indices = torch.load(train_indices_path)
            test_indices = torch.load(test_indices_path)
            train_dataset = torch.utils.data.Subset(spaq_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(spaq_dataset, test_indices)
        return train_dataset,test_dataset
    else:
        print("wrong dataset type")
        sys.exit(1)


