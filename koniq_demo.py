from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import gc
from scipy.stats import spearmanr
import scipy
import onnxruntime as ort

print("koniq")

device="cuda"

def SRCC(tensor1, tensor2):
    tensor1_np = tensor1.cpu().detach().numpy()
    tensor2_np = tensor2.cpu().detach().numpy()

    rank1 = scipy.stats.rankdata(tensor1_np)
    rank2 = scipy.stats.rankdata(tensor2_np)

    srcc, _ = spearmanr(rank1, rank2)

    return srcc

def PLCC(tensor1, tensor2):
    x_mean = tensor1.mean()
    y_mean = tensor2.mean()

    numerator = ((tensor1 - x_mean) * (tensor2 - y_mean)).sum()

    x_var = ((tensor1 - x_mean) ** 2).sum()
    y_var = ((tensor2 - y_mean) ** 2).sum()

    plcc = numerator / torch.sqrt(x_var * y_var)

    return plcc

class KoniqDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image= self.transform(image)

        image_score = torch.tensor(self.image_frame.iloc[idx, 7])/100.0

        return image,image_score.float()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("trained_models/student_model.onnx", providers=providers)

batch_size = 1

csv_dir = 'koniq/koniq10k_distributions_sets.csv'
img_dir = 'koniq/1024x768'
koniq_dataset = KoniqDataset(csv_file=csv_dir, root_dir=img_dir,transform=transform)

test_indices = torch.load('trained_models/koniq_test_indices.pth')

test_dataset = torch.utils.data.Subset(koniq_dataset, test_indices)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print("start_test")
pos = 0
import numpy as np

for jpeg_images, gt_scores in test_dataloader:
    pos = pos + 1
    if ((pos % 50) == 0):
        print('batch:' + str(pos) + '/' + str(len(test_dataloader)))
    with torch.no_grad():
        jpeg_images = jpeg_images.numpy().astype(np.float32)

        bsz = jpeg_images.shape[0]

        gt_scores = gt_scores.view(bsz, 1).float().to(device)

        # 运行推理
        s_score,_ = ort_session.run(None, {'input': jpeg_images})
        s_score=torch.tensor(s_score)

        if (pos == 1):
            s_all_scores = s_score[0]
            all_gt_scores = gt_scores
        else:
            s_all_scores = torch.cat([s_all_scores, s_score[0]], dim=0)
            all_gt_scores = torch.cat([all_gt_scores, gt_scores], dim=0)

        del jpeg_images, s_score,gt_scores
        gc.collect()
        torch.cuda.empty_cache()


s_plcc=PLCC(s_all_scores,all_gt_scores)
s_srcc = SRCC(s_all_scores,all_gt_scores)

print("student test plcc:" + str(s_plcc) + "  srocc:" + str(s_srcc))


