from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import gc
from scipy.stats import spearmanr,pearsonr
import scipy
import scipy.io as sio
import onnxruntime as ort

print("koniq_zeroshot")

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

class CLIVEDataset(Dataset):
    def __init__(self, mat_file, root_dir,score_file, transform=None):
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

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession("trained_models/student_model.onnx", providers=providers)

batch_size = 1

clive_mat_dir = 'clive/ChallengeDB_release/Data/AllImages_release.mat'
clive_score_file = 'clive/ChallengeDB_release/Data/AllMOS_release.mat'
clive_img_dir = 'clive/ChallengeDB_release/Images'
clive_dataset = CLIVEDataset(mat_file=clive_mat_dir, root_dir=clive_img_dir, score_file=clive_score_file,
                                     transform=transform)

clive_dataloader = DataLoader(clive_dataset, batch_size=batch_size, shuffle=True)

print("start_test")
pos = 0
import numpy as np

for jpeg_images, gt_scores in clive_dataloader:
    pos = pos + 1
    if ((pos % 50) == 0):
        print('batch:' + str(pos) + '/' + str(len(clive_dataloader)))
    with torch.no_grad():
        jpeg_images = jpeg_images.numpy().astype(np.float32)

        bsz = jpeg_images.shape[0]

        gt_scores = gt_scores.view(bsz, 1).float().to(device)

        s_score,_ = ort_session.run(None, {'input': jpeg_images})
        s_score=torch.tensor(s_score)

        if (pos == 1):
            s_all_scores = s_score
            all_gt_scores = gt_scores
        else:
            s_all_scores = torch.cat([s_all_scores, s_score], dim=0)
            all_gt_scores = torch.cat([all_gt_scores, gt_scores], dim=0)

        del jpeg_images, s_score,gt_scores
        gc.collect()
        torch.cuda.empty_cache()

s_all_scores=s_all_scores.to(device)
all_gt_scores=all_gt_scores.to(device)
s_plcc=PLCC(s_all_scores,all_gt_scores)
s_srcc = SRCC(s_all_scores,all_gt_scores)

print("student test plcc:" + str(s_plcc.item()) + "  srocc:" + str(s_srcc))


