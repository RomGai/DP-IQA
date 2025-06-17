from scipy.stats import spearmanr
import scipy
import torch

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