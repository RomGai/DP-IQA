import torch.nn as nn
import torch

class DynamicMarginRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        margin = 0.25*targets.std()

        total_loss = 0.0
        num_pairs = 0

        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                target_diff = targets[i] - targets[j]
                target_sign = torch.sign(target_diff)

                pred_diff = preds[i] - preds[j]

                loss = torch.clamp(-target_sign * (pred_diff) + margin, min=0)
                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs