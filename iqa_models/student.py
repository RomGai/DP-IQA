import torch
from torch import nn
from torchvision import models


class StudentEfficientNet(nn.Module):
    def __init__(self):
        super(StudentEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b7(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.base_model.avgpool = nn.Identity()

        self.conv_transpose = nn.ConvTranspose2d(in_channels=num_features, out_channels=8, kernel_size=3, stride=2,
                                                 padding=1, output_padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.base_model(x)
        bs=x.shape[0]
        x = x.view(bs, 2560, 16, 16)
        x = torch.relu(self.conv_transpose(x))
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode='nearest')
        feature_map = x
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        score = self.fc3(x)
        return score, feature_map
