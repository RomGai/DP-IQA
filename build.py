import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import sys


def build_optmizer(name,model):
    if(name=="koniq"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.2)
        batch_size = 16
        return optimizer, scheduler, batch_size
    elif(name=="clive"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.2)
        batch_size = 16
        return optimizer, scheduler, batch_size
    elif(name=="livefb"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.2)
        batch_size = 16
        return optimizer, scheduler, batch_size
    elif(name=="spaq"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.2)
        batch_size = 16
        return optimizer, scheduler, batch_size
    elif(name=="kadid"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.2)
        batch_size = 14
        return optimizer, scheduler, batch_size
    elif (name == "live"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[2], gamma=0.2)
        batch_size = 12
        return optimizer, scheduler, batch_size
    elif (name == "tid2013"):
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.2)
        batch_size = 12
        return optimizer, scheduler, batch_size
    else:
        print("wrong dataset type")
        sys.exit(1)

def build_student_optmizer(name,model):
    if(name=="koniq"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif(name=="clive"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif(name=="livefb"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[4], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif(name=="spaq"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[6], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif(name=="kadid"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[5,10], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif (name == "live"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    elif (name == "tid2013"):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[3,10], gamma=0.2)
        batch_size = 24
        return optimizer, scheduler, batch_size
    else:
        print("wrong dataset type")
        sys.exit(1)
