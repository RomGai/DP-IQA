from iqa_models.dpiqa import DPIQA
from datasets_in_the_wild import load_dataset
from build import build_optmizer
from magrin import DynamicMarginRankingLoss
from metrics import PLCC,SRCC
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
from torch.cuda.amp import autocast, GradScaler

dataset_name="koniq" #koniq, clive, livefb, spaq

device="cuda"
scaler = GradScaler()

model=DPIQA()
model.to(device)
criterion = nn.MSELoss().cuda()
criterion_rank=DynamicMarginRankingLoss()
optimizer, scheduler, batch_size=build_optmizer(dataset_name,model)
num_epochs=30

save_indices=True#For saving, there are default names. For loading, you need to manually enter the path.
load_indices=False #Turn to 'True' to load the training or testing indices from the saved file.
train_indices_path=None #i.e 'koniq_train_indices.pth'
test_indices_path=None #i.e 'koniq_test_indices.pth'

train_dataset,test_dataset=load_dataset(dataset_name,save_indices=save_indices,
                                        load_indices=load_indices,
                                        train_indices_path=train_indices_path,
                                        test_indices_path=test_indices_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_max_plcc=0
test_max_srcc=0
train_max_plcc=0
train_max_srcc=0

#By default, validation is performed at the end of each epoch. However, it is recommended to validate every 250 steps.

for epoch in range(num_epochs):
    print(str(epoch+1)+'/'+str(num_epochs))
    loss_sum=0.0
    test_loss_sum=0.0

    model.train()
    pos=0
    for jpeg_images, gt_scores in train_dataloader:
        pos = pos + 1
        jpeg_images=jpeg_images.to(device)
        bsz = jpeg_images.shape[0]
        gt_scores = gt_scores.view(bsz, 1).to(device)

        if((pos%50)==0):
            print('epoch'+str(epoch+1)+'batch:'+str(pos)+'/'+str(len(train_dataloader)))

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            preds = model(jpeg_images)
            loss_qu = criterion(preds, gt_scores)
            loss_rk = criterion_rank(preds.squeeze(), gt_scores.squeeze())
            loss = loss_qu#+loss_rk

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if(pos==1):
            train_all_preds=preds
            train_all_gt_scores=gt_scores
        else:
            train_all_preds = torch.cat([train_all_preds, preds], dim=0)
            train_all_gt_scores = torch.cat([train_all_gt_scores, gt_scores], dim=0)

        torch.cuda.empty_cache()
        loss_sum = loss_sum+ loss.item()

    train_plcc = PLCC(train_all_preds, train_all_gt_scores)
    train_srcc = SRCC(train_all_preds, train_all_gt_scores)

    print("train plcc:" + str(train_plcc.item())+"  srocc:"+str(train_srcc))
    print("epoch_"+str(epoch + 1) + "_mean train loss:" + str(loss_sum / len(train_dataloader)))

    model.eval()
    pos=0
    for jpeg_images, gt_scores in test_dataloader:
        pos = pos + 1
        if((pos%50)==0):
            print('epoch'+str(epoch+1)+'batch:'+str(pos)+'/'+str(len(test_dataloader)))
        with torch.no_grad():
            jpeg_images = jpeg_images.to(device)
            bsz = jpeg_images.shape[0]
            gt_scores = gt_scores.view(bsz, 1).to(device)

            test_preds = model(jpeg_images)

            test_loss_qu = criterion(test_preds, gt_scores)
            test_loss_rk = criterion_rank(test_preds.squeeze(), gt_scores.squeeze())
            test_loss = test_loss_qu#+test_loss_rk

            if (pos == 1):
                test_all_preds = test_preds
                test_all_gt_scores = gt_scores
            else:
                test_all_preds = torch.cat([test_all_preds, test_preds], dim=0)
                test_all_gt_scores = torch.cat([test_all_gt_scores, gt_scores], dim=0)

            test_loss_sum += test_loss.item()
            del jpeg_images, test_preds, gt_scores, test_loss
            gc.collect()
            torch.cuda.empty_cache()

    test_plcc = PLCC(test_all_preds, test_all_gt_scores)
    test_srcc=SRCC(test_all_preds,test_all_gt_scores)

    if(test_plcc>=test_max_plcc):
        test_max_plcc=test_plcc
        torch.save(model.state_dict(), dataset_name+'_model_plcc_best.pth')

    if(test_srcc>=test_max_srcc):
        test_max_srcc=test_srcc
        torch.save(model.state_dict(), dataset_name+'_model_srcc_best.pth')

    print("test plcc:"+str(test_plcc.item())+"  srocc:"+str(test_srcc))

    print("epoch_"+str(epoch + 1) + "_mean test loss:" + str(test_loss_sum / len(test_dataloader)))

    print("The max test PLCC is "+str(test_max_plcc)+", the max test SRCC is "+str(test_max_srcc))

    scheduler.step()
