import torch
from torch.nn import ModuleList
from utils.utils import AverageMeter,warmup_learning_rate, accuracy, save_model
import sys
import time
import numpy as np
from config.config_linear_competition import parse_option
from utils.utils_competition import set_loader_competition, set_model_competition_first, set_optimizer, adjust_learning_rate, accuracy_multilabel
from sklearn.metrics import average_precision_score,roc_auc_score, classification_report
import pandas as pd
from tqdm import tqdm

import os

def evaluate(dataloader, model, criterion, opt):
    print("BEGIN EVALUATION ...")
    model.eval()
    device = opt.device
    accuracy = 0
    # roc_auc = 0
    # precision = 0
    # recall = 0
    len_data = 0
    output_list = []
    label_list = []
    losses = AverageMeter()
    for idx, (image, label_tensor) in tqdm(enumerate(dataloader)):
        images = image.to(device) 
        labels = label_tensor.float() 
        bsz = labels.shape[0] 
        len_data += bsz
        labels=labels.to(device)
        with torch.cuda.amp.autocast(enabled=opt.amp):
            output = model(images)
            loss = criterion(output, labels)

            output = torch.round(torch.sigmoid(output))
        accuracy += (output == labels).float().sum()
        output_list.append(output.squeeze().detach().cpu().numpy())
        label_list.append(labels.squeeze().detach().cpu().numpy())
        
        losses.update(loss.item(), bsz)

    accuracy = (accuracy / (len_data * opt.num_class))
    roc_auc = roc_auc_score(label_list, output_list, average = 'macro')
    print("ROC AUC SCORE: ", roc_auc)
    return losses.avg, accuracy, roc_auc

def train_MedFM_multilabel(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    # model.eval()
    # classifier.train()
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    # print(train_loader[0])
    for idx, (image, type_tensor) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)

        images = image.to(device)

        labels = type_tensor
        labels = labels.float()
        bsz = labels.shape[0]
        labels=labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(enabled=opt.amp):
            output = model(images)
            loss = criterion(output, labels)
            
        # update metric
        losses.update(loss.item(), bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                epoch, idx + 1, len(train_loader)))
            sys.stdout.flush()
    loss_eval, acc_eval, roc_auc = evaluate(train_loader, model, criterion, opt)
    return losses.avg, acc_eval

def validate_multilabel(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    # classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, type_tensor) in enumerate(val_loader):
            images = image.float().to(device)

            labels = type_tensor
            labels = labels.float()
            label_list.append(labels.squeeze().detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            # output = classifier(model.encoder(images))

            # compute output
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(images)
                loss = criterion(output, labels)
                output = torch.round(torch.sigmoid(output))

            out_list.append(output.squeeze().detach().cpu().numpy())
            # update metric
            # losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    label_array = np.array(label_list)
    out_array = np.array(out_list)
    # out_array = np.concatenate(out_list, axis=0)n
    # r = roc_auc_score(label_array, out_array, average='macro')


    return losses.avg, r



def test_multilabel(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    # classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, img_name) in tqdm(enumerate(val_loader)):
            images = image.float().to(device)

            # print(f"idx:{idx}")

            # compute output
            with torch.cuda.amp.autocast(enabled=opt.amp):
                output = model(images)
                # output = torch.round(torch.sigmoid(output))
                output = torch.sigmoid(output)


            
            output = output.squeeze().detach().cpu().numpy().tolist()
            if opt.dataset == 'Colon_MedFM':
                output = [output, 1 - output]
            row = img_name + output
            # print(f"row:{row}")
            out_list.append(row)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    # out_array = np.array(out_list)
    # out_array = np.concatenate(out_list, axis=0)
    return out_list

def main_multilabel_competition():
    best_acc = 0
    opt = parse_option()

    # build data loader
    device = opt.device
    train_loader, val_loader, test_loader = set_loader_competition(opt)

    prediction = []
    # training routine
    for i in range(0,1):
        model, criterion = set_model_competition_first(opt)

        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_MedFM_multilabel(train_loader, model, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}, loss:{:.2f}'.format(
                epoch, time2 - time1, acc, loss))
            
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_{dataset}_epoch_{epoch}_nshot_{nshot}.pth'.format(dataset=opt.dataset, epoch=epoch, nshot=opt.nshot))
                save_model(model, optimizer, opt, epoch, save_file)

        if val_loader:
            loss_val, acc_val, roc_auc_val = evaluate(val_loader, model, criterion, opt)
            print("ACCURACY VALIDATION: ", acc_val)
            print("LOSS VALIDATION: ", loss_val)
        out_list = test_multilabel(test_loader, model, criterion, opt)
        prediction = out_list
    type_data = None
    if opt.dataset == "Chest_MedFM":
        df = pd.DataFrame(prediction, columns=['Path (Trial/Subject/Image Name)',
                                               'pleural_effusion', 'nodule', 'pneumonia', 'cardiomegaly', 'hilar_enlargement', 'fracture_old', 'fibrosis', 'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB', 'pneumothorax', 'emphysema', 'atelectasis', 'calcification', 'pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm', 'consolidation'])
        type_data = "chest"
    elif opt.dataset == "Colon_MedFM":
        df = pd.DataFrame(prediction, columns=['Path(Trial/Subject/Image Name)',
                                               'non_tumor', 'tumor'])
        type_data = "colon"
    elif opt.dataset == "Endo_MedFM":
        df = pd.DataFrame(prediction, columns=['Path(Trial/Subject/Image Name)',
                                               'ulcer', 'erosion', 'polyp', 'tumor'])
        type_data = "endo"
    else:
        raise ValueError(f"{opt.dataset} is not satisfied (Must in ['Chest_MedFM', 'Colon_MedFM', 'Endo_MedFM'])")
    # Lưu DataFrame thành tệp CSV
    # df.to_csv(f'./prediction_{opt.dataset}_{opt.nshot}.csv', index=False)
    
    df.to_csv(f'/content/drive/MyDrive/submit_MedFM/{type_data}/{type_data}_{opt.nshot}-shot_submission.csv', index=False)