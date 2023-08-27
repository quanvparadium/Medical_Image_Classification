import torch
from config.config_linear_competition import parse_option
from utils.utils_competition import set_loader_competition, set_model_competition_first
import pandas as pd
from tqdm import tqdm
import time
from utils.utils import AverageMeter,warmup_learning_rate, accuracy, save_model


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

def infer():
    opt = parse_option()

    device = opt.device
    train_loader, val_loader, test_loader = set_loader_competition(opt)

    prediction = []
    for i in range(0, 1):
        model, criterion = set_model_competition_first(opt)

        out_list = test_multilabel(test_loader, model, criterion, opt)
        prediction = out_list

    if opt.dataset == "Chest_MedFM":
        df = pd.DataFrame(prediction, columns=['Path (Trial/Subject/Image Name)',
                                               'pleural_effusion', 'nodule', 'pneumonia', 'cardiomegaly', 'hilar_enlargement', 'fracture_old', 'fibrosis', 'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB', 'pneumothorax', 'emphysema', 'atelectasis', 'calcification', 'pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm', 'consolidation'])
    elif opt.dataset == "Colon_MedFM":
        df = pd.DataFrame(prediction, columns=['Path(Trial/Subject/Image Name)',
                                               'non_tumor', 'tumor'])
    elif opt.dataset == "Endo_MedFM":
        df = pd.DataFrame(prediction, columns=['Path(Trial/Subject/Image Name)',
                                               'ulcer', 'erosion', 'polyp', 'tumor'])
    else:
        raise ValueError(f"{opt.dataset} is not satisfied (Must in ['Chest_MedFM', 'Colon_MedFM', 'Endo_MedFM'])")
    # Lưu DataFrame thành tệp CSV
    # df.to_csv(f'./prediction_{opt.dataset}_{opt.nshot}.csv', index=False)
    df.to_csv(f'/content/drive/MyDrive/prediction_{opt.dataset}_{opt.nshot}.csv', index=False)

if __name__ == '__main__':
    infer()