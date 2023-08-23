import pandas as pd

def refactor_endo(origin_val, predicted_val):
    origin_df = pd.read_csv(origin_val)
    predicted_df = pd.read_csv(predicted_val)
    out_list = []
    for i in range(origin_df.shape[0]):
        get_image = origin_df.iloc[i, 1]
        row_df = predicted_df.loc[predicted_df['Path(Trial/Subject/Image Name)'] == get_image]
        # print(row_df.head())
        ulcer = row_df.iloc[0, 1]
        erosion = row_df.iloc[0, 2]
        polyp = row_df.iloc[0, 3]
        tumor = row_df.iloc[0, 4]
        # print(ulcer, erosion, polyp, tumor)
        # if i == 3: break
        row = [get_image, ulcer, erosion, polyp, tumor]
        out_list.append(row)
    return out_list

def refactor_chest(origin_val, predicted_val):
    origin_df = pd.read_csv(origin_val)
    predicted_df = pd.read_csv(predicted_val)
    out_list = []
    for i in range(origin_df.shape[0]):
        get_image = origin_df.iloc[i, 1]
        row_df = predicted_df.loc[predicted_df['Path (Trial/Subject/Image Name)'] == get_image]
        pleural_effusion = row_df.iloc[0, 1]
        nodule = row_df.iloc[0, 2]
        pneumonia = row_df.iloc[0, 3]
        cardiomegaly = row_df.iloc[0, 4]
        hilar_enlargement = row_df.iloc[0, 5]
        fracture_old = row_df.iloc[0, 6]
        fibrosis = row_df.iloc[0, 7]
        aortic_calcification = row_df.iloc[0, 8]
        tortuous_aorta = row_df.iloc[0, 9]
        thickened_pleura = row_df.iloc[0, 10]
        TB = row_df.iloc[0, 11]
        pneumothorax = row_df.iloc[0, 12]
        emphysema = row_df.iloc[0, 13]
        atelectasis = row_df.iloc[0, 14]
        calcification = row_df.iloc[0, 15]
        pulmonary_edema = row_df.iloc[0, 16]
        increased_lung_markings = row_df.iloc[0, 17]
        elevated_diaphragm = row_df.iloc[0, 18]	
        consolidation = row_df.iloc[0, 19]

        row = [get_image, pleural_effusion, nodule, pneumonia, cardiomegaly, hilar_enlargement, fracture_old, fibrosis, aortic_calcification, tortuous_aorta, thickened_pleura, TB, pneumothorax, emphysema, atelectasis, calcification, pulmonary_edema, increased_lung_markings, elevated_diaphragm, consolidation]
        out_list.append(row)
    return out_list

if __name__ == "__main__":
    # for
    # df = pd.DataFrame(prediction)
    # df.to_csv()
    # endo_list = refactor_endo('endo_val.csv', 'prediction_Endo_MedFM_10.csv')
    # df = pd.DataFrame(endo_list)
    # df.to_csv('endo_10-shot_submission.csv', index=False)
    chest_list = refactor_chest('chest_val.csv', 'prediction_Chest_MedFM_10.csv')
    df = pd.DataFrame(chest_list)
    df.to_csv('chest_10-shot_submission.csv', index=False)