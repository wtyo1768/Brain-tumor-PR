import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from PIL import Image
from torchvision import transforms as T
import random
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from cfg import *
from sklearn.feature_selection import VarianceThreshold


def normalize(df, col):
    df[col] =  (df[col] - df[col].mean() ) /  df[col].std()
    return df 


def numerical_loader():
    df = pd.read_excel(xls_file, sheet_name='Sheet2')
    df = df.drop(columns=df.columns[0])
    df = df.drop(columns=[
        'T1+C (srs/img)', 'T1 (srs/img)', 'T2 (srs/img)', 
        'FLAIR (srs/img)', '1st MRI', 'OP date',	
        'The Latest MRI ', 'Date of MRI with P/R ',
        '1st MRI.1',
        # 'x', 'y', 'z', 
        'WHO grade 1.2.3 (benign, atypical, malignant)', '> 1y F/U MRI after OP (Y:1 N:0)', 
        'Location 1: Skull base(SB), 2: PSPF, 3: Convexity, 4: Others',
        'F/U time (month)', 'PR time (months)',
        'R/T before P/R (Y:1 N:0) ',
        '(1) Discovery 750 (2) Signa HD (3) Avanto (4) Symphony tim (5) Aera (6) Others'
    ])
    for c in ['ADC tumor', 'Maximal diameter', 'x', 'y', 'z',   ]:
        df = normalize(df, c)
    
    scaler = MinMaxScaler()
    minmax_col = [
        # 'PR time (months)', 'F/U time (month)', 
        'Age',
    ]
    df[minmax_col] = scaler.fit_transform(df[minmax_col])
    df = pd.get_dummies(df,prefix=['Special histology'], columns = ['Special histology (0:meningothelial 1:fibroblastic 2: angiomatous 3:transitional (mixed) 4:psammoma 5: microcystic 6: metaplastic'])
    df = pd.get_dummies(df,prefix=['Simpson grade resection'], columns = ['Simpson grade resection (1-5)'])

    

    df = df.fillna(df.mean())
    y_df = df.pop('Progression/Recurrence (Yes:1 No:0)')
    y_df = y_df.astype('int32')

    origin_columns = df.columns
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    df = sel.fit_transform(df)
    masked_columns = sel._get_support_mask()

    print(origin_columns)
    print(masked_columns.shape, origin_columns.shape)
    print('Number of Patients and feature :', df.shape[0], df.shape[1], )
    print('Number of Case 1 and Case 2 :', y_df[y_df==0].shape[0],  y_df[y_df==1].shape[0])
    print('-------------------------')
    print("Using feature:")
    for i, c in enumerate(origin_columns): 
        if masked_columns[i]: print(i,'|', c)
    print('-------------------------')
    
    return df, y_df


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class ImgPad(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
    def forward(self, img):
        w, h = img.size
        left =  (self.img_size - w) // 2
        right =  self.img_size - w - left
        top =   (self.img_size - h) //2
        bottom = self.img_size - h - top
        left, top = max(left, 0), max(top, 0) 
        right, bottom = max(right, 0), max(bottom, 0)
        im = T.Pad(padding=(left, top, right, bottom))(img)
        return im


class crop(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
    def forward(self, img):
        ratio = random.uniform(self.ratio, 1)
        w, h = img.size
        return T.RandomCrop((int(h*ratio), int(w*ratio)))(img)


train_aug = lambda ele:[
    # RandomApply(
    #     T.ColorJitter(0.8, 0.8, 0.8, 0.2),
    #     p=0.8
    #     # p = 0.5
    # ),
    # T.RandomGrayscale(p=0.2),
    # RandomApply(
    #     T.GaussianBlur((3, 3), (1.0, 2.0)),
    #     p = 0.2
    # ),
    RandomApply(crop(.75), p = 0.5),
    T.RandomRotation(2.8),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    
    ImgPad(IMAGE_SIZE),  
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=ele[0], std=ele[1]),
]
test_aug = lambda ele:[     
    ImgPad(IMAGE_SIZE), 
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    T.ToTensor(), 
    T.Normalize(mean=ele[0], std=ele[1]),
]
data_pipe = {
    'train': lambda stat:T.Compose(train_aug(stat)),
    'eval' : lambda stat:T.Compose(test_aug(stat)),
}
stat = {
    'T1':   [[.1302]*3, [.1656]*3],
    'T1c':  [[.1038]*3, [.1031]*3],
    'T2':   [[.1329]*3, [.1781]*3],
    'Flair':[[.1453]*3, [.1869]*3],
}


class PR_Dataset(Dataset):
    """
    Args:
        dtype: MRI type of image
        eval_mode : Specify eval mode to disabled the data augmentation
    """
    def __init__(self, df, dtype, eval_mode=False):
        super().__init__()
        self.df = df    
        self.dtype = dtype    
        self.T = data_pipe['eval'] if eval_mode else data_pipe['train']
        
        if dtype not in ['T1', 'T1c', 'T2', 'Flair', 'Mixed']: 
            raise ValueError('Wrong MRI type')
        
        self.features = []  
        for i in range(self.df.shape[0]):
            row = self.df.iloc[i]
            patient_id = str(row["病歷號 (yellow in WHO grade I file) "])
        
            if patient_id in ['16962871']: continue
            label = row["Progression/Recurrence (Yes:1 No:0)"]
            pr_class = 'PR' if label else 'non_PR'

            f = f'{segmented_img_dir}/{pr_class}/{dtype}/{patient_id}.jpg'
            assert(os.path.isfile(f))

            self.features.append({
                'img' : Image.open(f).convert('RGB'),
                'label' : torch.tensor(label),
            })

        self.class_weight  = np.unique(df["Progression/Recurrence (Yes:1 No:0)"].values.tolist(), return_counts=True)[1]
        self.class_weight = 1 / (self.class_weight / df.shape[0]*2)
        print(self.class_weight)
        print(f'{self.dtype} | {len(self.features)} images found')


    def __len__(self):
        return len(self.features)


    def __getitem__(self, index):
        example = self.features[index].copy()
        #TODO add 4 type of MRI to here
        example['img'] = self.T(stat[self.dtype])(example['img'])
        return example
    


if __name__ == '__main__':
    df = pd.read_excel(xls_file, sheet_name='Sheet2')
    # ['T1', 'T1c', 'T2', 'Flair']
    #TODO Fix T1c to T1
    print(PR_Dataset(df, 'T1c').__getitem__(1)['img'].size())