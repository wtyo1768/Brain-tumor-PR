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
from monai import transforms 
from skimage.feature import texture
import glob


def normalize(df, col):
    df[col] =  (df[col] - df[col].mean() ) /  df[col].std()
    return df 


def numerical_loader(drop_name=True):
    df = pd.read_excel(xls_file, sheet_name='Sheet2')
    if drop_name:
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
    for c in ['ADC tumor', 'Maximal diameter', 'x', 'y', 'z' ]:
        df = normalize(df, c)
  
    scaler = MinMaxScaler()
    minmax_col = [
        'Age',
    ]
    df[minmax_col] = scaler.fit_transform(df[minmax_col])
    df = pd.get_dummies(df,prefix=['Special histology'], columns = ['Special histology (0:meningothelial 1:fibroblastic 2: angiomatous 3:transitional (mixed) 4:psammoma 5: microcystic 6: metaplastic'])
    df = pd.get_dummies(df,prefix=['Simpson grade resection'], columns = ['Simpson grade resection (1-5)'])

    df = df.fillna(df.mean(0))
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

    used_col = origin_columns[masked_columns]
    for i, c in enumerate(used_col): print(i,'|', c)
    print('-------------------------')
    # df = SelectPercentile(mutual_info_classif, percentile=70).fit_transform(df, y_df)
    ## VIS
    # mask = [  True, True,  True,  True,  True, False, False,  True, False,  True,  True,  True,  True,
    #             True,  True, False,  True]
    # df = df[:, mask]
    return df, y_df


def img_features(dtype, return_df=False):
    def glcm_features(f):
        img = cv2.imread(f, 0)
        g = texture.greycomatrix(img, [1, 2, 3, 4, 5, 6, 7], [0, np.pi/2], levels=256, normed=True, symmetric=True)
        features = []
        for p in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            f = texture.greycoprops(g, p)
            features.append(f)
        return np.array(features).flatten() 

    rawX, y = [], []

    PR = glob.glob(f'{segmented_img_dir}/PR/T1c/*.jpg')
    non_PR = glob.glob(f'{segmented_img_dir}/non_PR/T1c/*.jpg')
    y = [1]*len(PR) + [0]*len(non_PR)
    ALL_PR = PR + non_PR
    for f in ALL_PR:
        mri_features = []
        for d in dtype:
            textual_feature = glcm_features(f.replace('T1c', d))
            # print(textual_feature.shape)
            mri_features.append(textual_feature)
         
        rawX.append(np.array(mri_features).flatten())
    # print(rawX[0].shape)       
    thr = .8
    
    X = VarianceThreshold(threshold=(thr * (1 - thr))).fit_transform(np.array(rawX))
    rawX = X.tolist()
    if return_df:
        rawX = pd.DataFrame({'fname':ALL_PR, 'img_feature':rawX})
        return rawX, np.array(y)
    
    return np.array(rawX), np.array(y)


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
        top =   (self.img_size - h) // 2
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
    RandomApply(
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        p=0.8
        # p = 0.5
    ),
    T.RandomGrayscale(p=0.2),
    # T.RandomAffine(),
    RandomApply(
        T.GaussianBlur((3, 3), (1.0, 2.0)),
        p = 0.2
    ),
    RandomApply(crop(.8), p = 0.5),
    T.RandomRotation(2.8),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    
    ImgPad(IMAGE_SIZE),  
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    # T.Normalize(mean=ele[0], std=ele[1]),
]
test_aug = lambda ele:[     
    ImgPad(IMAGE_SIZE), 
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    T.ToTensor(), 
    # T.Normalize(mean=ele[0], std=ele[1]),
]

data_pipe = {   
    # 'train': lambda stat:T.Compose(train_aug(stat)),
    # 'eval' : lambda stat:T.Compose(test_aug(stat)),
    'train' : transforms.Compose([
        transforms.LoadImage(image_only=True, reader='pilreader'),
        transforms.AddChannel(),
        transforms.RepeatChannel(3),
        # transforms.HistogramNormalize(),
        transforms.ScaleIntensity(),
        transforms.RandGaussianNoise(prob=0.4),
        transforms.Affine(
            rotate_params=np.pi/4, scale_params=(1.2, 1.2),
            translate_params=(5, 5), padding_mode='zeros', image_only=True
        ),
        transforms.RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        transforms.RandFlip(spatial_axis=0, prob=0.5),
        transforms.RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        
        transforms.SpatialPad(spatial_size=[IMAGE_SIZE, IMAGE_SIZE, ]),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE, ]),
        transforms.EnsureType(),
    ]),
    'eval' : transforms.Compose([
        transforms.LoadImage(image_only=True, reader='pilreader'),
        transforms.AddChannel(),
        transforms.RepeatChannel(3),
        transforms.ScaleIntensity(),
        
        transforms.SpatialPad(spatial_size=[IMAGE_SIZE, IMAGE_SIZE, ]),
        transforms.Resize([IMAGE_SIZE, IMAGE_SIZE,]),
        transforms.EnsureType(),
    ]),
}
stat = {
    'T1':   [[.1308]*3, [.1031]*3],
    'T1c':  [[.2381]*3, [.2496]*3],
    'T2':   [[.1813]*3, [.1712]*3],
    'Flair':[[.2105]*3, [.1901]*3],
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
                # 'img' : Image.open(f).convert('RGB'),
                "img" : f,
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
        # example['img'] = self.T(stat[self.dtype])(example['img'])
        example['img'] = self.T(example['img'])
        # example = self.T(example)

        return example
    


if __name__ == '__main__':
    df = pd.read_excel(xls_file, sheet_name='Sheet2')
    # ['T1', 'T1c', 'T2', 'Flair']
    #TODO Fix T1c to T1
    print(PR_Dataset(df, 'T1c').__getitem__(1)['img'].size())