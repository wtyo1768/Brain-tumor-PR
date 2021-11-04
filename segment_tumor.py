import os 
import numpy as np
from cfg import *
import cv2


output_dir = segmented_data_path
MRI_TYPE = ['T1c']


for dtype in MRI_TYPE:
    
    for patient_class in ['PR', 'non_PR']: 
        outdir = f'{output_dir}/{patient_class}/{dtype}'
        if not os.path.isdir(outdir): os.makedirs(outdir)

        _dir = f'{voc_data_path}/{patient_class}/{dtype}/JPEGImages'
        for dirPath, dirNames, fileNames in os.walk(_dir):
            for f in fileNames:
                img_path = os.path.join(dirPath, f)
                assert(os.path.isfile(img_path))
                mask_path = img_path.replace('JPEGImages', 'SegmentationClassPNG').replace('.jpg', '.png')
                img = cv2.imread(img_path, 0)
                
                
                # print(img[:,:,0].mean(), img[:,:,1].mean())
                ori_mask = cv2.imread(mask_path, 0)
                # print(img.shape, ori_mask.shape)
                
                mask = np.array(ori_mask>0, dtype=np.int)
                # mask = np.expand_dims(mask, axis=2)
                # mask = np.concatenate([mask, mask, mask], axis=2)
                segmented_img = img * mask
                # print(f)
                cv2.imwrite(f'{output_dir}/{patient_class}/{dtype}/{f}', segmented_img)