import os 
import numpy as np
import cv2
from cfg import *
import glob

MRI_TYPE = ['T1', 'T1c', 'T2', 'Flair']
output_dir = segmented_img_dir


def segment_black_region(segmented, original):  
    # if :
    #     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # else: gray = img
    
    pts = np.argwhere(segmented>0)
    x1,y1 = pts.min(axis=0)
    x2,y2 = pts.max(axis=0)
    
    if NO_SEGMENTED: return original[x1:x2+1, y1:y2+1]
    else : return segmented[x1:x2+1, y1:y2+1]


#TODO fix to glob
for dtype in MRI_TYPE:
    for patient_class in ['PR', 'non_PR']: 
        outdir = f'{output_dir}/{patient_class}/{dtype}'
        if not os.path.isdir(outdir): os.makedirs(outdir)

        _dir = f'{voc_data_path}/{patient_class}/{dtype}/JPEGImages'
        assert(os.path.isdir(_dir))

        for img_path in glob.glob(f'{_dir}/*.jpg'):
            
            mask_path = img_path.replace('JPEGImages', 'SegmentationClassPNG').replace('.jpg', '.png')
            print(img_path, mask_path)

            assert os.path.isfile(img_path)
            assert os.path.isfile(mask_path)
            
            img = cv2.imread(img_path, 0)              
            ori_mask = cv2.imread(mask_path, 0)
            
            ori_mask = cv2.resize(ori_mask, [img.shape[1], img.shape[0]])
            # print(ori_mask.shape)
            mask = np.array(ori_mask>0, dtype=np.int)
            segmented_img = img * mask
            segmented_img = segment_black_region(segmented_img, img)
            f = img_path.split('/')[-1]
            cv2.imwrite(f'{output_dir}/{patient_class}/{dtype}/{f}', segmented_img)