import os 
import cv2
import glob
from cfg import *


## Check all type of MRI image has the same size
dtype = ['T1', 'T1c', 'Flair', 'T2']

for patient_class in ['PR', 'non_PR']: 

    _dir = f'{voc_data_path}/{patient_class}/T1/JPEGImages'
    assert(os.path.isdir(_dir))
    for f in glob.glob(f'{_dir}/*.jpg'):
            
        img_path = [f] +  [f.replace('T1', d) for d in dtype[1:]]
        for p in img_path: assert(os.path.isfile(p))
        
        shape = [cv2.imread(p).shape for p in img_path]

        s = shape[0]
        for sh in shape[1:]:
            if not sh==s:
                print('Different image shape for :', f)
                break